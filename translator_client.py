from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Iterable

import httpx


class TranslationError(RuntimeError):
    pass


@dataclass(frozen=True)
class OpenAICompatConfig:
    api_base: str
    api_key: str | None
    model: str
    timeout_s: float = 120.0
    max_retries: int = 5


def _env(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    return v.strip()


def load_openai_compat_config(
    *,
    api_base: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
) -> OpenAICompatConfig:
    api_base = (api_base or _env("OPENAI_API_BASE") or _env("OPENAI_BASE_URL") or "https://api.openai.com").rstrip(
        "/"
    )
    api_key = api_key or _env("OPENAI_API_KEY")
    model = model or _env("OPENAI_MODEL") or "gpt-4.1-mini"
    return OpenAICompatConfig(api_base=api_base, api_key=api_key, model=model)


def _build_messages(
    *,
    source_lang: str,
    target_lang: str,
    lines: list[dict[str, str]],
    prev_context: str | None,
    next_context: str | None,
    glossary: dict[str, str] | None,
) -> list[dict[str, str]]:
    glossary = glossary or {}
    glossary_lines = "\n".join([f"- {k} => {v}" for k, v in glossary.items()]) if glossary else "None"

    system = (
        "You are a professional translator.\n"
        "Translate faithfully and naturally.\n"
        "Do not omit any lines. Do not merge or split lines.\n"
        "Keep numbers, dates, codes, URLs, emails, and punctuation.\n"
        "Return ONLY valid JSON.\n"
        'Output schema: [{"line_id": "...", "translated": "..."}]\n'
    )

    user_parts: list[str] = []
    user_parts.append(f"Source language: {source_lang}")
    user_parts.append(f"Target language: {target_lang}")
    user_parts.append("Terminology/glossary (must follow exactly where applicable):")
    user_parts.append(glossary_lines)
    user_parts.append("")
    if prev_context:
        user_parts.append("Previous context (do NOT translate, for reference only):")
        user_parts.append(prev_context)
        user_parts.append("")
    if next_context:
        user_parts.append("Next context (do NOT translate, for reference only):")
        user_parts.append(next_context)
        user_parts.append("")

    user_parts.append("Translate these lines. Keep 1 output item per input line_id, same order:")
    user_parts.append(json.dumps(lines, ensure_ascii=False))

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(user_parts)},
    ]


def _parse_json_array(text: str) -> list[dict[str, Any]]:
    text = text.strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        raise TranslationError(f"Model returned non-JSON output: {e}") from e
    if not isinstance(obj, list):
        raise TranslationError("Model JSON was not a list.")
    return obj  # type: ignore[return-value]


def translate_block_lines(
    *,
    cfg: OpenAICompatConfig,
    source_lang: str,
    target_lang: str,
    lines: list[dict[str, str]],
    prev_context: str | None = None,
    next_context: str | None = None,
    glossary: dict[str, str] | None = None,
) -> dict[str, str]:
    if not lines:
        return {}

    url = f"{cfg.api_base}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"

    payload: dict[str, Any] = {
        "model": cfg.model,
        "messages": _build_messages(
            source_lang=source_lang,
            target_lang=target_lang,
            lines=lines,
            prev_context=prev_context,
            next_context=next_context,
            glossary=glossary,
        ),
        "temperature": 0.2,
    }

    timeout = httpx.Timeout(cfg.timeout_s)
    last_err: Exception | None = None

    with httpx.Client(timeout=timeout) as client:
        for attempt in range(cfg.max_retries):
            try:
                r = client.post(url, headers=headers, json=payload)
                if r.status_code >= 400:
                    raise TranslationError(f"HTTP {r.status_code}: {r.text[:500]}")
                data = r.json()
                content = data["choices"][0]["message"]["content"]
                arr = _parse_json_array(content)

                mapping: dict[str, str] = {}
                for item in arr:
                    if not isinstance(item, dict):
                        continue
                    lid = item.get("line_id")
                    tr = item.get("translated")
                    if isinstance(lid, str) and isinstance(tr, str):
                        mapping[lid] = tr

                missing = [x["line_id"] for x in lines if x.get("line_id") not in mapping]
                if missing:
                    raise TranslationError(f"Missing translations for {len(missing)} line(s).")
                return mapping
            except Exception as e:
                last_err = e
                sleep_s = min(10.0, (2**attempt) + random.random())
                time.sleep(sleep_s)

    raise TranslationError(f"Translation failed after retries: {last_err}")


def build_in_run_glossary(texts: Iterable[str]) -> dict[str, str]:
    """
    Lightweight, in-run glossary: preserve tokens that should not be translated.
    We map token -> same token to enforce pass-through.
    """
    keep: set[str] = set()
    for t in texts:
        for token in _extract_keep_tokens(t):
            keep.add(token)
    return {k: k for k in sorted(keep)}


def _extract_keep_tokens(text: str) -> list[str]:
    # Keep obvious non-translatable tokens: URLs/emails and long alphanumerics.
    out: list[str] = []
    for raw in text.replace("\u00a0", " ").split():
        s = raw.strip("()[]{}<>\"'“”‘’.,;:!?")
        if not s:
            continue
        if "@" in s and "." in s:
            out.append(s)
            continue
        if s.startswith("http://") or s.startswith("https://"):
            out.append(s)
            continue
        alnum = sum(ch.isalnum() for ch in s)
        if alnum >= 8 and any(ch.isdigit() for ch in s):
            out.append(s)
            continue
    return out

