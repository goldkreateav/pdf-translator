import json
import os
from typing import Sequence, TypeVar

from openai import OpenAI
from openai import APIError, BadRequestError

from src.models import TextBlock

T = TypeVar("T")


def _chunks(values: Sequence[T], size: int) -> list[Sequence[T]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def _build_prompt(source_lang: str, target_lang: str, texts: list[str]) -> str:
    payload = [{"id": i, "text": text} for i, text in enumerate(texts)]
    return (
        f"Translate from {source_lang} to {target_lang}. "
        "Return strict JSON with key 'translations' as an array of objects {id, text}. "
        "Do not add or reorder items. Preserve numbers, units, punctuation.\n\n"
        f"Input:\n{json.dumps(payload, ensure_ascii=False)}"
    )


def translate_block_batch(
    blocks: list[TextBlock],
    source_lang: str,
    target_lang: str,
    model: str,
    batch_size: int = 20,
    api_key: str | None = None,
    base_url: str | None = None,
) -> list[str]:
    if not blocks:
        return []

    resolved_key = api_key or os.getenv("OPENAI_API_KEY")
    if not resolved_key:
        raise RuntimeError("OPENAI_API_KEY is missing.")

    resolved_base_url = (base_url or os.getenv("OPENAI_BASE_URL") or "").strip() or None
    client = OpenAI(api_key=resolved_key, base_url=resolved_base_url)
    translated: list[str] = []

    for batch in _chunks(blocks, batch_size):
        source_texts = [block.source_text for block in batch]
        prompt = _build_prompt(source_lang, target_lang, source_texts)

        messages = [
            {
                "role": "system",
                "content": "You are a translation engine. Output valid JSON only.",
            },
            {"role": "user", "content": prompt},
        ]

        # Some OpenAI-compatible servers don't support response_format.
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=messages,
                response_format={"type": "json_object"},
            )
        except (BadRequestError, APIError):
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=messages,
            )

        content = response.choices[0].message.content or "{}"
        data = json.loads(content)
        items = data.get("translations", [])
        if len(items) != len(batch):
            raise RuntimeError("Translation alignment error: count mismatch.")

        by_id = {int(item["id"]): item["text"] for item in items}
        for idx in range(len(batch)):
            translated.append(str(by_id.get(idx, "")))

    return translated
