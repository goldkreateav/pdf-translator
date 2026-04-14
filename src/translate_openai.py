import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def _translate_one_batch(
    batch_index: int,
    batch: Sequence[TextBlock],
    source_lang: str,
    target_lang: str,
    model: str,
    api_key: str,
    base_url: str | None,
) -> tuple[int, list[str]]:
    source_texts = [block.source_text for block in batch]
    prompt = _build_prompt(source_lang, target_lang, source_texts)
    messages = [
        {
            "role": "system",
            "content": "You are a translation engine. Output valid JSON only.",
        },
        {"role": "user", "content": prompt},
    ]
    client = OpenAI(api_key=api_key, base_url=base_url)

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
    ordered_translations = [str(by_id.get(idx, "")) for idx in range(len(batch))]
    return (batch_index, ordered_translations)


def translate_block_batch(
    blocks: list[TextBlock],
    source_lang: str,
    target_lang: str,
    model: str,
    batch_size: int = 20,
    concurrency: int = 1,
    api_key: str | None = None,
    base_url: str | None = None,
) -> list[str]:
    if not blocks:
        return []
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1.")

    resolved_key = api_key or os.getenv("OPENAI_API_KEY")
    if not resolved_key:
        raise RuntimeError("OPENAI_API_KEY is missing.")

    resolved_base_url = (base_url or os.getenv("OPENAI_BASE_URL") or "").strip() or None
    batches = _chunks(blocks, batch_size)
    translated_by_batch: dict[int, list[str]] = {}

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                _translate_one_batch,
                batch_index,
                batch,
                source_lang,
                target_lang,
                model,
                resolved_key,
                resolved_base_url,
            )
            for batch_index, batch in enumerate(batches)
        ]
        for future in as_completed(futures):
            batch_index, batch_translation = future.result()
            translated_by_batch[batch_index] = batch_translation

    translated: list[str] = []
    for batch_index in range(len(batches)):
        translated.extend(translated_by_batch[batch_index])
    return translated
