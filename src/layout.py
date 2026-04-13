from collections import defaultdict

from src.models import OCRWord, TextBlock


def _line_sort_key(word: OCRWord) -> tuple[int, int]:
    return (word.word_num, word.left)


def _intersection_area(a: TextBlock, b: TextBlock) -> int:
    left = max(a.left, b.left)
    top = max(a.top, b.top)
    right = min(a.right, b.right)
    bottom = min(a.bottom, b.bottom)
    if right <= left or bottom <= top:
        return 0
    return (right - left) * (bottom - top)


def _block_area(block: TextBlock) -> int:
    return max(1, block.width * block.height)


def _resolve_overlaps(blocks: list[TextBlock], overlap_ratio_threshold: float = 0.35) -> list[TextBlock]:
    """
    Suppress near-duplicate / heavily overlapping blocks.
    Keeps higher-confidence and larger candidates.
    """
    kept: list[TextBlock] = []
    sorted_blocks = sorted(
        blocks,
        key=lambda b: (b.top, b.left, -b.confidence, -_block_area(b)),
    )
    for candidate in sorted_blocks:
        suppressed = False
        for existing in kept:
            overlap = _intersection_area(existing, candidate)
            if overlap <= 0:
                continue
            ratio = overlap / min(_block_area(existing), _block_area(candidate))
            if ratio >= overlap_ratio_threshold:
                existing_score = (existing.confidence, _block_area(existing))
                candidate_score = (candidate.confidence, _block_area(candidate))
                if candidate_score > existing_score:
                    kept.remove(existing)
                    kept.append(candidate)
                suppressed = True
                break
        if not suppressed:
            kept.append(candidate)
    return sorted(kept, key=lambda b: (b.top, b.left))


def words_to_blocks(
    words: list[OCRWord],
    page_index: int,
    granularity: str = "paragraph",
) -> list[TextBlock]:
    """
    Build paragraph-like blocks from Tesseract structure:
    - line key: (block_num, par_num, line_num)
    - block key: (block_num, par_num)
    """
    if not words:
        return []

    lines: dict[tuple[int, int, int], list[OCRWord]] = defaultdict(list)
    for word in words:
        lines[(word.block_num, word.par_num, word.line_num)].append(word)

    block_lines: dict[tuple[int, int] | tuple[int, int, int], list[dict]] = defaultdict(list)
    for (block_num, par_num, line_num), line_words in lines.items():
        sorted_words = sorted(line_words, key=_line_sort_key)
        line_text = " ".join(word.text for word in sorted_words).strip()
        if not line_text:
            continue
        left = min(word.left for word in sorted_words)
        top = min(word.top for word in sorted_words)
        right = max(word.right for word in sorted_words)
        bottom = max(word.bottom for word in sorted_words)
        avg_conf = sum(word.confidence for word in sorted_words) / len(sorted_words)
        avg_height = sum(word.height for word in sorted_words) / len(sorted_words)
        if granularity == "line":
            key: tuple[int, int, int] | tuple[int, int] = (block_num, par_num, line_num)
        else:
            key = (block_num, par_num)
        block_lines[key].append(
            {
                "line_num": line_num,
                "text": line_text,
                "left": left,
                "top": top,
                "right": right,
                "bottom": bottom,
                "confidence": avg_conf,
                "height": avg_height,
            }
        )

    blocks: list[TextBlock] = []
    ordered_block_keys = sorted(
        block_lines.keys(),
        key=lambda key: (
            min(row["top"] for row in block_lines[key]),
            min(row["left"] for row in block_lines[key]),
        ),
    )
    for block_id, block_key in enumerate(ordered_block_keys):
        line_entries = sorted(
            block_lines[block_key],
            key=lambda row: (row["top"], row["left"], row["line_num"]),
        )
        text = "\n".join(row["text"] for row in line_entries)
        left = min(row["left"] for row in line_entries)
        top = min(row["top"] for row in line_entries)
        right = max(row["right"] for row in line_entries)
        bottom = max(row["bottom"] for row in line_entries)
        avg_conf = sum(row["confidence"] for row in line_entries) / len(line_entries)
        avg_line_height = sum(row["height"] for row in line_entries) / len(line_entries)
        blocks.append(
            TextBlock(
                page_index=page_index,
                block_id=block_id,
                source_text=text,
                translated_text="",
                left=left,
                top=top,
                width=right - left,
                height=bottom - top,
                confidence=avg_conf,
                source_line_height=avg_line_height,
            )
        )

    return _resolve_overlaps(blocks)
