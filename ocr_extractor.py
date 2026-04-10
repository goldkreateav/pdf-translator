from __future__ import annotations

import logging
from dataclasses import dataclass
from statistics import median
from typing import Sequence

import easyocr
import fitz  # PyMuPDF
import numpy as np


logger = logging.getLogger("pdf_translator.ocr")


@dataclass(frozen=True)
class OcrLine:
    line_id: str
    text: str
    bbox: tuple[float, float, float, float]
    size: float


@dataclass(frozen=True)
class OcrWord:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def y_center(self) -> float:
        return (self.y0 + self.y1) / 2.0

    @property
    def height(self) -> float:
        return max(1.0, self.y1 - self.y0)


def extract_ocr_lines_by_page(
    pdf_path: str,
    *,
    ocr_langs: Sequence[str] = ("ch_sim",),
    dpi: int = 220,
) -> list[list[OcrLine]]:
    """
    OCR fallback for scanned PDFs.
    Returns per-page lines with PDF-space bboxes.
    """
    reader = easyocr.Reader(list(ocr_langs), gpu=False, verbose=False)
    doc = fitz.open(pdf_path)
    out_pages: list[list[OcrLine]] = []

    for pno in range(doc.page_count):
        page = doc.load_page(pno)
        rect = page.rect
        pix = page.get_pixmap(dpi=dpi, alpha=False)

        # EasyOCR expects HxWxC RGB array.
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            img = img[:, :, :3]

        results = reader.readtext(img, detail=1, paragraph=False)
        words_px: list[OcrWord] = []
        for item in results:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            quad = item[0]
            text = str(item[1] or "").strip()
            if not text:
                continue
            xs = [float(pt[0]) for pt in quad]
            ys = [float(pt[1]) for pt in quad]
            words_px.append(
                OcrWord(
                    text=text,
                    x0=min(xs),
                    y0=min(ys),
                    x1=max(xs),
                    y1=max(ys),
                )
            )

        scale_x = rect.width / float(pix.width)
        scale_y = rect.height / float(pix.height)
        words_pdf = [
            OcrWord(
                text=w.text,
                x0=w.x0 * scale_x,
                y0=w.y0 * scale_y,
                x1=w.x1 * scale_x,
                y1=w.y1 * scale_y,
            )
            for w in words_px
        ]

        line_words = _group_words_to_lines(words_pdf)
        page_lines: list[OcrLine] = []
        for i, words in enumerate(line_words):
            x0 = min(w.x0 for w in words)
            y0 = min(w.y0 for w in words)
            x1 = max(w.x1 for w in words)
            y1 = max(w.y1 for w in words)
            text = _join_words(words)
            if not text:
                continue
            size = max(6.0, (y1 - y0) * 0.9)
            page_lines.append(
                OcrLine(
                    line_id=f"p{pno}_ocr_l{i}",
                    text=text,
                    bbox=(x0, y0, x1, y1),
                    size=size,
                )
            )

        out_pages.append(page_lines)
        logger.info("OCR page %d/%d: %d word(s), %d line(s)", pno + 1, doc.page_count, len(words_pdf), len(page_lines))

    doc.close()
    return out_pages


def _group_words_to_lines(words: list[OcrWord]) -> list[list[OcrWord]]:
    if not words:
        return []
    heights = [w.height for w in words]
    median_h = median(heights) if heights else 12.0
    y_threshold = max(2.0, median_h * 0.6)

    words_sorted = sorted(words, key=lambda w: (w.y_center, w.x0))
    lines: list[list[OcrWord]] = []
    current: list[OcrWord] = [words_sorted[0]]
    current_y = words_sorted[0].y_center

    for w in words_sorted[1:]:
        if abs(w.y_center - current_y) <= y_threshold:
            current.append(w)
            current_y = sum(x.y_center for x in current) / len(current)
        else:
            lines.append(sorted(current, key=lambda x: x.x0))
            current = [w]
            current_y = w.y_center
    lines.append(sorted(current, key=lambda x: x.x0))
    return lines


def _join_words(words: list[OcrWord]) -> str:
    if not words:
        return ""
    parts = [words[0].text]
    prev = words[0]
    for cur in words[1:]:
        if _needs_space(prev.text, cur.text):
            parts.append(" ")
        parts.append(cur.text)
        prev = cur
    return "".join(parts).strip()


def _needs_space(left: str, right: str) -> bool:
    if not left or not right:
        return False
    # If both sides are CJK, keep no spaces.
    if _is_cjk(left[-1]) and _is_cjk(right[0]):
        return False
    return True


def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0x20000 <= code <= 0x2A6DF
        or 0x2A700 <= code <= 0x2B73F
        or 0x2B740 <= code <= 0x2B81F
        or 0x2B820 <= code <= 0x2CEAF
        or 0xF900 <= code <= 0xFAFF
    )

