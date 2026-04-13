from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from src.models import OCRWord


def _to_bgr(image: Image.Image) -> np.ndarray:
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _to_pil(image_bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def build_text_mask_from_boxes(
    image_size: tuple[int, int],
    words: list[OCRWord],
    padding: int = 1,
) -> np.ndarray:
    width, height = image_size
    mask = np.zeros((height, width), dtype=np.uint8)
    for word in words:
        x1 = max(0, word.left - padding)
        y1 = max(0, word.top - padding)
        x2 = min(width, word.right + padding)
        y2 = min(height, word.bottom + padding)
        if x2 <= x1 or y2 <= y1:
            continue
        mask[y1:y2, x1:x2] = 255
    return mask


def detect_table_lines_mask(image: Image.Image) -> np.ndarray:
    """
    Detect long horizontal/vertical line art (tables, separators).
    Returned mask uses 255 for detected lines.
    """
    bgr = _to_bgr(image)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    binary_inv = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        9,
    )

    h, w = binary_inv.shape
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, w // 35), 1))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, h // 35)))

    horizontal = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, horiz_kernel, iterations=1)
    vertical = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, vert_kernel, iterations=1)

    lines = cv2.bitwise_or(horizontal, vertical)
    lines = cv2.dilate(lines, np.ones((2, 2), dtype=np.uint8), iterations=1)
    return lines


def remove_text_preserve_lines(
    image: Image.Image,
    words: list[OCRWord],
    preserve_lines: bool = True,
    inpaint_mode: str = "telea",
    padding: int = 1,
) -> Image.Image:
    """
    Remove OCR-detected text pixels while protecting detected table/line-art.
    """
    bgr = _to_bgr(image)
    text_mask = build_text_mask_from_boxes(image.size, words, padding=padding)
    final_mask = text_mask
    if preserve_lines:
        line_mask = detect_table_lines_mask(image)
        inverted_line_mask = cv2.bitwise_not(line_mask)
        final_mask = cv2.bitwise_and(text_mask, inverted_line_mask)

    if inpaint_mode == "telea":
        # Telea may create streak artifacts on large masked regions;
        # keep radius modest.
        cleaned = cv2.inpaint(bgr, final_mask, 2, cv2.INPAINT_TELEA)
    else:
        cleaned = bgr.copy()
        cleaned[final_mask > 0] = (255, 255, 255)

    return _to_pil(cleaned)
