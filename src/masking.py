from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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


@dataclass
class MaskDebug:
    ink_mask: np.ndarray | None = None
    text_mask: np.ndarray | None = None
    line_mask: np.ndarray | None = None
    protect_mask: np.ndarray | None = None
    remove_mask: np.ndarray | None = None
    cleaned_bgr: np.ndarray | None = None


def _save_mask(debug_dir: str, name: str, mask: np.ndarray) -> None:
    Path(debug_dir).mkdir(parents=True, exist_ok=True)
    out_path = str(Path(debug_dir) / name)
    cv2.imwrite(out_path, mask)


def _save_bgr(debug_dir: str, name: str, bgr: np.ndarray) -> None:
    Path(debug_dir).mkdir(parents=True, exist_ok=True)
    out_path = str(Path(debug_dir) / name)
    cv2.imwrite(out_path, bgr)


def _build_ink_mask(image: Image.Image) -> np.ndarray:
    """
    Binary mask of 'ink' pixels (text strokes + lines) using adaptive threshold.
    Returns 255 where ink is present.
    """
    bgr = _to_bgr(image)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # Slight blur helps include anti-aliased stroke edges.
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    ink = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        41,
        7,
    )
    # Expand strokes to cover light halos and avoid ghosting.
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8), iterations=1)
    ink = cv2.dilate(ink, np.ones((3, 3), dtype=np.uint8), iterations=1)
    return ink


def _build_protect_mask(
    image: Image.Image,
    line_mask: np.ndarray,
) -> np.ndarray:
    """
    Protect non-text: table/separator lines and large filled graphics/logos.
    """
    ink = _build_ink_mask(image)
    # Large blobs are likely graphics/stamps; protect them.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    h, w = ink.shape[:2]
    page_area = float(h * w)

    protect = line_mask.copy()
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area <= 0:
            continue
        # Protect very large components (graphics / big stamps / big filled shapes).
        if area / page_area >= 0.0025:
            protect[labels == label] = 255

    protect = cv2.dilate(protect, np.ones((2, 2), dtype=np.uint8), iterations=1)
    return protect


def build_symbol_mask(
    image: Image.Image,
    preserve_lines: bool = True,
    debug: MaskDebug | None = None,
) -> np.ndarray:
    """
    Detect symbol-like components (characters) by pixel morphology + connected components.
    Returns 255 where symbol pixels should be removed.
    """
    ink = _build_ink_mask(image)
    if debug is not None:
        debug.ink_mask = ink

    line_mask = detect_table_lines_mask(image) if preserve_lines else np.zeros_like(ink)
    if debug is not None:
        debug.line_mask = line_mask

    protect = _build_protect_mask(image, line_mask=line_mask) if preserve_lines else np.zeros_like(ink)
    if debug is not None:
        debug.protect_mask = protect

    candidate = cv2.bitwise_and(ink, cv2.bitwise_not(protect))

    # Filter connected components by size/aspect to keep character-like shapes.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    h, w = candidate.shape[:2]
    remove = np.zeros((h, w), dtype=np.uint8)

    for label in range(1, num_labels):
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        cw = stats[label, cv2.CC_STAT_WIDTH]
        ch = stats[label, cv2.CC_STAT_HEIGHT]
        area = stats[label, cv2.CC_STAT_AREA]

        if area < 6:
            continue
        if area > 6000:
            # Too big; likely graphics; keep protected by default.
            continue
        if cw <= 0 or ch <= 0:
            continue

        aspect = ch / float(cw)
        if aspect > 12 or aspect < 0.08:
            continue

        # Characters tend to be relatively compact.
        if cw > w * 0.4 or ch > h * 0.15:
            continue

        remove[labels == label] = 255

    # Expand a bit more to fully cover stroke edges.
    remove = cv2.dilate(remove, np.ones((3, 3), dtype=np.uint8), iterations=2)
    return remove


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
    debug_dir: str | None = None,
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

    if debug_dir:
        _save_mask(debug_dir, "text_mask.png", text_mask)
        if preserve_lines:
            _save_mask(debug_dir, "line_mask.png", line_mask)
        _save_mask(debug_dir, "remove_mask.png", final_mask)
        _save_bgr(debug_dir, "cleaned.png", cleaned)

    return _to_pil(cleaned)


def remove_symbols_preserve_lines(
    image: Image.Image,
    preserve_lines: bool = True,
    inpaint_mode: str = "whiten",
    debug_dir: str | None = None,
) -> Image.Image:
    bgr = _to_bgr(image)
    debug = MaskDebug()
    remove_mask = build_symbol_mask(image=image, preserve_lines=preserve_lines, debug=debug)
    if debug is not None:
        debug.remove_mask = remove_mask

    if inpaint_mode == "telea":
        cleaned = cv2.inpaint(bgr, remove_mask, 2, cv2.INPAINT_TELEA)
    else:
        cleaned = bgr.copy()
        cleaned[remove_mask > 0] = (255, 255, 255)

    if debug_dir:
        if debug.ink_mask is not None:
            _save_mask(debug_dir, "ink_mask.png", debug.ink_mask)
        if debug.line_mask is not None:
            _save_mask(debug_dir, "line_mask.png", debug.line_mask)
        if debug.protect_mask is not None:
            _save_mask(debug_dir, "protect_mask.png", debug.protect_mask)
        _save_mask(debug_dir, "remove_mask.png", remove_mask)
        _save_bgr(debug_dir, "cleaned.png", cleaned)

    return _to_pil(cleaned)
