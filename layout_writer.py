from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import fitz  # PyMuPDF


@dataclass(frozen=True)
class RenderConfig:
    font_path: str | None = None
    default_font_size_scale: float = 1.0
    min_font_size: float = 4.0
    max_shrink_steps: int = 10
    dpi: int = 150
    whiteout_expand: float = 1.0
    rasterize_background: bool = False

logger = logging.getLogger("pdf_translator.render")


def pick_default_font_path() -> str | None:
    candidates = [
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\times.ttf",
        r"C:\Windows\Fonts\calibri.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def _rect_from_bbox(bbox: tuple[float, float, float, float]) -> fitz.Rect:
    x0, y0, x1, y1 = bbox
    return fitz.Rect(x0, y0, x1, y1)


def _expanded(rect: fitz.Rect, amount: float) -> fitz.Rect:
    if amount <= 0:
        return rect
    return fitz.Rect(rect.x0 - amount, rect.y0 - amount, rect.x1 + amount, rect.y1 + amount)


def render_translated_pdf(
    *,
    src_pdf_path: str,
    out_pdf_path: str,
    pages_lines: list[list[dict]],
    cfg: RenderConfig,
) -> None:
    """
    Creates a visually translated PDF by:
    Default (recommended): keep original pages (small output),
    white-out each original line bbox, draw translated text inside same bbox.

    Optional: rasterize each source page as background image (large output).
    """
    src = fitz.open(src_pdf_path)

    font_path = cfg.font_path or pick_default_font_path()
    _ = fitz.Font(fontfile=font_path) if font_path else None  # validate font if provided

    if cfg.rasterize_background:
        logger.info("Render mode: rasterize background (large output)")
        out = fitz.open()
        for page_index in range(src.page_count):
            page = src.load_page(page_index)
            rect = page.rect

            pix = page.get_pixmap(dpi=cfg.dpi, alpha=False)
            bg_bytes = pix.tobytes("png")

            out_page = out.new_page(width=rect.width, height=rect.height)
            out_page.insert_image(rect, stream=bg_bytes, keep_proportion=True)
            _render_lines_on_page(out_page, pages_lines, page_index, cfg, font_path)

        out.save(out_pdf_path, garbage=4, deflate=True, clean=True)
        out.close()
    else:
        logger.info("Render mode: edit original PDF (small output)")
        # Edit original PDF in-memory and save compressed.
        for page_index in range(src.page_count):
            page = src.load_page(page_index)
            lines = pages_lines[page_index] if page_index < len(pages_lines) else []

            # Add redactions to erase original text under each line bbox.
            redactions = 0
            for line in lines:
                bbox = line.get("bbox")
                translated = line.get("translated", "")
                if not bbox or not translated:
                    continue
                r = _expanded(_rect_from_bbox(bbox), cfg.whiteout_expand)
                page.add_redact_annot(r, fill=(1, 1, 1))
                redactions += 1

            # Apply redactions (API differs slightly across versions).
            try:
                page.apply_redactions()
            except TypeError:
                page.apply_redactions(0)

            _render_lines_on_page(page, pages_lines, page_index, cfg, font_path)
            if (page_index + 1) % 5 == 0 or page_index == src.page_count - 1:
                logger.info("Rendered page %d/%d (redactions: %d)", page_index + 1, src.page_count, redactions)

        src.save(out_pdf_path, garbage=4, deflate=True, clean=True)
    src.close()


def _render_lines_on_page(
    page: fitz.Page,
    pages_lines: list[list[dict]],
    page_index: int,
    cfg: RenderConfig,
    font_path: str | None,
) -> None:
    lines = pages_lines[page_index] if page_index < len(pages_lines) else []
    for line in lines:
        bbox = line.get("bbox")
        translated = line.get("translated", "")
        if not bbox or not translated:
            continue

        r = _expanded(_rect_from_bbox(bbox), cfg.whiteout_expand)
        base_size = float(line.get("size") or 10.0) * cfg.default_font_size_scale
        size = base_size

        # Try shrinking until it fits.
        for _ in range(cfg.max_shrink_steps + 1):
            if size < cfg.min_font_size:
                size = cfg.min_font_size
            rc = page.insert_textbox(
                r,
                translated,
                fontfile=font_path if font_path else None,
                fontname=None if font_path else "helv",
                fontsize=size,
                color=(0, 0, 0),
                align=fitz.TEXT_ALIGN_LEFT,
                overlay=True,
            )
            if rc >= 0:
                break
            size *= 0.9
