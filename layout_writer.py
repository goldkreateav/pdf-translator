from __future__ import annotations

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
    - rasterizing each source page as background image
    - white-out each original line bbox
    - drawing translated text inside the same bbox
    """
    src = fitz.open(src_pdf_path)
    out = fitz.open()

    font_path = cfg.font_path or pick_default_font_path()
    font = fitz.Font(fontfile=font_path) if font_path else None

    for page_index in range(src.page_count):
        page = src.load_page(page_index)
        rect = page.rect

        pix = page.get_pixmap(dpi=cfg.dpi, alpha=False)
        bg_bytes = pix.tobytes("png")

        out_page = out.new_page(width=rect.width, height=rect.height)
        out_page.insert_image(rect, stream=bg_bytes, keep_proportion=True)

        lines = pages_lines[page_index] if page_index < len(pages_lines) else []
        for line in lines:
            bbox = line.get("bbox")
            translated = line.get("translated", "")
            if not bbox or not translated:
                continue

            r = _expanded(_rect_from_bbox(bbox), cfg.whiteout_expand)

            # White-out original text region.
            out_page.draw_rect(r, color=None, fill=(1, 1, 1), overlay=True)

            base_size = float(line.get("size") or 10.0) * cfg.default_font_size_scale
            size = base_size

            # Try shrinking until it fits.
            for _ in range(cfg.max_shrink_steps + 1):
                if size < cfg.min_font_size:
                    size = cfg.min_font_size
                rc = out_page.insert_textbox(
                    r,
                    translated,
                    fontfile=font_path if font_path else None,
                    fontname=None if font_path else "helv",
                    fontsize=size,
                    color=(0, 0, 0),
                    align=fitz.TEXT_ALIGN_LEFT,
                    overlay=True,
                )
                # insert_textbox returns > 0 if there is leftover space; < 0 means text did not fit.
                if rc >= 0:
                    break
                size *= 0.9

    out.save(out_pdf_path)
    out.close()
    src.close()

