from io import BytesIO
from pathlib import Path
from textwrap import wrap

import numpy as np
from PIL.Image import Image
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

from src.models import TextBlock


def _register_fonts(font_path: str | None, font_bold_path: str | None) -> tuple[str, str]:
    regular_name = "Helvetica"
    bold_name = "Helvetica-Bold"
    if font_path and Path(font_path).exists():
        regular_name = "OverlayFontRegular"
        pdfmetrics.registerFont(TTFont(regular_name, font_path))
        bold_name = regular_name
    if font_bold_path and Path(font_bold_path).exists():
        bold_name = "OverlayFontBold"
        pdfmetrics.registerFont(TTFont(bold_name, font_bold_path))
    return regular_name, bold_name


def _estimate_char_width(font_size: float) -> float:
    return max(1.0, font_size * 0.55)


def _fit_text(
    text: str,
    width: int,
    height: int,
    preferred_font: float,
    min_font: float = 6,
    max_font: float = 30,
) -> tuple[float, list[str]]:
    best_size = max(min_font, min(max_font, preferred_font))
    best_lines = [text]
    low, high = min_font, max_font

    while low <= high:
        mid = (low + high) / 2
        chars_per_line = max(1, int(width / _estimate_char_width(mid)))
        candidate_lines: list[str] = []
        for paragraph in text.splitlines() or [""]:
            candidate_lines.extend(wrap(paragraph, width=chars_per_line) or [""])

        line_height = mid * 1.25
        total_height = line_height * max(1, len(candidate_lines))
        if total_height <= height:
            best_size = mid
            best_lines = candidate_lines
            low = mid + 0.5
        else:
            high = mid - 0.5

    return best_size, best_lines


def _estimate_block_color(image: Image, block: TextBlock) -> tuple[int, int, int]:
    img = np.array(image.convert("RGB"))
    h, w = img.shape[:2]
    x1 = max(0, block.left)
    y1 = max(0, block.top)
    x2 = min(w, block.right)
    y2 = min(h, block.bottom)
    if x2 <= x1 or y2 <= y1:
        return (0, 0, 0)

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return (0, 0, 0)

    lum = (
        0.299 * crop[:, :, 0] + 0.587 * crop[:, :, 1] + 0.114 * crop[:, :, 2]
    ).astype(np.float32)

    # Use only the darkest pixels (more robust on light backgrounds).
    threshold = np.percentile(lum, 8)
    dark_pixels = crop[lum <= threshold]
    if dark_pixels.size == 0:
        dark_pixels = crop.reshape(-1, 3)

    r, g, b = np.median(dark_pixels, axis=0)
    rgb = (int(r), int(g), int(b))

    # If it still looks too bright, fall back to black to avoid invisible overlays.
    if (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) > 170:
        return (0, 0, 0)
    return rgb


def render_translated_pdf(
    page_images: list[Image],
    page_blocks: list[list[TextBlock]],
    output_pdf: str,
    font_path: str | None = None,
    font_bold_path: str | None = None,
    cover_mode: str = "box",
    color_source_images: list[Image] | None = None,
) -> None:
    if len(page_images) != len(page_blocks):
        raise ValueError("Page image count does not match block page count.")
    if color_source_images is not None and len(color_source_images) != len(page_images):
        raise ValueError("Color source image count does not match page count.")

    c = canvas.Canvas(output_pdf)
    regular_font, bold_font = _register_fonts(font_path, font_bold_path)

    for page_index, (page_image, blocks) in enumerate(zip(page_images, page_blocks)):
        color_image = color_source_images[page_index] if color_source_images else page_image
        page_w, page_h = page_image.size
        c.setPageSize((page_w, page_h))

        image_buffer = BytesIO()
        page_image.save(image_buffer, format="PNG")
        image_buffer.seek(0)
        c.drawImage(ImageReader(image_buffer), 0, 0, width=page_w, height=page_h)

        line_heights = [block.source_line_height for block in blocks if block.source_line_height > 0]
        median_line_height = float(np.median(line_heights)) if line_heights else 12.0

        for block in blocks:
            draw_x = block.left
            draw_y = page_h - block.top - block.height
            draw_w = max(1, block.width)
            draw_h = max(1, block.height)
            text = (block.translated_text or "").strip()
            if not text:
                continue

            if cover_mode == "box":
                c.setFillColorRGB(1, 1, 1)
                c.rect(draw_x, draw_y, draw_w, draw_h, stroke=0, fill=1)

            if block.color_rgb is None:
                block.color_rgb = _estimate_block_color(color_image, block)

            heading_by_height = block.source_line_height > (median_line_height * 1.35)
            short_text = len(text) <= 40
            block.is_heading = block.is_heading or (heading_by_height and short_text)

            preferred_font = max(6.0, min(42.0, block.source_line_height * 0.95))
            font_size, lines = _fit_text(text, draw_w, draw_h, preferred_font=preferred_font)
            line_height = font_size * 1.25
            start_y = draw_y + draw_h - line_height

            r, g, b = block.color_rgb
            c.setFillColorRGB(r / 255.0, g / 255.0, b / 255.0)
            selected_font = bold_font if block.is_heading else regular_font
            c.setFont(selected_font, font_size)
            for idx, line in enumerate(lines):
                y = start_y - idx * line_height
                if y < draw_y:
                    break
                c.drawString(draw_x, y, line)

        c.showPage()

    c.save()
