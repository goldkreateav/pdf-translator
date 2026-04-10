from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass

import fitz  # PyMuPDF

from layout_writer import RenderConfig, render_translated_pdf
from translator_client import build_in_run_glossary, load_openai_compat_config, translate_block_lines


@dataclass(frozen=True)
class ExtractedLine:
    line_id: str
    text: str
    bbox: tuple[float, float, float, float]
    size: float


logger = logging.getLogger("pdf_translator")


def _norm_space(s: str) -> str:
    return " ".join((s or "").replace("\u00a0", " ").split())


def extract_lines_by_page(pdf_path: str) -> list[list[ExtractedLine]]:
    doc = fitz.open(pdf_path)
    pages: list[list[ExtractedLine]] = []

    for pno in range(doc.page_count):
        page = doc.load_page(pno)
        d = page.get_text("rawdict")
        page_lines: list[ExtractedLine] = []
        for b_idx, block in enumerate(d.get("blocks", [])):
            if block.get("type") != 0:
                continue
            for l_idx, line in enumerate(block.get("lines", [])):
                spans = line.get("spans", [])
                text = _norm_space("".join([s.get("text", "") for s in spans]))
                if not text:
                    continue
                bbox = tuple(line.get("bbox"))  # type: ignore[arg-type]
                size = float(spans[0].get("size") if spans else 10.0)
                line_id = f"p{pno}_b{b_idx}_l{l_idx}"
                page_lines.append(ExtractedLine(line_id=line_id, text=text, bbox=bbox, size=size))
        pages.append(page_lines)
        if (pno + 1) % 5 == 0 or pno == doc.page_count - 1:
            logger.info("Extracted page %d/%d: %d line(s)", pno + 1, doc.page_count, len(page_lines))
    doc.close()
    return pages


def group_lines_into_blocks(lines: list[ExtractedLine]) -> list[list[ExtractedLine]]:
    """
    Simple layout-based grouping: consecutive lines close in Y and similar left margin are treated as a block.
    """
    if not lines:
        return []

    # Sort by y0 then x0 for stable reading order.
    lines_sorted = sorted(lines, key=lambda l: (l.bbox[1], l.bbox[0]))
    blocks: list[list[ExtractedLine]] = []
    cur: list[ExtractedLine] = []
    last = None

    for ln in lines_sorted:
        if last is None:
            cur = [ln]
            last = ln
            continue

        y_gap = ln.bbox[1] - last.bbox[3]
        x_delta = abs(ln.bbox[0] - last.bbox[0])
        similar_size = abs(ln.size - last.size) <= 2.0

        if y_gap <= max(6.0, last.size * 0.8) and x_delta <= 20.0 and similar_size:
            cur.append(ln)
        else:
            blocks.append(cur)
            cur = [ln]
        last = ln

    if cur:
        blocks.append(cur)
    return blocks


def translate_pages(
    *,
    pages: list[list[ExtractedLine]],
    source_lang: str,
    target_lang: str,
    api_base: str | None,
    api_key: str | None,
    model: str | None,
) -> list[list[dict]]:
    cfg = load_openai_compat_config(api_base=api_base, api_key=api_key, model=model)
    logger.info("Using API base: %s", cfg.api_base)
    logger.info("Using model: %s", cfg.model)
    if not cfg.api_key:
        logger.warning("No API key provided (OPENAI_API_KEY or --api-key). If your endpoint requires a key, calls will fail.")

    # In-run glossary seeded from the whole document (pass-through tokens).
    all_texts = (ln.text for page in pages for ln in page)
    glossary = build_in_run_glossary(all_texts)
    logger.info("Glossary passthrough tokens: %d", len(glossary))

    out_pages: list[list[dict]] = []
    for page_index, page_lines in enumerate(pages):
        blocks = group_lines_into_blocks(page_lines)
        logger.info("Page %d: %d line(s), %d block(s)", page_index + 1, len(page_lines), len(blocks))

        translated_by_id: dict[str, str] = {}
        for i, block in enumerate(blocks):
            prev_context = "\n".join([x.text for x in blocks[i - 1]]) if i - 1 >= 0 else None
            next_context = "\n".join([x.text for x in blocks[i + 1]]) if i + 1 < len(blocks) else None

            req_lines = [{"line_id": x.line_id, "text": x.text} for x in block]
            logger.info("Translating page %d block %d/%d (%d line(s))", page_index + 1, i + 1, len(blocks), len(req_lines))
            mapping = translate_block_lines(
                cfg=cfg,
                source_lang=source_lang,
                target_lang=target_lang,
                lines=req_lines,
                prev_context=prev_context,
                next_context=next_context,
                glossary=glossary,
            )
            translated_by_id.update(mapping)

        out_page: list[dict] = []
        for ln in page_lines:
            out_page.append(
                {
                    "line_id": ln.line_id,
                    "text": ln.text,
                    "translated": translated_by_id.get(ln.line_id, ""),
                    "bbox": ln.bbox,
                    "size": ln.size,
                }
            )
        out_pages.append(out_page)
    return out_pages


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Translate a PDF and preserve line positions.")
    p.add_argument("pdf_path", help="Path to input PDF (e.g. original.pdf)")
    p.add_argument("--output", default="translated.pdf", help="Output PDF path (default: translated.pdf)")
    p.add_argument("--source", default="zh", help="Source language code (default: zh)")
    p.add_argument("--target", default="ru", help="Target language code (default: ru)")
    p.add_argument("--log-level", default="INFO", help="Log level: DEBUG, INFO, WARNING, ERROR (default: INFO)")

    p.add_argument("--api-base", default=None, help="OpenAI-compatible API base URL (env: OPENAI_API_BASE)")
    p.add_argument("--api-key", default=None, help="API key (env: OPENAI_API_KEY)")
    p.add_argument("--model", default=None, help="Model name (env: OPENAI_MODEL)")

    p.add_argument("--font-path", default=None, help="Path to a TTF font with Cyrillic support (optional)")
    p.add_argument("--dpi", type=int, default=150, help="Background render DPI (default: 150)")
    p.add_argument(
        "--rasterize-background",
        action="store_true",
        help="Rasterize each page as an image background (much larger output). Default: off.",
    )
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    pdf_path = args.pdf_path
    out_path = args.output

    if not os.path.exists(pdf_path):
        raise SystemExit(f"Input PDF not found: {pdf_path}")

    logger.info("Input: %s", os.path.abspath(pdf_path))
    logger.info("Output: %s", os.path.abspath(out_path))

    pages = extract_lines_by_page(pdf_path)
    total_lines = sum(len(p) for p in pages)
    logger.info("Total extracted lines: %d", total_lines)
    if total_lines == 0:
        raise SystemExit(
            "No text lines extracted from PDF. If this is a scanned/image PDF, OCR is required (not implemented in v1)."
        )

    pages_lines = translate_pages(
        pages=pages,
        source_lang=args.source,
        target_lang=args.target,
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
    )

    translated_lines = sum(1 for p in pages_lines for ln in p if (ln.get("translated") or "").strip())
    logger.info("Total translated lines: %d", translated_lines)
    if translated_lines == 0:
        raise SystemExit(
            "Got 0 translated lines. Check that your API base/model/key are correct and that the PDF contains extractable text."
        )

    render_translated_pdf(
        src_pdf_path=pdf_path,
        out_pdf_path=out_path,
        pages_lines=pages_lines,
        cfg=RenderConfig(
            font_path=args.font_path,
            dpi=args.dpi,
            rasterize_background=bool(args.rasterize_background),
        ),
    )
    logger.info("Done. Wrote: %s", os.path.abspath(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

