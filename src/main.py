import argparse
import os
import random

from dotenv import load_dotenv
from tqdm import tqdm

from src.layout import words_to_blocks
from src.masking import remove_text_preserve_lines
from src.ocr import extract_words
from src.pdf_to_images import rasterize_pdf
from src.render_pdf import render_translated_pdf
from src.translate_openai import translate_block_batch


FISH_RU = [
    "карась",
    "щука",
    "сом",
    "окунь",
    "форель",
    "лосось",
    "осётр",
    "судак",
    "карп",
    "килька",
    "сардина",
    "треска",
    "селёдка",
    "камбала",
]


def _fake_russian_fish(text: str) -> str:
    seed = hash(text)
    rng = random.Random(seed)
    approx_words = max(1, min(60, max(len(text) // 6, len(text.split()) or 1)))
    return " ".join(rng.choice(FISH_RU) for _ in range(approx_words))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate scanned PDFs by OCR + OpenAI + overlay rendering."
    )
    parser.add_argument("--in", dest="input_pdf", required=True, help="Input PDF path")
    parser.add_argument("--out", dest="output_pdf", required=True, help="Output PDF path")
    parser.add_argument("--src-lang", default="zh", help="Source language code")
    parser.add_argument("--tgt-lang", default="ru", help="Target language code")
    parser.add_argument("--dpi", type=int, default=300, help="Rasterization DPI")
    parser.add_argument(
        "--first-page",
        type=int,
        default=None,
        help="First page to process (1-based, passed to pdf2image).",
    )
    parser.add_argument(
        "--last-page",
        type=int,
        default=None,
        help="Last page to process (1-based, passed to pdf2image).",
    )
    parser.add_argument(
        "--tesseract-lang",
        default="chi_sim",
        help="Tesseract language(s), e.g. chi_sim or chi_sim+eng",
    )
    parser.add_argument(
        "--tesseract-psm",
        type=int,
        default=6,
        help="Tesseract page segmentation mode (PSM).",
    )
    parser.add_argument(
        "--tesseract-cmd",
        default=None,
        help="Optional path to tesseract.exe",
    )
    parser.add_argument(
        "--poppler-path",
        default=None,
        help="Optional Poppler bin path for pdf2image on Windows",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=35.0,
        help="Minimum OCR confidence threshold",
    )
    parser.add_argument(
        "--cover-mode",
        choices=["none", "box"],
        default="none",
        help="How to hide original text before drawing translation",
    )
    parser.add_argument(
        "--mask-mode",
        choices=["text_pixels"],
        default="text_pixels",
        help="Masking mode for removing source text before overlay.",
    )
    parser.add_argument(
        "--inpaint",
        choices=["telea", "whiten"],
        default="telea",
        help="Background restoration mode after text masking.",
    )
    parser.add_argument(
        "--preserve-lines",
        action="store_true",
        default=True,
        help="Protect detected table and separator lines while masking text.",
    )
    parser.add_argument(
        "--no-preserve-lines",
        action="store_false",
        dest="preserve_lines",
        help="Disable table-line protection while masking text.",
    )
    parser.add_argument(
        "--mask-padding",
        type=int,
        default=1,
        help="Padding (px) around OCR word boxes for text mask.",
    )
    parser.add_argument(
        "--block-granularity",
        choices=["paragraph", "line"],
        default="line",
        help="OCR grouping granularity for overlay blocks.",
    )
    parser.add_argument(
        "--debug-dir",
        default=None,
        help="If set, save debug PNGs (masks/cleaned/boxes) into this directory.",
    )
    parser.add_argument(
        "--font-path",
        default="assets/fonts/DejaVuSans.ttf",
        help="Path to TTF font that supports target language",
    )
    parser.add_argument(
        "--font-bold-path",
        default="assets/fonts/DejaVuSans-Bold.ttf",
        help="Optional bold font for heading-like blocks.",
    )
    parser.add_argument(
        "--openai-model",
        default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        help="OpenAI model name",
    )
    parser.add_argument(
        "--openai-base-url",
        default=os.getenv("OPENAI_BASE_URL", ""),
        help="OpenAI-compatible base URL (e.g. https://api.openai.com/v1 or proxy).",
    )
    parser.add_argument(
        "--openai-api-key",
        default=os.getenv("OPENAI_API_KEY", ""),
        help="OpenAI API key override (otherwise uses OPENAI_API_KEY env var).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Translation batch size",
    )
    parser.add_argument(
        "--skip-translate",
        action="store_true",
        help="Skip OpenAI calls and replace text with random Russian fish.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    print("Rasterizing PDF...")
    page_images = rasterize_pdf(
        input_pdf=args.input_pdf,
        dpi=args.dpi,
        poppler_path=args.poppler_path,
        first_page=args.first_page,
        last_page=args.last_page,
    )
    print(f"Rasterized {len(page_images)} page(s).")

    pages_blocks = []
    cleaned_page_images = []
    for page_index, page_image in enumerate(tqdm(page_images, desc="OCR + layout"), start=0):
        words = extract_words(
            image=page_image,
            tesseract_lang=args.tesseract_lang,
            min_confidence=args.min_confidence,
            tesseract_cmd=args.tesseract_cmd,
            psm=args.tesseract_psm,
        )
        blocks = words_to_blocks(words=words, page_index=page_index, granularity=args.block_granularity)
        if args.mask_mode == "text_pixels":
            cleaned_page = remove_text_preserve_lines(
                image=page_image,
                words=words,
                preserve_lines=args.preserve_lines,
                inpaint_mode=args.inpaint,
                padding=args.mask_padding,
            )
        else:
            cleaned_page = page_image
        pages_blocks.append(blocks)
        cleaned_page_images.append(cleaned_page)

    for page_idx, blocks in enumerate(tqdm(pages_blocks, desc="Translate"), start=0):
        if args.skip_translate:
            for block in blocks:
                block.translated_text = _fake_russian_fish(block.source_text)
        else:
            translated = translate_block_batch(
                blocks=blocks,
                source_lang=args.src_lang,
                target_lang=args.tgt_lang,
                model=args.openai_model,
                batch_size=args.batch_size,
                api_key=(args.openai_api_key or None),
                base_url=(args.openai_base_url or None),
            )
            for block, translated_text in zip(blocks, translated):
                block.translated_text = translated_text
        pages_blocks[page_idx] = blocks

    print("Rendering translated PDF...")
    render_translated_pdf(
        page_images=cleaned_page_images,
        page_blocks=pages_blocks,
        output_pdf=args.output_pdf,
        font_path=args.font_path,
        font_bold_path=args.font_bold_path,
        cover_mode=args.cover_mode,
        color_source_images=page_images,
    )
    print(f"Done. Output saved to: {args.output_pdf}")


if __name__ == "__main__":
    main()
