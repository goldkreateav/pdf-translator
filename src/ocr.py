from PIL.Image import Image
import pytesseract
from pytesseract import Output
from pytesseract.pytesseract import TesseractNotFoundError

from src.models import OCRWord


def extract_words(
    image: Image,
    tesseract_lang: str,
    min_confidence: float = 35.0,
    tesseract_cmd: str | None = None,
    psm: int = 6,
) -> list[OCRWord]:
    """Extract OCR words and bounding boxes from a page image."""
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    try:
        raw = pytesseract.image_to_data(
            image,
            lang=tesseract_lang,
            output_type=Output.DICT,
            config=f"--oem 3 --psm {psm}",
        )
    except TesseractNotFoundError as e:
        hint = (
            "Tesseract not found. Install Tesseract OCR and either:\n"
            "- add tesseract.exe to PATH, or\n"
            "- pass --tesseract-cmd \"C:\\Program Files\\Tesseract-OCR\\tesseract.exe\"."
        )
        raise TesseractNotFoundError(f"{e}\n\n{hint}") from e

    words: list[OCRWord] = []
    total = len(raw["text"])
    for index in range(total):
        text = (raw["text"][index] or "").strip()
        if not text:
            continue

        try:
            confidence = float(raw["conf"][index])
        except (TypeError, ValueError):
            continue

        if confidence < min_confidence:
            continue

        width = int(raw["width"][index])
        height = int(raw["height"][index])
        # Filter obvious OCR noise (tiny specks / separators).
        if width * height < 20:
            continue
        if width > 0 and (height / width) > 8:
            continue

        words.append(
            OCRWord(
                text=text,
                confidence=confidence,
                left=int(raw["left"][index]),
                top=int(raw["top"][index]),
                width=width,
                height=height,
                block_num=int(raw["block_num"][index]),
                par_num=int(raw["par_num"][index]),
                line_num=int(raw["line_num"][index]),
                word_num=int(raw["word_num"][index]),
            )
        )

    return words
