from pathlib import Path

from pdf2image.exceptions import PDFInfoNotInstalledError
from pdf2image import convert_from_path
from PIL.Image import Image
import os


def rasterize_pdf(
    input_pdf: str,
    dpi: int = 300,
    poppler_path: str | None = None,
    first_page: int | None = None,
    last_page: int | None = None,
) -> list[Image]:
    """Convert each PDF page into a PIL image."""
    input_path = Path(input_pdf)
    if not input_path.exists():
        raise FileNotFoundError(f"Input PDF not found: {input_path}")

    resolved_poppler_path = poppler_path or os.getenv("POPPLER_PATH")
    try:
        return convert_from_path(
            str(input_path),
            dpi=dpi,
            fmt="png",
            poppler_path=resolved_poppler_path,
            first_page=first_page,
            last_page=last_page,
        )
    except PDFInfoNotInstalledError as e:
        hint = (
            "Poppler not found. Install Poppler for Windows and either:\n"
            "- add Poppler's 'bin' folder to PATH, or\n"
            "- pass --poppler-path \"C:\\path\\to\\poppler\\Library\\bin\", or\n"
            "- set env var POPPLER_PATH to that folder.\n"
        )
        raise PDFInfoNotInstalledError(f"{e}\n\n{hint}") from e
