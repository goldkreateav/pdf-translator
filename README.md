# PDF Translate Overlay Tool

Translate scanned Chinese PDFs into Russian with visible text overlays.

## What this does
- Converts PDF pages to images
- Runs Tesseract OCR (`chi_sim`) with block/paragraph/line-aware grouping
- Translates text blocks with OpenAI
- Removes source text with pixel masks while preserving table/separator lines
- Draws translated Russian text over the cleaned page with approximate color/style
- Exports a new PDF

## Requirements
- Python 3.11+
- Tesseract OCR installed with Chinese language data (`chi_sim`)
- Poppler installed (needed by `pdf2image` on Windows)

## Windows OCR/PDF tools
- Tesseract default path is often:
  - `C:\Program Files\Tesseract-OCR\tesseract.exe`
- Poppler path usually points to:
  - `C:\path\to\poppler\Library\bin`
- If these are not on `PATH`, pass them with `--tesseract-cmd` and `--poppler-path`.

## Setup
1. Create virtual environment and install deps:
   - `python -m venv .venv`
   - `.venv\\Scripts\\Activate.ps1`
   - `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and set `OPENAI_API_KEY`.
3. Place `DejaVuSans.ttf` into `assets/fonts/` (or pass `--font-path`).

## Run
```powershell
python -m src.main --in input.pdf --out output_ru.pdf --src-lang zh --tgt-lang ru --dpi 300 --tesseract-lang chi_sim --mask-mode text_pixels --inpaint telea --cover-mode none
```

Optional:
- `--poppler-path "C:\\path\\to\\poppler\\Library\\bin"`
- `--tesseract-cmd "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"`
- `--font-path "C:\\path\\to\\DejaVuSans.ttf"`
- `--font-bold-path "C:\\path\\to\\DejaVuSans-Bold.ttf"`
- `--preserve-lines` (default on) / `--no-preserve-lines`
- `--first-page 1 --last-page 1` for quick visual iteration
- `--skip-translate` for debug output without OpenAI calls
- `--openai-base-url "https://api.openai.com/v1"` for OpenAI-compatible proxies/hosts

## Smoke test script
```powershell
.\scripts\smoke_test.ps1 -InputPdf .\input.pdf -OutputPdf .\output_ru.pdf
```

## Smoke test checklist
- Output PDF is created.
- Output page count matches input.
- Russian text is visible near original Chinese text.
- Table lines and separators remain visible after masking.
- Heading color/style is approximately preserved.
- If Russian glyphs are broken, provide a proper Unicode font via `--font-path`.

## Notes
- This is bbox-based overlay replacement, not true editable text replacement.
- For complex layouts (tables, vertical text), block grouping may need tuning.
