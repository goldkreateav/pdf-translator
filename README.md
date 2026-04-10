# PDF Translator (layout-preserving)

One-command PDF translation that keeps each translated line drawn at the **same position** as the source line.

## Install

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

## Configure OpenAI-compatible API

You can use **env vars** and/or **CLI args** (CLI overrides env).

- `OPENAI_API_BASE`: e.g. `https://api.openai.com` or your compatible endpoint base
- `OPENAI_API_KEY`: your key (if required by the endpoint)
- `OPENAI_MODEL`: model name (default in code: `gpt-4.1-mini`)

## Run

Default is **Chinese -> Russian**:

```bash
python pdf_translator.py original.pdf
```

This writes `translated.pdf` in the current folder.

Optional overrides:

```bash
python pdf_translator.py original.pdf --source zh --target ru --output translated.pdf --model your-model
```

If Cyrillic glyphs render incorrectly, provide a font:

```bash
python pdf_translator.py original.pdf --font-path "C:\\Windows\\Fonts\\arial.ttf"
```

## How layout is preserved (v1)

- Each page is rasterized as a **background image**.
- Each extracted text line bbox is **white-out** filled.
- The translated line is drawn back into the **same bbox**; if it doesn’t fit, font size is reduced.

This is designed to keep the *visual* layout stable. (Text won’t be selectable like native PDF text in this v1 approach.)

## Limitations

- **Scanned PDFs** (images) are not OCR’d in v1; only real text content is extracted.
- Very complex layouts may need tuning of grouping/whiteout/fit behavior.

