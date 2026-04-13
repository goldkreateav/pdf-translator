param(
  [Parameter(Mandatory = $true)][string]$InputPdf,
  [Parameter(Mandatory = $true)][string]$OutputPdf,
  [string]$PopplerPath = "",
  [string]$TesseractCmd = ""
)

$cmd = @(
  "python -m src.main",
  "--in `"$InputPdf`"",
  "--out `"$OutputPdf`"",
  "--src-lang zh",
  "--tgt-lang ru",
  "--dpi 150",
  "--first-page 1",
  "--last-page 1",
  "--tesseract-lang chi_sim",
  "--mask-mode text_pixels",
  "--inpaint telea",
  "--cover-mode none",
  "--skip-translate"
)

if ($PopplerPath -ne "") {
  $cmd += "--poppler-path `"$PopplerPath`""
}
if ($TesseractCmd -ne "") {
  $cmd += "--tesseract-cmd `"$TesseractCmd`""
}

Invoke-Expression ($cmd -join " ")
