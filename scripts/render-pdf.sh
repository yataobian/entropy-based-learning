#!/usr/bin/env bash
# render-pdf.sh
#
# Build a clean, print-ready PDF of the current book draft.
# HTML site generation is left untouched (uses _book/; this uses _book-pdf/).
#
# Usage (from the book root, or anywhere):
#   ./scripts/render-pdf.sh
#   ./scripts/render-pdf.sh --open
#   ./scripts/render-pdf.sh --keep-tex
#
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  render-pdf.sh [--open] [--keep-tex]

  (no flags)   Render the book to PDF with the `pdf` Quarto profile.
  --open       Open the PDF after a successful build (macOS).
  --keep-tex   Keep the intermediate .tex (for debugging layout).
  -h, --help   Show this help.

Output:
  _book-pdf/Entropy-Based-Learning.pdf   (build artifact, gitignored)
  local/Entropy-Based-Learning.pdf       (stable local copy, gitignored)
  assets/Entropy-Based-Learning.pdf      (tracked; published with the site)

Style: KOMA-Script scrbook, TeX Gyre Pagella, muted teal links.
Requires: quarto, lualatex (TeX Live).
EOF
}

OPEN=0
KEEP_TEX=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --open) OPEN=1; shift ;;
    --keep-tex) KEEP_TEX=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$ROOT"

if ! command -v quarto >/dev/null 2>&1; then
  echo "error: quarto is not on PATH" >&2
  exit 1
fi
if ! command -v lualatex >/dev/null 2>&1; then
  echo "error: lualatex is not on PATH (install TeX Live / MacTeX)" >&2
  exit 1
fi

# Rebuild Matplotlib figures so the PDF sibling is font-complete.
if [[ -f code/min-entropy-phase-diagram.py ]]; then
  echo "regenerating figures/min-entropy-phase.pdf ..."
  MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-ebm}" \
    /usr/bin/python3 code/min-entropy-phase-diagram.py >/dev/null
fi

# SVG figures need a PDF sibling for the Lua filter.
# Prefer a PDF already written by the figure script (Matplotlib embeds
# fonts). ImageMagick's SVG parser drops math glyphs and yields empty boxes.
svg_to_pdf() {
  local svg=$1
  local pdf="${svg%.svg}.pdf"
  if [[ -f $pdf ]]; then
    echo "using existing $(basename "$pdf")"
    return 0
  fi
  echo "error: missing $pdf" >&2
  echo "Export PDF from the figure script (not via ImageMagick from SVG)." >&2
  exit 1
}

shopt -s nullglob
for svg in figures/*.svg; do
  svg_to_pdf "$svg"
done
shopt -u nullglob

echo "rendering PDF (profile=pdf) ..."
if [[ $KEEP_TEX -eq 1 ]]; then
  quarto render --profile pdf --to pdf -M keep-tex:true
else
  quarto render --profile pdf --to pdf
fi

# Quarto names the book PDF after the project title / output-file.
PDF_SRC=""
for candidate in \
  "$ROOT/_book-pdf/Entropy-Based-Learning.pdf" \
  "$ROOT/_book-pdf/"*.pdf
do
  if [[ -f $candidate ]]; then
    PDF_SRC=$candidate
    break
  fi
done

if [[ -z $PDF_SRC ]]; then
  echo "error: quarto finished but no PDF was found in _book-pdf/" >&2
  exit 1
fi

mkdir -p "$ROOT/local" "$ROOT/assets"
STABLE="$ROOT/local/Entropy-Based-Learning.pdf"
PUBLISHED="$ROOT/assets/Entropy-Based-Learning.pdf"
cp -f "$PDF_SRC" "$STABLE"
cp -f "$PDF_SRC" "$PUBLISHED"

SIZE=$(du -h "$PUBLISHED" | awk '{print $1}')
PAGES=""
if command -v pdfinfo >/dev/null 2>&1; then
  PAGES=$(pdfinfo "$PUBLISHED" 2>/dev/null | awk '/^Pages:/ {print $2}' || true)
elif command -v mdls >/dev/null 2>&1; then
  PAGES=$(mdls -name kMDItemNumberOfPages -raw "$PUBLISHED" 2>/dev/null || true)
  [[ $PAGES == "(null)" ]] && PAGES=""
fi

echo
echo "PDF ready:"
echo "  $PDF_SRC"
echo "  $STABLE"
echo "  $PUBLISHED"
if [[ -n ${PAGES} ]]; then
  echo "  ${PAGES} pages, ${SIZE}"
else
  echo "  ${SIZE}"
fi

if [[ $OPEN -eq 1 ]]; then
  if command -v open >/dev/null 2>&1; then
    open "$STABLE"
  else
    echo "note: 'open' not available; PDF is at $STABLE"
  fi
fi
