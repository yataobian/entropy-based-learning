#!/usr/bin/env bash
# render-pdf.sh
#
# Build a clean, print-ready PDF of the current book draft.
# HTML site generation is left untouched (uses _book/; this uses _book-pdf/).
#
# Usage (from the book root, or anywhere):
#   ./scripts/render-pdf.sh                  # option 1, the print PDF
#   ./scripts/render-pdf.sh --style 2        # option 2, STIX / FEP-like
#   ./scripts/render-pdf.sh --style both
#   ./scripts/render-pdf.sh --open
#   ./scripts/render-pdf.sh --keep-tex
#
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  render-pdf.sh [--style 1|2|both] [--open] [--keep-tex]

  --style 1    Option 1 (default of this script). KOMA-Script, TeX Gyre Pagella,
               LuaLaTeX, handwritten contents.
  --style 2    Option 2. XeLaTeX, STIXGeneral, STIX Two Math, generated
               contents, margins as in the FEP note. Overwrites
               assets/Entropy-Based-Learning.pdf, which is what the site publishes.
  --style both Render option 1, then option 2. Option 2 is the file left in assets/.
  --open       Open the PDF after a successful build (macOS).
  --keep-tex   Keep the intermediate .tex (for debugging layout).
  -h, --help   Show this help.

Both options write the same published file:
  assets/Entropy-Based-Learning.pdf      (tracked; the site serves this)
  local/Entropy-Based-Learning.pdf       (stable local copy, gitignored)

Build directories (gitignored):
  _book-pdf/        option 1
  _book-pdf-stix/   option 2

Requires: quarto, and lualatex (option 1) or xelatex (option 2).
EOF
}

OPEN=0
KEEP_TEX=0
STYLE=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --open) OPEN=1; shift ;;
    --keep-tex) KEEP_TEX=1; shift ;;
    --style)
      if [[ $# -lt 2 ]]; then
        echo "error: --style needs 1, 2, or both" >&2
        usage
        exit 2
      fi
      STYLE=$2
      shift 2
      ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

case "$STYLE" in
  1|2|both) ;;
  *) echo "error: --style must be 1, 2, or both (got: $STYLE)" >&2; usage; exit 2 ;;
esac

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$ROOT"

if ! command -v quarto >/dev/null 2>&1; then
  echo "error: quarto is not on PATH" >&2
  exit 1
fi
if [[ $STYLE == 1 || $STYLE == both ]]; then
  if ! command -v lualatex >/dev/null 2>&1; then
    echo "error: lualatex is not on PATH (install TeX Live / MacTeX)" >&2
    exit 1
  fi
fi
if [[ $STYLE == 2 || $STYLE == both ]]; then
  if ! command -v xelatex >/dev/null 2>&1; then
    echo "error: xelatex is not on PATH (install TeX Live / MacTeX)" >&2
    exit 1
  fi
fi

# Rebuild the phase diagram only when the script is newer than the
# committed products, or when a product is missing. The figure is
# generated once and reused; ImageMagick must not rewrite the PDF
# from the SVG (it drops math glyphs).
need_phase_figure=0
PHASE_PY="code/min-entropy-phase-diagram.py"
if [[ -f $PHASE_PY ]]; then
  for ext in pdf svg png; do
    out="figures/min-entropy-phase.$ext"
    if [[ ! -f $out || $PHASE_PY -nt $out ]]; then
      need_phase_figure=1
      break
    fi
  done
  if [[ $need_phase_figure -eq 1 ]]; then
    echo "regenerating figures/min-entropy-phase.{pdf,svg,png} ..."
    MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-ebm}" \
      /usr/bin/python3 "$PHASE_PY" >/dev/null
  else
    echo "using existing figures/min-entropy-phase.{pdf,svg,png}"
  fi
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

# Title-page mark. Each profile's tex looks in its output dir and the book root.
if [[ -f $ROOT/cover.png ]]; then
  if [[ $STYLE == 1 || $STYLE == both ]]; then
    mkdir -p "$ROOT/_book-pdf"
    cp -f "$ROOT/cover.png" "$ROOT/_book-pdf/cover.png"
  fi
  if [[ $STYLE == 2 || $STYLE == both ]]; then
    mkdir -p "$ROOT/_book-pdf-stix"
    cp -f "$ROOT/cover.png" "$ROOT/_book-pdf-stix/cover.png"
  fi
fi

# Render one profile and copy the PDF to the paths the site publishes.
# Both styles use the same filename. The style that ran last is the one in assets/.
render_style() {
  local style=$1
  local profile outdir
  local stable="$ROOT/local/Entropy-Based-Learning.pdf"
  local published="$ROOT/assets/Entropy-Based-Learning.pdf"
  if [[ $style == 1 ]]; then
    profile=pdf
    outdir="$ROOT/_book-pdf"
  else
    profile=pdf2
    outdir="$ROOT/_book-pdf-stix"
  fi

  echo "rendering PDF option ${style} (profile=${profile}) ..."
  if [[ $KEEP_TEX -eq 1 ]]; then
    quarto render --profile "$profile" --to pdf -M keep-tex:true
  else
    quarto render --profile "$profile" --to pdf
  fi

  local pdf_src="" candidate
  for candidate in \
    "$outdir/Entropy-Based-Learning.pdf" \
    "$outdir/"*.pdf
  do
    if [[ -f $candidate ]]; then
      pdf_src=$candidate
      break
    fi
  done
  if [[ -z $pdf_src ]]; then
    echo "error: quarto finished but no PDF was found in ${outdir}/" >&2
    exit 1
  fi

  mkdir -p "$ROOT/local" "$ROOT/assets"
  cp -f "$pdf_src" "$stable"
  cp -f "$pdf_src" "$published"

  local size pages=""
  size=$(du -h "$stable" | awk '{print $1}')
  if command -v pdfinfo >/dev/null 2>&1; then
    pages=$(pdfinfo "$stable" 2>/dev/null | awk '/^Pages:/ {print $2}' || true)
  elif command -v mdls >/dev/null 2>&1; then
    pages=$(mdls -name kMDItemNumberOfPages -raw "$stable" 2>/dev/null || true)
    [[ $pages == "(null)" ]] && pages=""
  fi

  echo
  echo "PDF option ${style} ready:"
  echo "  $pdf_src"
  echo "  $stable"
  echo "  $published"
  if [[ -n $pages ]]; then
    echo "  ${pages} pages, ${size}"
  else
    echo "  ${size}"
  fi

  if [[ $OPEN -eq 1 ]]; then
    if command -v open >/dev/null 2>&1; then
      open "$stable"
    else
      echo "note: 'open' not available; PDF is at $stable"
    fi
  fi
}

if [[ $STYLE == 1 || $STYLE == both ]]; then
  render_style 1
fi
if [[ $STYLE == 2 || $STYLE == both ]]; then
  render_style 2
fi
