#!/usr/bin/env bash
# sync-book-pdf.sh
#
# Refresh the tracked book PDF (assets/Entropy-Based-Learning.pdf) from
# the current sources. This is the file that is committed, pushed, and
# copied onto the GitHub Pages site.
#
# Usage:
#   ./scripts/sync-book-pdf.sh
#   ./scripts/sync-book-pdf.sh --if-stale
#
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  sync-book-pdf.sh [--if-stale]

  (no flags)    Always rebuild the book PDF and copy it to assets/.
  --if-stale    Rebuild only if sources are newer than the tracked PDF,
                or if the tracked PDF is missing.
  -h, --help    Show this help.

The PDF is built locally (quarto + lualatex). GitHub Actions only
publishes the committed file; it does not compile TeX.
EOF
}

IF_STALE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --if-stale) IF_STALE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$ROOT"

PDF="$ROOT/assets/Entropy-Based-Learning.pdf"
RENDER="$ROOT/scripts/render-pdf.sh"

pdf_is_stale() {
  if [[ ! -f $PDF ]]; then
    return 0
  fi
  # BSD find (macOS) has no -quit; a non-empty first hit is enough.
  if find \
    index.qmd 01-intro.qmd summary.qmd references.qmd references.bib \
    _quarto.yml _quarto-pdf.yml custom.scss \
    chapters tex filters code figures \
    \( -type f -o -type l \) \
    ! -name '*.pdf' \
    -newer "$PDF" \
    2>/dev/null | grep -q .
  then
    return 0
  fi
  return 1
}

if [[ $IF_STALE -eq 1 ]]; then
  if pdf_is_stale; then
    echo "tracked PDF is missing or stale; rebuilding ..."
  else
    echo "tracked PDF is current: $PDF"
    exit 0
  fi
fi

if [[ ! -x $RENDER ]]; then
  echo "error: missing executable $RENDER" >&2
  exit 1
fi

"$RENDER"
