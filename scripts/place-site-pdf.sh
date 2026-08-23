#!/usr/bin/env bash
# place-site-pdf.sh
#
# Quarto post-render hook for the HTML site. Copies the tracked book PDF
# to the site root so `book.downloads: [pdf]` resolves to
# /Entropy-Based-Learning.pdf on GitHub Pages.
#
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
SRC="$ROOT/assets/Entropy-Based-Learning.pdf"
DEST_DIR="$ROOT/_book"
DEST="$DEST_DIR/Entropy-Based-Learning.pdf"

if [[ ! -f $SRC ]]; then
  echo "error: tracked PDF missing: $SRC" >&2
  echo "Build it locally with ./scripts/sync-book-pdf.sh and commit it." >&2
  exit 1
fi

if [[ ! -d $DEST_DIR ]]; then
  echo "error: HTML output directory missing: $DEST_DIR" >&2
  echo "This script is meant to run as a Quarto post-render step." >&2
  exit 1
fi

cp -f "$SRC" "$DEST"
echo "placed site PDF: $DEST"
