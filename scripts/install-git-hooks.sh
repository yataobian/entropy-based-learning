#!/usr/bin/env bash
# install-git-hooks.sh
#
# Copy tracked hooks from .githooks/ into .git/hooks/ (no git-config change).
#
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
SRC_DIR="$ROOT/.githooks"
DEST_DIR="$ROOT/.git/hooks"

if [[ ! -d $ROOT/.git ]]; then
  echo "error: $ROOT is not a git working tree" >&2
  exit 1
fi

mkdir -p "$DEST_DIR"
installed=0
for hook in pre-push; do
  if [[ ! -f $SRC_DIR/$hook ]]; then
    echo "error: missing $SRC_DIR/$hook" >&2
    exit 1
  fi
  cp -f "$SRC_DIR/$hook" "$DEST_DIR/$hook"
  chmod +x "$DEST_DIR/$hook"
  echo "installed .git/hooks/$hook"
  installed=1
done

if [[ $installed -eq 0 ]]; then
  echo "error: no hooks installed" >&2
  exit 1
fi
