#!/usr/bin/env bash
# Render a PLY file and save a timestamped screenshot.
# Usage: ./scripts/render_screenshot.sh <ply_path> <label> [side]
#
# Saves to .claude/screenshots/<label>_<timestamp>.png
# Also prints the path to stdout as SCREENSHOT_PATH=<path>

set -e

PLY_PATH="${1:?Usage: $0 <ply_path> <label> [side]}"
LABEL="${2:?Need label}"
SIDE="${3:-1}"  # default -Z view

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCREENSHOTS_DIR="$ROOT/.claude/screenshots"
mkdir -p "$SCREENSHOTS_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_PATH="$SCREENSHOTS_DIR/${LABEL}_${TIMESTAMP}.png"

echo "Rendering: $PLY_PATH"
echo "Side: $SIDE"
echo "Output: $OUT_PATH"

# Convert PLY path to a URL path relative to server root
# screenshot.js serves from ROOT, so strip ROOT prefix for URL
REL_PLY="${PLY_PATH#$ROOT}"
# If it doesn't start with /, add it
[[ "$REL_PLY" == /* ]] || REL_PLY="/$REL_PLY"

node "$ROOT/scripts/screenshot.js" "$REL_PLY" "$OUT_PATH" "$SIDE"

echo "SCREENSHOT_PATH=$OUT_PATH"
