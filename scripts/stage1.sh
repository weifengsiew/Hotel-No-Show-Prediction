#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DATA_PATH="$PROJECT_ROOT/data/raw/noshow.db"
RAW_DATA_URL="https://techassessment.blob.core.windows.net/aiap-pys-2/noshow.db"

if [ -f "$RAW_DATA_PATH" ]; then
  printf '\nRaw data already exists: %s\n' "$RAW_DATA_PATH"
else
  mkdir -p "$(dirname "$RAW_DATA_PATH")"

  printf '\n==> Downloading raw data\n'
  curl -fL "$RAW_DATA_URL" -o "$RAW_DATA_PATH"

  printf '\nDownloaded raw data to %s\n' "$RAW_DATA_PATH"
fi
