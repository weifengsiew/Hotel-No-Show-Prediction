#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$SCRIPT_DIR/stage1.sh"
"$SCRIPT_DIR/stage2.sh"
"$SCRIPT_DIR/stage3.sh"
"$SCRIPT_DIR/stage4.sh"
