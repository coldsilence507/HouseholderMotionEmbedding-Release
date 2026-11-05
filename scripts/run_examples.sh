#!/usr/bin/env bash
# Run the example launchers in assets/*/run.sh.
#   bash scripts/run_examples.sh              # all examples
#   bash scripts/run_examples.sh 'davis_*'    # only folders matching a glob
# Pass-through env vars: LORA_PATH, OUT_DIR.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."   # repository root
PATTERN=${1:-*}
for s in assets/$PATTERN/run.sh; do
    [ -f "$s" ] || { echo "No example matches '$PATTERN'"; exit 1; }
    echo "=== $s ==="
    bash "$s"
done
