#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONSTANTS="$SCRIPT_DIR/common_constants.py"
MAX_CLICKS=2_000_000

for val in 10 50 100 200 300 400 500 600 700 800 900 1000 1500 2000; do
    echo "=== Starting run: TRAIN_EVERY_N_CLICKS=$val ==="
    sed -i "s/^TRAIN_EVERY_N_CLICKS = .*/TRAIN_EVERY_N_CLICKS = $val/" "$CONSTANTS"
    cd "$SCRIPT_DIR"
    uv run python train.py --max_clicks $MAX_CLICKS
    echo "=== Finished run: TRAIN_EVERY_N_CLICKS=$val ==="
done

# Restore default
sed -i "s/^TRAIN_EVERY_N_CLICKS = .*/TRAIN_EVERY_N_CLICKS = 500/" "$CONSTANTS"
echo "All runs complete. Restored TRAIN_EVERY_N_CLICKS=500."