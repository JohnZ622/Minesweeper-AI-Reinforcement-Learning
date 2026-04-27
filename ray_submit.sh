#!/bin/bash

# Third line is the IP address
HEAD_IP=$(ray get-head-ip ray_cluster.yaml | sed -n '3p')

RUNTIME_ENV=$(cat <<EOF
{
  "excludes": ["models/", "replay/", "logs/", "logs-bak/", ".git/", "__pycache__/", "notes/", ".venv/", ".vscode/", "img/"],
  "env_vars": {
    "SDL_VIDEODRIVER": "dummy",
    "SDL_AUDIODRIVER": "dummy",
    "WANDB_API_KEY": "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"
  }
}
EOF
)

ray job submit \
  --address "http://${HEAD_IP}:8265" \
  --working-dir . \
  --runtime-env-json "$RUNTIME_ENV" \
  -- python3 tune_sweep.py "$@"
