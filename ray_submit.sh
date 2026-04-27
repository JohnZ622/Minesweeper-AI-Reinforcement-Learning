#!/bin/bash

# Third line is the IP address
HEAD_IP=$(ray get-head-ip ray_cluster.yaml | sed -n '3p')

ray job submit \
  --address "http://${HEAD_IP}:8265" \
  --working-dir . \
  --runtime-env-json '{"excludes": ["models/", "replay/", "logs/", "logs-bak/", ".git/", "__pycache__/", "notes/", ".venv/", ".vscode/", "img/"], "env_vars": {"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy"}}' \
  -- python3 tune_sweep.py "$@"
