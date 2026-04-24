#!/bin/bash

docker run -it \
  --gpus all \
  --shm-size=2gb \
  -v "$(pwd):/home/ray/project" \
  -w /home/ray/project \
  -p 8265:8265 \
  minesweeper-ai:latest \
  bash
