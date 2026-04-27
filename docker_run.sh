#!/bin/bash

docker run -it --rm \
  --gpus all \
  --shm-size=2gb \
  -v "$(pwd):/home/ray/project" \
  -w /home/ray/project \
  -p 8265:8265 \
  -p 6006:6006 \
  minesweeper-ai:latest \
  bash
