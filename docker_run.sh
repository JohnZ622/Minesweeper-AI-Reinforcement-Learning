#!/bin/bash

# Use the current directory name as the container name
CONTAINER_NAME="ray_dev_$(basename "$PWD")"

docker run -it --rm \
  --name "$CONTAINER_NAME" \
  --gpus all \
  --shm-size=2gb \
  -v "$(pwd):/home/ray/project" \
  -w /home/ray/project \
  -p 8265:8265 \
  rayproject/ray-ml:latest-py310-gpu \
  bash
