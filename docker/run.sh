#!/bin/bash

TAG='funahashi/ga-generator-pytorch:2.0.0-cuda11.7-cudnn8-devel'
PROJECT_DIR="$(cd "$(dirname "${0}")/.." || exit; pwd)"

# run
docker run -it --rm \
  --shm-size=8g \
  --gpus all \
  -v "${PROJECT_DIR}:/work" \
  -w "/work" \
  "${TAG}" \
  bash