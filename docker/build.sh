#!/bin/bash

TAG='funahashi/ga-generator-pytorch:2.0.0-cuda11.7-cudnn8-devel'
PROJECT_DIR="$(cd "$(dirname "${0}")/.." || exit; pwd)"

# build
cd "$(dirname "${0}")/.." || exit
DOCKER_BUILDKIT=1 docker build --progress=plain -t ${TAG} docker