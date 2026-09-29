#!/bin/bash
# Run any command inside the ai_trainer_frcnn container with the project
# data, configs and outputs mounted. Build the image first with
# ./docker_build_run.sh or: docker build -t ai_trainer_frcnn .
# Usage: ./docker_run.sh python inference.py --input data/images/test ...

IMAGE_NAME="ai_trainer_frcnn"

cd "$(dirname "$0")" || exit 1

if [ $# -eq 0 ]; then
  echo "Usage: $0 <command> [args...]"
  exit 1
fi

TTY_FLAGS=""
[ -t 0 ] && TTY_FLAGS="-it"

mkdir -p outputs weights "$HOME/.cache/torch"

docker run --rm $TTY_FLAGS --gpus all \
    --shm-size=8g \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -v "$HOME/.cache/torch:/tmp/.cache/torch" \
    -v "$(pwd)/data:/app/data" \
    -v "$(pwd)/data_configs:/app/data_configs" \
    -v "$(pwd)/outputs:/app/outputs" \
    -v "$(pwd)/weights:/app/weights" \
    $IMAGE_NAME \
    "$@"
