#!/bin/bash
# Build the image and run training in a container.
# Usage: ./docker_build_run.sh [EPOCHS] [NAME] [BATCH]

IMAGE_NAME="ai_trainer_frcnn"
EPOCHS="${1:-20}"
NAME="${2:-box_training}"
BATCH="${3:-8}"

cd "$(dirname "$0")" || exit 1

echo "Building Docker image: $IMAGE_NAME..."
docker build -t $IMAGE_NAME .

if [ $? -ne 0 ]; then
  echo "Docker build failed. Exiting."
  exit 1
fi

# Interactive flags only when attached to a terminal.
TTY_FLAGS=""
[ -t 0 ] && TTY_FLAGS="-it"

# Run as the host user so outputs are not root-owned; reuse the host
# torch cache for pretrained weights.
mkdir -p outputs "$HOME/.cache/torch"

echo "Running Docker container: $IMAGE_NAME..."
docker run --rm $TTY_FLAGS --gpus all \
    --shm-size=8g \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -v "$HOME/.cache/torch:/tmp/.cache/torch" \
    -v "$(pwd)/data:/app/data" \
    -v "$(pwd)/data_configs:/app/data_configs" \
    -v "$(pwd)/outputs:/app/outputs" \
    $IMAGE_NAME \
    python train.py --data data_configs/box.yaml --epochs "$EPOCHS" \
    --model fasterrcnn_resnet50_fpn --name "$NAME" --batch "$BATCH" --disable-wandb

if [ $? -ne 0 ]; then
  echo "Docker run failed. Exiting."
  exit 1
fi

echo "Training complete. Results: outputs/training/$NAME/"
