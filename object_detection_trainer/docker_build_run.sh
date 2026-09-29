#!/bin/bash

# Define the image name
IMAGE_NAME="ai_trainer_frcnn"

# Build the Docker image
echo "Building Docker image: $IMAGE_NAME..."
docker build -t $IMAGE_NAME .

# Check if the build was successful
if [ $? -ne 0 ]; then
  echo "Docker build failed. Exiting."
  exit 1
fi

# Run the Docker container with the training command and increased shared memory
echo "Running Docker container: $IMAGE_NAME..."
docker run --rm -it --gpus all \
    --shm-size=20g \
    -v "$(pwd)/data:/app/data" \
    -v "$(pwd)/data_configs:/app/data_configs" \
    -v "$(pwd)/outputs:/app/outputs" \
    $IMAGE_NAME \
    python3 train.py --data /app/data_configs/box.yaml --epochs 20 --model fasterrcnn_resnet50_fpn --name box_training --batch 16

# Check if the run was successful
if [ $? -ne 0 ]; then
  echo "Docker run failed. Exiting."
  exit 1
fi

echo "Docker image built and training run successfully."
