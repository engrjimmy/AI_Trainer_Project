#!/bin/bash
################################################################################
# AI Trainer - Quick Training Script
################################################################################
#
# Author:        Jimmy Majumder
# Position:      Sr. Robotics Engineer
# Organization:  QibiTech Inc.
# Email:         j.majumder@qibitech.com
# Date:          2024_Jun
#
# Copyright (c) 2024 QibiTech Inc.
# All rights reserved.
#
# NOTICE:
# This code and data is for research and testing purposes only.
# DO NOT use for product development or production phase.
# For commercial use or inquiries, contact: j.majumder@qibitech.com
#
################################################################################

cd "$(dirname "$0")/object_detection_trainer"

echo "================================"
echo "AI Trainer - Quick Start"
echo "================================"
echo ""

# Default values
EPOCHS=2
BATCH=2
MODEL="fasterrcnn_resnet50_fpn"
NAME="box_training"
DEVICE="cpu"
WORKERS=0

# Detect if GPU is available
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        echo "🎮 GPU detected! Using CUDA acceleration."
        DEVICE=""  # Auto-detect (will use GPU)
        BATCH=8   # Larger batch for GPU
        WORKERS=4
    else
        echo "💻 No GPU detected. Using CPU."
    fi
else
    echo "💻 No GPU detected. Using CPU."
fi
echo ""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch)
            BATCH="$2"
            shift 2
            ;;
        --name)
            NAME="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --cpu)
            DEVICE="cpu"
            WORKERS=0
            BATCH=2
            echo "🔧 Forced CPU mode"
            shift 1
            ;;
        --gpu)
            DEVICE=""
            WORKERS=4
            BATCH=8
            echo "🔧 Forced GPU mode"
            shift 1
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--epochs N] [--batch N] [--name NAME] [--model MODEL] [--cpu] [--gpu]"
            exit 1
            ;;
    esac
done

echo "Starting training with:"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH"
echo "  Model: $MODEL"
echo "  Name: $NAME"
if [ -z "$DEVICE" ]; then
    echo "  Device: GPU (CUDA)"
else
    echo "  Device: $DEVICE"
fi
echo "  Workers: $WORKERS"
echo ""
echo "Training output will be saved to: outputs/training/$NAME/"
echo ""

# Build command
CMD="nice -n 19 python3 train.py \
    --data data_configs/box.yaml \
    --epochs \"$EPOCHS\" \
    --model \"$MODEL\" \
    --name \"$NAME\" \
    --batch \"$BATCH\" \
    --disable-wandb \
    --workers \"$WORKERS\""

# Add device flag only if set to cpu
if [ ! -z "$DEVICE" ]; then
    CMD="$CMD --device \"$DEVICE\""
fi

# Run training
eval "$CMD" 2>&1 | tee "training_${NAME}.log"

echo ""
echo "================================"
echo "Training complete!"
echo "Results saved to: runs/training/$NAME/"
echo "Log file: training_${NAME}.log"
echo "================================"
