# AI Trainer Project

Object Detection Model Training Framework using PyTorch Faster R-CNN

Repository: AI_Trainer_Project

---

## System Requirements

### Operating System
- Ubuntu 22.04 / 20.04 (recommended)
- Other Linux distributions
- Windows WSL2

### Python Environment
- Python 3.8 or higher (Python 3.10 recommended)
- pip3 package manager

### Hardware Specifications

**CPU Training Configuration:**
- 4+ CPU cores
- 8GB+ RAM
- Training duration: ~2-4 hours for 50 epochs
- Batch size: 2-4

**GPU Training Configuration:**
- NVIDIA GPU (GTX 1060 or better)
- NVIDIA drivers installed
- CUDA 11.3 or higher
- 4GB+ VRAM
- Training duration: ~20-40 minutes for 50 epochs
- Batch size: 8-32

---

## Installation

### Step 1: Install Dependencies

**IMPORTANT: Run this command first before training**

Navigate to the project directory and install required Python packages:

```bash
git clone https://github.com/engrjimmy/AI_Trainer_Project.git
cd AI_Trainer_Project/object_detection_trainer
pip3 install --user -r requirements.txt
```

### Step 2: Verify Installation

```bash
# Verify PyTorch
python3 -c "import torch; print('PyTorch:', torch.__version__)"

# Check CUDA availability (GPU)
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Verify dependencies
python3 -c "import cv2, torchvision, yaml; print('All dependencies OK')"
```

### Step 3: Fix Known Issues (if needed)

If albumentations compatibility errors occur:
```bash
pip3 install --user albumentations==1.3.1 --force-reinstall
```

### Optional: Virtual Environment Setup

For isolated environment management:

```bash
cd AI_Trainer_Project/object_detection_trainer
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Project Structure

```
AI_Trainer_Project/
├── object_detection_trainer/     # Core training framework
│   ├── train.py                  # Main training script
│   ├── eval.py                   # Model evaluation
│   ├── inference.py              # Image inference
│   ├── inference_video.py        # Video inference
│   ├── export.py                 # ONNX export utility
│   ├── data_configs/             # Dataset configurations
│   │   └── box.yaml              # Box detection config
│   ├── data/                     # Training dataset
│   │   ├── images/               # Image files
│   │   │   ├── train/            # 70 training images
│   │   │   ├── val/              # 8 validation images
│   │   │   └── test/             # 10 test images
│   │   └── annotations/          # Pascal VOC XML
│   │       ├── train/            # Training annotations
│   │       ├── val/              # Validation annotations
│   │       └── test/             # Test annotations
│   ├── models/                   # Model architectures
│   ├── utils/                    # Utility functions
│   ├── torch_utils/              # PyTorch utilities
│   └── requirements.txt          # Python dependencies
├── train.sh                      # Training wrapper script
├── LICENSE                       # License information
└── README.md                     # This file
```

---

## 1. Training Setup

### 1.1 GPU environment

Use this first to confirm the GPU is active:
```bash
cd AI_Trainer_Project/object_detection_trainer
nvidia-smi
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

Expected output:
```text
PyTorch: 2.4.0+cu121
CUDA available: True
```

If the result shows `CUDA available: True`, use the GPU commands below.

### 1.2 CPU environment

If no GPU is available, run the CPU version with `--device cpu`:
```bash
cd AI_Trainer_Project/object_detection_trainer
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

Expected output:
```text
PyTorch: 2.4.0+cpu
CUDA available: False
```

This confirms the project is running on CPU mode and should use the CPU training command below.

---

## 2. Training Commands

### Quick test (2 epochs, ~5-10 min)
```bash
cd AI_Trainer_Project/object_detection_trainer

python3 train.py \
  --data data_configs/box.yaml \
  --epochs 2 \
  --model fasterrcnn_resnet50_fpn \
  --name box_test_quick \
  --batch 8

# When asked "wandb: Enter your choice:" - Type: 3 (Don't visualize)
```

### Full training (20 epochs, ~1-2 hours)
```bash
cd AI_Trainer_Project/object_detection_trainer

python3 train.py \
  --data data_configs/box.yaml \
  --epochs 20 \
  --model fasterrcnn_resnet50_fpn \
  --name box_training \
  --batch 16
```

### CPU training (original PC test flow)
```bash
cd AI_Trainer_Project/object_detection_trainer

python3 train.py \
  --data data_configs/box.yaml \
  --epochs 2 \
  --model fasterrcnn_resnet50_fpn \
  --name box_test_quick \
  --batch 2 \
  --device cpu \
  --workers 0 \
  --disable-wandb
```

This is the original CPU-style test path used in the project when no NVIDIA CUDA driver is available or when running on a standard PC.

For a wrapper-based CPU run, the project also supports:
```bash
cd AI_Trainer_Project
./train.sh --cpu --epochs 2 --batch 2 --name box_cpu_test
```

This wrapper uses `nice -n 19` and sets low-priority CPU execution automatically.

**Training output location:**
```text
outputs/training/box_training/
# or outputs/training/box_test_quick/
# or outputs/training/box_cpu_test/
```

---

## 3. Find Models

**Best model (use this):**
```bash
outputs/training/box_training/best_model.pth
```

**ONNX model (for deployment):**
```bash
outputs/training/box_training/model.onnx
```

**Other models:**
- `final_model.pth` - Last epoch
- `last_model.pth` - Latest checkpoint

---

## 4. Run Detection (ONNX)
```bash
cd AI_Trainer_Project/object_detection_trainer

python3 object_detection_onnx_inference.py \
  --input /path/to/test/images \
  --weights outputs/training/box_training/model.onnx \
  --data data_configs/box.yaml \
  --imgsz 640
```

**Detection results saved in:**
```text
outputs/inference/res_X/
# where X is auto-incremented: res_1, res_2, res_3, etc.
```

---

## 5. Check Detection Images
```bash
cd AI_Trainer_Project/object_detection_trainer

find outputs/inference -name "*.jpg" | head -10
ls -lh outputs/inference/res_1/*.jpg
ls -lh outputs/inference/res_2/*.jpg
find outputs/inference -name "*.jpg" | wc -l
```

---

## 6. Run Evaluation
```bash
cd AI_Trainer_Project/object_detection_trainer

python3 eval.py \
  --data data_configs/box.yaml \
  --weights outputs/training/box_training/best_model.pth \
  --model fasterrcnn_resnet50_fpn
```

**Evaluation results:**
```text
outputs/training/box_training/evaluation/
```

---

## 7. Check Training Results
```bash
cd AI_Trainer_Project/object_detection_trainer/outputs/training/box_training

cat results.csv
cat train.log
ls *.png
```

---

## 8. Quick Verify Everything Works
```bash
cd AI_Trainer_Project/object_detection_trainer

ls -lh outputs/training/box_training/*.pth
ls -lh outputs/training/box_training/*.onnx

tail outputs/training/box_training/results.csv
ls outputs/training/box_training/image_*.jpg
```

---

## 9. Key Files Summary

| File | Location | Purpose |
|------|----------|---------|
| **Training script** | `train.py` | Train model |
| **Best model** | `outputs/training/box_training/best_model.pth` | **Use this** |
| **ONNX model** | `outputs/training/box_training/model.onnx` | **Deployment** |
| **Training results** | `outputs/training/box_training/results.csv` | mAP, loss |
| **Detection script** | `object_detection_onnx_inference.py` | Run inference |
| **Eval script** | `eval.py` | Get accuracy |
| **Config** | `data_configs/box.yaml` | Dataset config |

---

## 10. Quick Test Example Results (2 epochs)
- **mAP@0.5:** 74.6%
- **mAP@0.5-0.95:** 41.9%
- **Training time:** ~5-10 minutes
- **Location:** `outputs/training/box_test_quick/best_model.pth`
- **ONNX:** `outputs/training/box_test_quick/model.onnx`
- **Detection results:** `outputs/inference/res_X/*.jpg`

---

## Available Models

- `fasterrcnn_resnet50_fpn` - FasterRCNN with ResNet50 backbone (default, recommended)
- `fasterrcnn_resnet101_fpn` - FasterRCNN with ResNet101 backbone (larger model)
- `fasterrcnn_resnet152_fpn` - FasterRCNN with ResNet152 backbone (largest model)
- `fasterrcnn_mobilenetv3_large_fpn` - FasterRCNN with MobileNetV3 backbone (lightweight)

Refer to `models/` directory for complete model list.

---

## Dataset Format

**Images:** PNG or JPG format  
**Annotations:** Pascal VOC XML format

Example annotation structure:
```xml
<annotation>
  <filename>image_001.png</filename>
  <size>
    <width>640</width>
    <height>480</height>
  </size>
  <object>
    <name>box</name>
    <bndbox>
      <xmin>100</xmin>
      <ymin>200</ymin>
      <xmax>300</xmax>
      <ymax>400</ymax>
    </bndbox>
  </object>
</annotation>
```

---

## Troubleshooting

### GPU Issues

**Error: "Found no NVIDIA driver on your system"**
- Solution: Add `--device cpu` flag to training command

**Error: "CUDA out of memory"**
- Solution: Reduce `--batch` parameter (try 8, 4, or 2)

**Verify GPU:**
```bash
nvidia-smi
python3 -c "import torch; print(torch.cuda.is_available())"
```

### Training Issues

**OpenCV errors during validation image saving**
- Already fixed: `SAVE_VALID_PREDICTION_IMAGES: False` in box.yaml

**Albumentations version compatibility errors**
```bash
pip3 install --user albumentations==1.3.1 --force-reinstall
```

**System freezing during CPU training**
- Training script uses `nice -n 19` for low priority execution
- Close unnecessary applications
- Reduce batch size to 2
- Ensure sufficient RAM (8GB+ recommended)

**WandB interactive prompts**
- Use `--disable-wandb` flag to skip W&B logging

---

## Best Practices

### CPU Training
- Use batch size 2-4
- Set workers to 0
- Use nice priority for system responsiveness
- Expected training time: 2-4 hours for 50 epochs
- Suitable for development and testing

### GPU Training
- Use batch size 8-32 (depending on VRAM)
- Set workers to 4-8
- Monitor GPU utilization with `nvidia-smi`
- Expected training time: 20-40 minutes for 50-100 epochs
- Recommended for production training

### Dataset Recommendations
- Minimum 50 images per class
- Balance training/validation split (80/20 or 90/10)
- Use data augmentation (already configured in datasets.py)
- Validate annotations before training

---

## Performance Metrics

Training metrics logged during execution:
- Loss (classification + regression)
- mAP (mean Average Precision)
- mAP@0.5 (IoU threshold = 0.5)
- mAP@0.75 (IoU threshold = 0.75)

Evaluation uses COCO metrics standard.

---

## Dataset and Usage Restrictions

This project includes a shared dataset and associated annotations that are proprietary to QibiTech Inc. and must be used only under explicit permission.

### Important policy
- The dataset files in this repository are not open-public data.
- All dataset contents remain the copyright property of QibiTech Inc.
- Use of the dataset for research, training, evaluation, or redistribution requires prior written permission.
- Reuse, sharing, or publication of any dataset subset or derived data must be approved before distribution.
- Any external use, collaboration, or transfer must be coordinated with Jimmy Majumder and QibiTech Inc. stakeholders.
- For research purposes, dataset use is permitted only after approval and only when collaboration with Jimmy is in place.
- If you do not have permission, do not use the dataset or any files from the shared dataset folder.
- Do not upload, copy, or redistribute the dataset to other repositories, systems, or public locations without approval.

### Research collaboration requirement
For any research-related use of the dataset, the project must involve or be approved by Jimmy Majumder. External or independent use without such collaboration is not permitted.

### Public repository note
This repository is shared for project visibility and documentation only. The dataset and any derived materials remain restricted until explicit permission is granted by the copyright owner and project lead.

---

## License

Proprietary software owned by QibiTech Inc.

For licensing inquiries, contact: j.majumder@qibitech.com

---

## Contact & Author Information

**Author:** Jimmy Majumder  
**Position:** Sr. Robotics Engineer  
**Organization:** QibiTech Inc.  
**Email:** j.majumder@qibitech.com  
**Date:** 2024_Jun  

**Copyright (c) 2024 QibiTech Inc. All rights reserved.**

**NOTICE:** This code and data is for research and testing purposes only.  
DO NOT use for product development or production phase.  
For commercial use or inquiries, contact: j.majumder@qibitech.com

---

## Version History

- 2024_Jun: Initial release
- Dataset: 70 train / 8 validation / 10 test images
- Model: Faster R-CNN ResNet50 FPN
- Classes: background, box
