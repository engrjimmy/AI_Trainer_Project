# AI Trainer – Quick Note (Native Ubuntu + NVIDIA GPU)

Tested on: Dell G16 7630, RTX 4060 Laptop (8 GB), Ubuntu 22.04, kernel 6.8.0-138 (HWE), Secure Boot ON.
Project: `AI_Trainer_Project` (branch `development`). Run the commands below from the folder that contains it.

---

## 1. NVIDIA driver (one-time)

Check what the GPU needs (read-only):
```bash
lspci -nn | grep -iE "vga|3d"
ubuntu-drivers devices          # look for "recommended"
mokutil --sb-state
```

Install the recommended driver **with Canonical's pre-signed kernel module**. With this package there's no DKMS build and no MOK enrollment, even with Secure Boot on.
```bash
sudo apt update
sudo apt install nvidia-driver-595-open linux-modules-nvidia-595-open-generic-hwe-22.04
sudo reboot
```
If apt asks to install `nvidia-dkms-*` or asks for a MOK password, answer **n**. That means the signed module isn't being used.

Verify:
```bash
nvidia-smi
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
You don't need the CUDA toolkit, because PyTorch (cu126) ships its own CUDA runtime.

Recovery (black screen or no login after reboot): press `Ctrl+Alt+F3`, log in, then run:
```bash
sudo apt purge '^nvidia-.*' '^libnvidia-.*' '^linux-modules-nvidia-.*'
sudo reboot
```
Fallback driver: `nvidia-driver-580-open linux-modules-nvidia-580-open-generic-hwe-22.04`.

---

## 2. Project

```bash
cd AI_Trainer_Project
git checkout development && git pull
cd object_detection_trainer
```

Data (already set in `data_configs/box.yaml`, relative to `train.py`):
```
data/images/{train,val,test}        70 / 8 / 10 images
data/annotations/{train,val,test}   Pascal VOC XML
```

Check the images:
```bash
python3 test_image_quality_check.py --input data/images/test
```

---

## 3. Train

```bash
# quick test (5 epochs, ~1 min on RTX 4060)
python3 train.py --data data_configs/box.yaml --epochs 5 \
  --model fasterrcnn_resnet50_fpn --name box_test_quick \
  --batch 8 --disable-wandb

# full run
python3 train.py --data data_configs/box.yaml --epochs 20 \
  --model fasterrcnn_resnet50_fpn --name box_training \
  --batch 8 --disable-wandb
```
- Without `--disable-wandb`, answer `3` at the W&B prompt.
- Out of GPU memory: use `--batch 4`.
- Same `--name` reuses the folder and overwrites the old files.
- Alternative: `cd .. && ./train.sh --gpu --batch 4 --epochs 5 --name box_test_quick`.

Output: `outputs/training/<name>/`
| File | Purpose |
|---|---|
| `best_model.pth` | best validation mAP; use this for PyTorch inference/eval |
| `final_model.pth` / `last_model.pth` | last epoch |
| `model.onnx` | exported automatically at the end of training |
| `image_*_1.jpg` | validation predictions |
| `results.csv`, `*.png`, `train.log` | metrics, plots, log |

---

## 4. Evaluate (PyTorch, COCO mAP)

```bash
python3 eval.py --data data_configs/box.yaml \
  --model fasterrcnn_resnet50_fpn \
  --weights outputs/training/box_test_quick/best_model.pth
```

---

## 5. Detect on test images

Each run creates the next `outputs/inference/res_N`.

Both in one command (ONNX → `res_1`, then PyTorch/GPU → `res_2`):
```bash
python3 object_detection_onnx_inference.py --input data/images/test --weights outputs/training/box_test_quick/model.onnx --data data_configs/box.yaml --imgsz 640 && \
python3 inference.py --input data/images/test --weights outputs/training/box_test_quick/best_model.pth --data data_configs/box.yaml --imgsz 640 --threshold 0.5 && \
ls outputs/inference/
```

| Script | Model | Device | Speed |
|---|---|---|---|
| `object_detection_onnx_inference.py` | `model.onnx` | CPU (onnxruntime) | ~2.5 FPS |
| `inference.py` | `best_model.pth` | GPU | ~15 FPS |

View:
```bash
xdg-open outputs/inference/res_1
xdg-open outputs/inference/res_2
```

---

## 6. Evaluate ONNX (precision / recall / F1)

```bash
python3 eval_matrix_onnx_inference.py --input_image data/images/test \
  --ground_truth data/annotations/test \
  --weights outputs/training/box_test_quick/model.onnx \
  --threshold 0.5 --output outputs/training/box_test_quick
```
Results are written to `outputs/training/box_test_quick/evaluation/evalN/` (images, per-image JSON, `summary.json`).

Reference (5 epochs, test set, threshold 0.5): P 0.88 / R 0.98 / F1 0.93, mAP@0.5 = 0.99.

---

## 7. Other tools

```bash
python3 export.py --weights outputs/training/box_training/best_model.pth --data data_configs/box.yaml --out model.onnx   # -> weights/model.onnx
python3 split_data.py --data-path <dir with images/ and Annotations/>     # 80/10/10 split
```

---

## 8. Docker (optional)

Not needed on this PC: the native setup above runs everything on the GPU. Use Docker to share one environment across PCs or to isolate the trainer's packages from ROS.

Full procedure: [DOCKER_GUIDE.md](DOCKER_GUIDE.md). In short (from `object_detection_trainer/`):
```bash
# one-time: NVIDIA Container Toolkit (DOCKER_GUIDE.md section 2), then
docker build -t ai_trainer_frcnn .
./docker_run.sh nvidia-smi                                    # GPU visible in container

./docker_build_run.sh 5 box_test_quick 8                      # train: EPOCHS NAME BATCH
./docker_run.sh python inference.py --input data/images/test \
  --weights outputs/training/box_test_quick/best_model.pth \
  --data data_configs/box.yaml --imgsz 640 --threshold 0.5    # any script works the same way
```
Results land in the same host folders (`outputs/training/`, `outputs/inference/res_N/`). `--show` is not available in the container.

---

## 9. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `nvidia-smi: command not found` | Driver not installed (see section 1) |
| `torch.cuda.is_available() == False` | Driver not loaded; reboot, then check `nvidia-smi` |
| `cv2.error ... img.depth() == CV_8U in putText` | OpenCV 5 draws only on 8-bit images; fixed in `utils/general.py` (validation images) |
| `cv2.error ... destroyAllWindows ... not implemented` | Headless OpenCV has no GUI; fixed in `inference.py`, `inference_video.py`, `onnx_inference_image.py`, `onnx_inference_video.py`, `object_detection_onnx_inference.py` (window calls only with `--show`) |
| `--show` or `--vis-transformed` fails / no window | Needs OpenCV with GUI. `opencv-python-headless` 5.0 (pulled in by albumentations) replaced 4.11 in `cv2`. To restore 4.11 with GUI:<br>`pip uninstall -y opencv-python opencv-python-headless`<br>`pip install --user "opencv-python-headless==4.11.0.86"`<br>`pip install --user --force-reinstall --no-deps "opencv-python==4.11.0.86"`<br>Check: `python3 -c "import cv2; print(cv2.__version__)"` |
| Which OpenCV works? | All scripts tested on both 5.0.0 (headless) and 4.11.0 (GUI) with the same results; without `--show`, no downgrade is needed |
| `CUDA out of memory` | Lower `--batch` (8 uses ~4.5 GB, 4 uses ~2.5 GB) |
| No `outputs/inference` after training | Expected; only detection scripts (section 5) create it |
| TensorFlow / Axes3D / autocast / TracerWarning / onnxruntime "Removing initializer" messages | Harmless warnings |
