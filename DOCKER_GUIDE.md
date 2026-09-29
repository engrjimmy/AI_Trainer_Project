# AI Trainer – Docker Guide

Run training, detection and evaluation inside a container, with the same commands and results as the native setup ([AI_Trainer_Quick_Note.md](AI_Trainer_Quick_Note.md)).

Docker is **optional**. Use it to:
- share one environment across PCs or hand the project to someone else
- keep the trainer's Python packages isolated from ROS and other tools

Tested on: Ubuntu 22.04, RTX 4060 Laptop (8 GB), driver 595.91.07, Docker 29.8.1, NVIDIA Container Toolkit 1.20.1.

---

## 1. Files

All in `object_detection_trainer/`:

| File | Purpose |
|---|---|
| `Dockerfile` | Image `ai_trainer_frcnn`: `python:3.10-slim` + pinned packages (PyTorch 2.7.0 / CUDA 12.6, OpenCV 4.11 headless) |
| `requirements-prod.txt` | Pinned Python packages for the image |
| `.dockerignore` | Keeps `outputs/`, `data/`, weights and logs out of the build |
| `docker_build_run.sh` | Builds the image and runs training |
| `docker_run.sh` | Runs any command in the container (detection, evaluation, export, ...) |

The container mounts three host folders, so data stays on the host and results appear on the host:

| Host | Container |
|---|---|
| `object_detection_trainer/data/` | `/app/data` |
| `object_detection_trainer/data_configs/` | `/app/data_configs` |
| `object_detection_trainer/outputs/` | `/app/outputs` |
| `object_detection_trainer/weights/` | `/app/weights` (`docker_run.sh`, for `export.py`) |
| `~/.cache/torch/` | pretrained weights cache |

It runs as your user, so output files are not owned by root.

---

## 2. Prerequisites (one-time)

**NVIDIA driver** on the host (see Quick Note section 1), then check:
```bash
nvidia-smi
```

**Docker**, with your user in the `docker` group:
```bash
docker --version
groups | grep docker        # if missing: sudo usermod -aG docker $USER, then log out/in
```

**NVIDIA Container Toolkit** (lets containers use the GPU):
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker          # restarts running containers
nvidia-ctk --version
```

---

## 3. Build and test the image

```bash
cd AI_Trainer_Project/object_detection_trainer
docker build -t ai_trainer_frcnn .            # ~4 min first time, ~6.4 GB image
```

GPU visible in the container:
```bash
./docker_run.sh nvidia-smi
./docker_run.sh python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# True NVIDIA GeForce RTX 4060 Laptop GPU
```

Rebuild after changing code or `requirements-prod.txt`. Data and config changes need no rebuild (they are mounted).

---

## 4. Train

**Script** (builds the image if needed, then trains):
```bash
./docker_build_run.sh [EPOCHS] [NAME] [BATCH]

./docker_build_run.sh 5 box_test_quick 8     # quick test, ~1 min
./docker_build_run.sh                        # default: 20 epochs, box_training, batch 8
```

**Same thing with `docker_run.sh`** (any `train.py` option):
```bash
./docker_run.sh python train.py --data data_configs/box.yaml --epochs 5 \
  --model fasterrcnn_resnet50_fpn --name box_test_quick --batch 8 --disable-wandb
```

Output on the host: `outputs/training/<name>/` with `best_model.pth`, `final_model.pth`, `last_model.pth`, `model.onnx` (exported automatically), validation images `image_*_1.jpg`, `results.csv`, plots and `train.log`.

CPU only (no GPU): add `--device cpu --workers 0 --batch 2`.

---

## 5. Evaluate (COCO mAP)

```bash
./docker_run.sh python eval.py --data data_configs/box.yaml \
  --model fasterrcnn_resnet50_fpn \
  --weights outputs/training/box_test_quick/best_model.pth
```

---

## 6. Detect

Each run creates the next `outputs/inference/res_N/` on the host.

**Images, ONNX (CPU, ~2.5 FPS):**
```bash
./docker_run.sh python object_detection_onnx_inference.py --input data/images/test \
  --weights outputs/training/box_test_quick/model.onnx \
  --data data_configs/box.yaml --imgsz 640
```

**Images, PyTorch (GPU, ~15 FPS):**
```bash
./docker_run.sh python inference.py --input data/images/test \
  --weights outputs/training/box_test_quick/best_model.pth \
  --data data_configs/box.yaml --imgsz 640 --threshold 0.5
```

**Both in one command** (ONNX → `res_1`, PyTorch → `res_2` on an empty `outputs/inference`):
```bash
./docker_run.sh python object_detection_onnx_inference.py --input data/images/test --weights outputs/training/box_test_quick/model.onnx --data data_configs/box.yaml --imgsz 640 && \
./docker_run.sh python inference.py --input data/images/test --weights outputs/training/box_test_quick/best_model.pth --data data_configs/box.yaml --imgsz 640 --threshold 0.5 && \
ls outputs/inference/
```

**Video** (the file must be inside a mounted folder, e.g. `data/`):
```bash
./docker_run.sh python inference_video.py --input data/my_video.mp4 \
  --weights outputs/training/box_test_quick/best_model.pth \
  --data data_configs/box.yaml --imgsz 640 --threshold 0.5

./docker_run.sh python onnx_inference_video.py --input data/my_video.mp4 \
  --weights outputs/training/box_test_quick/model.onnx \
  --data data_configs/box.yaml --imgsz 640
```

### Output options

| Option | Effect | In Docker |
|---|---|---|
| (default) | Annotated images/video saved to `outputs/inference/res_N/` | ✅ |
| `--threshold 0.5` | Hide detections below the score | ✅ |
| `--no-labels` | Boxes only, no class/score text | ✅ |
| `--log-json` | Also save `log.json` (COCO-style detections) | ✅ |
| `--show` / `--mpl-show` | Live window | ❌ no display in the container; open the saved files instead |

ONNX image detection always writes `detection_log.json` next to the images.

### Output locations

| Command | Output |
|---|---|
| `object_detection_onnx_inference.py` | `outputs/inference/res_N/*.jpg`, `detection_log.json` |
| `inference.py` | `outputs/inference/res_N/*.jpg` (`log.json` with `--log-json`) |
| `inference_video.py` / `onnx_inference_video.py` | `outputs/inference/res_N/<video name>.mp4` |
| `eval_matrix_onnx_inference.py` | `outputs/training/<name>/evaluation/evalN/` |

View on the host:
```bash
ls outputs/inference/
xdg-open outputs/inference/res_1
```

---

## 7. Evaluate ONNX (precision / recall / F1)

```bash
./docker_run.sh python eval_matrix_onnx_inference.py --input_image data/images/test \
  --ground_truth data/annotations/test \
  --weights outputs/training/box_test_quick/model.onnx \
  --threshold 0.5 --output outputs/training/box_test_quick
```
Results: `outputs/training/box_test_quick/evaluation/evalN/` (annotated images, per-image JSON, `summary.json`).

---

## 8. Other tools

```bash
./docker_run.sh python export.py --weights outputs/training/box_test_quick/best_model.pth \
  --data data_configs/box.yaml --out model.onnx                   # -> weights/model.onnx
./docker_run.sh python test_image_quality_check.py --input data/images/test
./docker_run.sh python split_data.py --data-path data/my_dataset     # dataset must be under data/
```

Use paths inside the mounted folders (`data/`, `data_configs/`, `outputs/`, `weights/`); anything else is not visible in the container.

---

## 9. Tested results

Same dataset, 5 epochs, batch 8:

| | Native | Docker |
|---|---|---|
| Time per epoch | 6–7 s | 6–7 s |
| Validation mAP@0.5:0.95 | 0.666 | 0.670 |
| Validation mAP@0.5 | 0.953 | 0.956 |
| Test mAP@0.5 (`eval.py`) | 0.99 | 0.99 |
| ONNX / PyTorch detection | 2.5 / 15 FPS | 2.5 / 15 FPS |

Verified in the container: training, ONNX export (in `train.py` and `export.py`), `eval.py`, ONNX and PyTorch image detection, both video scripts, `--log-json`, `--no-labels`, `eval_matrix_onnx_inference.py`.

---

## 10. Cleanup

```bash
docker images ai_trainer_frcnn      # size
docker rmi ai_trainer_frcnn         # remove the image
docker builder prune                # remove build cache
```

---

## 11. Troubleshooting

| Symptom | Fix |
|---|---|
| `could not select device driver "" with capabilities: [[gpu]]` | NVIDIA Container Toolkit missing or Docker not restarted (section 2) |
| `permission denied ... docker.sock` | Add your user to the `docker` group and log in again |
| `cv2.error ... cvShowImage ... not implemented` | `--show` needs a display; remove it and open the saved files |
| `FileNotFoundError` / `PermissionError` for a path | Path is outside the mounted folders; move the file into `data/` (inputs) or write to `outputs/` |
| Code change has no effect | Rebuild: `docker build -t ai_trainer_frcnn .` |
| `CUDA out of memory` | Lower the batch size (`./docker_build_run.sh 20 box_training 4`) |
| DataLoader `bus error` / shared memory errors | Increase `--shm-size` in the scripts or lower `--workers` |
