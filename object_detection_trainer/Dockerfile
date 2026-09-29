FROM python:3.10-slim

WORKDIR /app

# libgomp1: OpenMP runtime for PyTorch / onnxruntime; libglib2.0-0: OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements-prod.txt /app/
# deep-sort-realtime pulls opencv-python (GUI); install it without deps so
# only the pinned opencv-python-headless provides cv2.
RUN pip install --no-cache-dir -r requirements-prod.txt && \
    pip install --no-cache-dir --no-deps deep-sort-realtime==1.3.2

COPY . /app

CMD ["python", "train.py", "--data", "data_configs/box.yaml", "--disable-wandb"]
