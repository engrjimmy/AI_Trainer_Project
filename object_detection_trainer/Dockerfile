# Use the PyTorch image with the latest CUDA support as the base image
FROM nvidia/cuda:12.4.0-runtime-ubuntu20.04

# Set the working directory in the container
WORKDIR /app

# Copy the project files into the container
COPY . /app

# Copy the data and configuration files into the container
COPY data /app/data
COPY data_configs /app/data_configs

# Install necessary system dependencies
RUN apt-get update && \
    apt-get install -y \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements-prod.txt file and install the Python packages
COPY requirements-prod.txt /app/
RUN pip install --upgrade pip && \
    pip install -r requirements-prod.txt

# Expose the port the app runs on (if applicable)
EXPOSE 8000

# Define the command to run the application
CMD ["python", "train.py"]
