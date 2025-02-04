# Use NVIDIA CUDA runtime base image with Ubuntu 22.04 (comes with Python 3.10)
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Set non-interactive mode for apt-get to avoid prompts (e.g., tzdata)
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies and Python3-pip
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch \
        torchvision \
        git+https://github.com/huggingface/transformers \
        accelerate \
        qwen-vl-utils[decord]==0.0.8 \
        fastapi \
        uvicorn[standard] \
        gunicorn

# Create a non-root user and set working directory
RUN useradd -m -u 1000 user
WORKDIR /app

# Copy the application files to the container
COPY --chown=user . /app

# Switch to non-root user
USER user

# Set environment variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Start the application using Gunicorn with Uvicorn workers
#CMD ["gunicorn", "main:app", "--worker-class", "uvicorn.workers.UvicornWorker", "--workers", "1", "--bind", "0.0.0.0:7860"]
CMD ["sh", "-c", "gunicorn main:app --worker-class uvicorn.workers.UvicornWorker --workers ${WORKERS:-4} --bind 0.0.0.0:7860"]


