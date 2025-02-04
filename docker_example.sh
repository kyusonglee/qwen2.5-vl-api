#!/bin/bash

# Build the Docker image
docker build -t qwen2.5-vl-api .

# Download the model checkpoint at /resources/Qwen2.5-VL-3B-Instruct from https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct

# Run the Docker container
docker run  \
  --gpus all \
  -p 7860:7860 \
  -e CHECKPOINT=/app/checkpoints/Qwen2.5-VL-3B-Instruct \
  -v /resources/:/app/checkpoints \
  qwen2.5-vl-api 
