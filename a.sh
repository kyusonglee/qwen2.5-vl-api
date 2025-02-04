docker run  \
  --gpus all \
  -p 7860:7860 \
  -e CHECKPOINT=/app/checkpoints/Qwen2.5-VL-3B-Instruct \
  -v /home/kai/Documents/resources/:/app/checkpoints \
  kyusonglee/qwen2vl 
