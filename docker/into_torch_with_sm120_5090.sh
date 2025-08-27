#!/usr/bin/bash

docker run -it \
  --gpus all \
  --ipc=host \
  -v "$(pwd)":/app \
  --entrypoint bash \
  vllm-sm120:latest
