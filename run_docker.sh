#!/bin/bash
PHYS_DIR="/home/mazzacano/embedding-inversion-demo"
LLM_CACHE_DIR="/llms"

docker run \
    -v "$PHYS_DIR":/workspace \
    -v "$LLM_CACHE_DIR":/llms \
    -e HF_HOME="/llms" \
    --rm \
    --memory="30g" \
    --gpus '"device='"$CUDA_VISIBLE_DEVICES"'"' \
    emb-inversion \
    "/workspace/train.sh" \
    "$@"
