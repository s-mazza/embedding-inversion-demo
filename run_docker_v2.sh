#!/bin/bash
PHYS_DIR="/home/mazzacano/embedding-inversion-demo"
LLM_CACHE_DIR="/llms"
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

docker run \
    -v "$PHYS_DIR":/workspace \
    -v "$LLM_CACHE_DIR":/llms \
    -e HF_HOME="/llms" \
    -e NUM_GPUS=$NUM_GPUS \
    -e CONFIG=${CONFIG:-configs/v2_jinav3.yaml} \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --rm \
    --init \
    --memory="60g" \
    --ipc=host \
    --gpus '"device='"$CUDA_VISIBLE_DEVICES"'"' \
    emb-inversion \
    "/workspace/train.sh" \
    "$@"
