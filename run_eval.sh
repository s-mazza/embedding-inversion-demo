#!/bin/bash
PHYS_DIR="/home/mazzacano/embedding-inversion-demo"
LLM_CACHE_DIR="/llms"

docker run \
    -v "$PHYS_DIR":/workspace \
    -v "$LLM_CACHE_DIR":/llms \
    -e HF_HOME="/llms" \
    --rm \
    --gpus all \
    emb-inversion \
    python3 /workspace/${EVAL_SCRIPT:-eval_quality.py} \
        --checkpoint /workspace/checkpoints_v3_mmbert_jinav3/best.pt \
        "$@"
