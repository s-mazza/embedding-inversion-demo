#!/bin/bash

# Stage 1: download only, no GPU needed — run directly on faretra:
# docker run -v /home/mazzacano/embedding-inversion-demo:/workspace -v /llms:/llms -e HF_HOME=/llms --rm --memory=30g emb-inversion /workspace/prepare_data_stage1.sh

# Stage 2: encode with GPU
docker run \
    -v /home/mazzacano/embedding-inversion-demo:/workspace \
    -v /llms:/llms \
    -e HF_HOME=/llms \
    --rm --memory=30g \
    --gpus '"device='"$CUDA_VISIBLE_DEVICES"'"' \
    emb-inversion /workspace/prepare_data_stage2.sh
