#!/bin/bash
docker run \
    -v /home/mazzacano/embedding-inversion-demo:/workspace \
    -v /llms:/llms \
    -e HF_HOME=/llms \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --rm --memory=30g \
    --gpus '"device='"$CUDA_VISIBLE_DEVICES"'"' \
    emb-inversion /workspace/prepare_data_stage2_v2.sh
