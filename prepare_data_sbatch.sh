#!/bin/bash

docker run \
    -v /home/mazzacano/embedding-inversion-demo:/workspace \
    -v /llms:/llms \
    -e HF_HOME=/llms \
    --rm --memory=30g \
    --gpus '"device='"$CUDA_VISIBLE_DEVICES"'"' \
    emb-inversion /workspace/prepare_data.sh
