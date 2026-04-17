#!/bin/bash

# Stage 2: tokenize + encode with jina-embeddings-v3 (GPU needed)
python3 /workspace/prepare_data_fast.py \
    --stage 2 \
    --config /workspace/configs/v3_mmbert_jinav3.yaml \
    --encode-batch 512
