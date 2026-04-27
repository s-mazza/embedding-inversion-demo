#!/bin/bash
# Stage 2 for v2_jinav3: tokenize with xlm-roberta + encode with jina-v3
python3 /workspace/prepare_data_fast.py \
    --stage 2 \
    --config /workspace/configs/v2_jinav3.yaml \
    --encode-batch 512
