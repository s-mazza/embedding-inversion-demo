#!/bin/bash

# Stage 1: download 2M English samples from C4 (no GPU needed)
python3 /workspace/prepare_data_fast.py \
    --stage 1 \
    --langs en \
    --n-samples 2000000
