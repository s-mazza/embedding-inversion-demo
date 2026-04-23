#!/bin/bash
NUM_GPUS=${NUM_GPUS:-1}
torchrun --nproc_per_node=$NUM_GPUS \
    /workspace/train.py --config /workspace/configs/v3_mmbert_jinav3.yaml "$@"
