#!/bin/bash
NUM_GPUS=${NUM_GPUS:-1}
CONFIG=${CONFIG:-configs/v3_mmbert_jinav3.yaml}
torchrun --nproc_per_node=$NUM_GPUS \
    /workspace/train.py --config /workspace/$CONFIG "$@"
