#!/bin/bash
NUM_GPUS=${NUM_GPUS:-1}
CONFIG=${CONFIG:-configs/v2_jinav3.yaml}
CKPT_PATH="/workspace/checkpoints_v2_jinav3/latest.pt"

# Pre-flight: run health check tests 1-6 (CPU-only, ~15s) before wasting GPU-hours.
# Skipped on fresh start (no checkpoint). Abort training if any test FAILs.
echo "=== Pre-flight checks ==="
if [ -f "$CKPT_PATH" ]; then
    python3 /workspace/training_health_check.py \
        --checkpoint "$CKPT_PATH" --cpu --tests 1,2,3,4,5,6 || {
        echo "PRE-FLIGHT FAILED: health check found critical issues — aborting training"
        exit 1
    }
    echo "=== Pre-flight OK ==="
else
    echo "=== No checkpoint found — skipping pre-flight (fresh start) ==="
fi

torchrun --nproc_per_node=$NUM_GPUS \
    /workspace/train.py --config /workspace/$CONFIG "$@"
