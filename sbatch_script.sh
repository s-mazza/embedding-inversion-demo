#!/bin/bash
# Launch fresh v2_jinav3 training (8-layer from-scratch, jina-v3 encoder).
#
# Strategy:
#   1-GPU job starts immediately (v2_jinav3_1gpu config, grad_accum=8)
#   2-GPU job queues until 2x RTX 3090 are free (v2_jinav3 config, grad_accum=4)
#   watch_and_cancel kills the 1-GPU job the moment 2-GPU transitions to RUNNING
#   cluster_monitor sends Telegram updates for state changes, val_loss, errors
#
# IMPORTANT — fresh run: delete old checkpoints first:
#   rm -rf /home/mazzacano/embedding-inversion-demo/checkpoints_v2_jinav3/
#
# Run this from faretra (the cluster).

set -e

cd /home/mazzacano/embedding-inversion-demo

# ── 1-GPU job: starts immediately ──────────────────────────────────────────
ONE_GPU_JOB=$(sbatch --parsable \
    --output="slurm-%j-1gpu.out" \
    -N 1 --gpus=nvidia_geforce_rtx_3090:1 -w faretra \
    --export=ALL,CONFIG=configs/v2_jinav3_1gpu.yaml \
    run_docker_v2.sh --resume)
echo "1-GPU job: $ONE_GPU_JOB  (log: slurm-${ONE_GPU_JOB}-1gpu.out)"

# ── 2-GPU job: queues until 2x RTX 3090 available ──────────────────────────
TWO_GPU_JOB=$(sbatch --parsable \
    -N 1 --gpus=nvidia_geforce_rtx_3090:2 -w faretra \
    --export=ALL,CONFIG=configs/v2_jinav3.yaml \
    run_docker_v2.sh --resume)
echo "2-GPU job: $TWO_GPU_JOB  (log: slurm-${TWO_GPU_JOB}.out)"

# ── Watcher: cancels 1-GPU when 2-GPU starts running ───────────────────────
nohup bash watch_and_cancel.sh "$TWO_GPU_JOB" "$ONE_GPU_JOB" \
    > "watch_${TWO_GPU_JOB}_${ONE_GPU_JOB}.log" 2>&1 &
echo "Watcher PID $!  (log: watch_${TWO_GPU_JOB}_${ONE_GPU_JOB}.log)"

echo ""
echo "Queue:  squeue -u \$USER"
echo "Tail:   tail -f slurm-${TWO_GPU_JOB}.out"
echo "Monitor: nohup bash cluster_monitor.sh $TWO_GPU_JOB $ONE_GPU_JOB > monitor.log 2>&1 &"
