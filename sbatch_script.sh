#!/bin/bash

# 2 GPU (DDP) - pin to faretra where dataset lives, resume from checkpoint
sbatch -N 1 --gpus=nvidia_geforce_rtx_3090:2 -w faretra run_docker.sh --resume

# 1 GPU fallback (if only 1 GPU available on faretra)
# sbatch -N 1 --gpus=nvidia_geforce_rtx_3090:1 -w faretra run_docker.sh --resume
