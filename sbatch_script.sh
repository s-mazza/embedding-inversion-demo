#!/bin/bash

sbatch -N 1 --gpus=nvidia_geforce_rtx_3090:1 run_docker.sh

# To resume from checkpoint:
# sbatch -N 1 --gpus=nvidia_geforce_rtx_3090:1 run_docker.sh --resume

# If your dataset is too large to replicate on all nodes, pin to faretra:
# sbatch -N 1 --gpus=nvidia_geforce_rtx_3090:1 -w faretra run_docker.sh
