#!/bin/bash
# Cancels ONE_GPU_JOB as soon as TWO_GPU_JOB transitions to RUNNING.
# Usage: watch_and_cancel.sh <two_gpu_job_id> <one_gpu_job_id>
TWO_GPU=$1
ONE_GPU=$2

echo "$(date): watching job $TWO_GPU — will cancel $ONE_GPU when it starts"

while true; do
    state=$(squeue -j "$TWO_GPU" --format="%T" --noheader 2>/dev/null | tr -d ' ')
    if [ "$state" = "RUNNING" ]; then
        echo "$(date): job $TWO_GPU is RUNNING — cancelling 1-GPU job $ONE_GPU"
        scancel "$ONE_GPU"
        echo "$(date): scancel sent for $ONE_GPU"
        exit 0
    elif [ -z "$state" ]; then
        echo "$(date): job $TWO_GPU no longer in queue — exiting"
        exit 0
    fi
    sleep 60
done
