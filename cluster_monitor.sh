#!/bin/bash
# Monitors SLURM training jobs and sends Telegram notifications.
# Run as: nohup bash cluster_monitor.sh > monitor.log 2>&1 &

source ~/.telegram_credentials || { echo "Missing ~/.telegram_credentials"; exit 1; }
LOG_DIR="$HOME/embedding-inversion-demo"
STATE_DIR="/tmp/cluster_monitor_$$"
mkdir -p "$STATE_DIR"

# job_id:num_gpus:log_suffix
JOBS=(
    "11063544:2:slurm-11063544.out"
    "11078393:1:slurm-11078393-1gpu.out"
)

MILESTONE_INTERVAL=10000   # Telegram update every N steps
REPORT_VAL_EVERY=10        # also report every Nth val_loss line (fallback)

send() {
    curl -s -X POST "https://api.telegram.org/bot${BOT_TOKEN}/sendMessage" \
        --data-urlencode "chat_id=${CHAT_ID}" \
        --data-urlencode "text=$1" \
        --max-time 15 > /dev/null 2>&1
}

monitor_log() {
    local log_file="$1"
    local label="$2"
    local last_step_file="$STATE_DIR/step_${label//[^a-z0-9]/_}"
    echo "0" > "$last_step_file"

    # Wait up to 10 min for log file to appear
    local waited=0
    while [ ! -f "$log_file" ] && [ $waited -lt 120 ]; do
        sleep 5
        waited=$((waited + 1))
    done
    [ ! -f "$log_file" ] && send "⚠️ [$label] log not found after 10 min" && return

    local val_counter=0
    tail -n +1 -f "$log_file" 2>/dev/null | while IFS= read -r line; do

        # --- errors (always notify) ---
        if echo "$line" | grep -qE "Traceback|CUDA out of memory|RuntimeError|FAILED|Killed|Error:"; then
            send "🚨 [$label] $line"
            continue
        fi

        # --- new best checkpoint ---
        if echo "$line" | grep -q "Saved best"; then
            send "⭐ [$label] $line"
            continue
        fi

        # --- val_loss lines: notify every REPORT_VAL_EVERY or at step milestones ---
        if echo "$line" | grep -q "val_loss"; then
            val_counter=$((val_counter + 1))
            step=$(echo "$line" | grep -oP 'step \K[0-9]+' | head -1)
            last_step=$(cat "$last_step_file" 2>/dev/null || echo 0)

            should_send=0
            # milestone-based
            if [ -n "$step" ] && [ "$step" -ge $((last_step + MILESTONE_INTERVAL)) ] 2>/dev/null; then
                echo "$step" > "$last_step_file"
                should_send=1
            fi
            # count-based fallback (when step not parseable)
            if [ $((val_counter % REPORT_VAL_EVERY)) -eq 0 ]; then
                should_send=1
            fi
            [ "$should_send" -eq 1 ] && send "📊 [$label] $line"
        fi

    done
}

# Initialise state files
for entry in "${JOBS[@]}"; do
    job_id="${entry%%:*}"
    echo "PENDING" > "$STATE_DIR/state_$job_id"
done

send "🤖 Monitor started.
• Job 11063544 (2-GPU, 96h)
• Job 11078393 (1-GPU backup, 7d)
Updates: job state changes, new best model, errors, every ${MILESTONE_INTERVAL} steps."

# Main loop: poll job states, start log monitor when job transitions to RUNNING
while true; do
    for entry in "${JOBS[@]}"; do
        IFS=':' read -r job_id gpus log_name <<< "$entry"
        label="${gpus}-GPU #${job_id}"
        state_file="$STATE_DIR/state_$job_id"
        log_started_file="$STATE_DIR/log_started_$job_id"

        new_state=$(squeue -j "$job_id" --format="%T" --noheader 2>/dev/null | tr -d ' ')
        old_state=$(cat "$state_file" 2>/dev/null)

        if [ -z "$new_state" ] && [ "$old_state" != "DONE" ]; then
            echo "DONE" > "$state_file"
            final=$(sacct -j "$job_id" --format=State --noheader 2>/dev/null | head -1 | tr -d ' ')
            send "🏁 [$label] finished — $final"

        elif [ -n "$new_state" ] && [ "$new_state" != "$old_state" ]; then
            echo "$new_state" > "$state_file"
            send "🔄 [$label] $old_state → $new_state"

            # Start log monitor only when job becomes RUNNING
            if [ "$new_state" = "RUNNING" ] && [ ! -f "$log_started_file" ]; then
                touch "$log_started_file"
                monitor_log "$LOG_DIR/$log_name" "$label" &
            fi
        fi
    done
    sleep 60
done
