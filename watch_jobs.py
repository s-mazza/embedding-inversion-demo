#!/usr/bin/env python3
"""
Polls SLURM on faretra and sends a Telegram notification when a job starts (PD -> R).

Usage:
    python3 watch_jobs.py

Requirements:
    pip install paramiko requests
"""

import time
import subprocess
import requests

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CLUSTER_USER = "mazzacano"
CLUSTER_HOST = "137.204.107.40"
CLUSTER_PORT = 37335

TELEGRAM_TOKEN = "<YOUR_BOT_TOKEN>"
TELEGRAM_CHAT_ID = "<YOUR_CHAT_ID>"

POLL_INTERVAL = 30  # seconds

# ─────────────────────────────────────────────
# TELEGRAM
# ─────────────────────────────────────────────

def send_telegram(message: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message})


# ─────────────────────────────────────────────
# SLURM
# ─────────────────────────────────────────────

def get_jobs() -> dict:
    """SSH into faretra and return {job_id: status} for current user's jobs."""
    result = subprocess.run(
        [
            "ssh", "-p", str(CLUSTER_PORT),
            f"{CLUSTER_USER}@{CLUSTER_HOST}",
            f"squeue -u {CLUSTER_USER} --format='%i %T %j' --noheader"
        ],
        capture_output=True, text=True
    )
    jobs = {}
    for line in result.stdout.strip().splitlines():
        parts = line.strip().split()
        if len(parts) >= 3:
            job_id, status, name = parts[0], parts[1], parts[2]
            jobs[job_id] = {"status": status, "name": name}
    return jobs


# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────

def main():
    print(f"Watching SLURM jobs for {CLUSTER_USER}@{CLUSTER_HOST}...")
    send_telegram("👀 Job watcher started.")

    previous = get_jobs()
    print(f"Current jobs: {previous}")

    while True:
        time.sleep(POLL_INTERVAL)
        current = get_jobs()

        for job_id, info in current.items():
            prev_status = previous.get(job_id, {}).get("status")
            curr_status = info["status"]
            name = info["name"]

            if prev_status == "PD" and curr_status == "R":
                msg = f"🚀 Job started!\nID: {job_id}\nName: {name}"
                print(msg)
                send_telegram(msg)

            if job_id not in previous and curr_status == "R":
                msg = f"🚀 Job started!\nID: {job_id}\nName: {name}"
                print(msg)
                send_telegram(msg)

        for job_id, info in previous.items():
            if job_id not in current:
                msg = f"✅ Job completed (or cancelled)\nID: {job_id}\nName: {info['name']}"
                print(msg)
                send_telegram(msg)

        previous = current


if __name__ == "__main__":
    main()
