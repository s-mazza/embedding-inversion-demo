#!/usr/bin/env python3
"""
Live training dashboard for CMDLM v2.

Parses SLURM log (local file or via SSH from faretra) and renders:
- Progress bar, elapsed/ETA
- Current val_loss + token_acc vs paper targets
- ASCII loss curve with paper reference line
- Recent anomalies

Usage:
    # From local file:
    python3 training_dashboard.py --log slurm-11108132.out

    # From cluster (auto-SSH):
    python3 training_dashboard.py --job 11108132

    # Auto-discover current job from squeue:
    python3 training_dashboard.py
"""

import argparse
import math
import os
import re
import subprocess
import sys
import time
from datetime import timedelta
from typing import Optional

PAPER_TARGET_STEP = 62_500
PAPER_VAL_LOSS    = 1.60
PAPER_TOKEN_ACC   = 0.76
MAX_STEPS         = 200_000
SSH_HOST          = "faretra"
LOG_DIR_REMOTE    = "~/embedding-inversion-demo"


# ── Log parsing ──────────────────────────────────────────────────────────────

_STEP_RE   = re.compile(r'^step (\d+)/(\d+) \| loss ([\d.]+) \| acc ([\d.]+) \| lr ([\S]+)')
_ELAPSED_RE = re.compile(r'elapsed ([\d.]+)min')
_VAL_RE    = re.compile(r'val_loss \(ema\): ([\d.]+)')
_ACC_RE    = re.compile(r'token_acc \(EMA, 100% mask\): ([\d.]+)')
_ERR_RE    = re.compile(r'Traceback|CUDA out of memory|RuntimeError|FAILED|Killed|Error:')
_BEST_RE   = re.compile(r'Saved best')


def parse_log(text: str) -> dict:
    steps = []
    current = {}
    anomalies = []
    pending_val = None

    for line in text.splitlines():
        m = _STEP_RE.match(line)
        if m:
            current = {
                "step": int(m.group(1)),
                "max_steps": int(m.group(2)),
                "loss": float(m.group(3)),
                "acc": float(m.group(4)),
                "lr": m.group(5),
            }
            em = _ELAPSED_RE.search(line)
            if em:
                current["elapsed_min"] = float(em.group(1))
            pending_val = None
            steps.append(current)
            continue

        m = _VAL_RE.search(line)
        if m and current:
            pending_val = float(m.group(1))
            current = dict(current, val_loss=pending_val)
            steps[-1] = current
            continue

        m = _ACC_RE.search(line)
        if m and pending_val is not None and current:
            current = dict(current, token_acc=float(m.group(1)))
            steps[-1] = current
            pending_val = None
            continue

        if _ERR_RE.search(line):
            anomalies.append(("ERROR", line.strip()[:120]))
        elif _BEST_RE.search(line):
            anomalies.append(("BEST", line.strip()[:120]))

    val_steps = [s for s in steps if "val_loss" in s and "token_acc" in s]

    return {
        "all_steps": steps,
        "val_steps": val_steps,
        "anomalies": anomalies,
        "latest": steps[-1] if steps else None,
        "latest_val": val_steps[-1] if val_steps else None,
        "best_val": min(val_steps, key=lambda s: s["val_loss"]) if val_steps else None,
    }


# ── Remote log fetch ──────────────────────────────────────────────────────────

def fetch_log_ssh(job_id: str, is_1gpu: bool = False) -> Optional[str]:
    suffix = "-1gpu" if is_1gpu else ""
    remote_path = f"{LOG_DIR_REMOTE}/slurm-{job_id}{suffix}.out"
    try:
        result = subprocess.run(
            ["ssh", SSH_HOST, f"cat {remote_path}"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0 and result.stdout:
            return result.stdout
    except Exception:
        pass
    return None


def discover_job_ids() -> list[str]:
    try:
        result = subprocess.run(
            ["ssh", SSH_HOST, "squeue -u mazzacano --format='%.10i %.8T' --noheader"],
            capture_output=True, text=True, timeout=15
        )
        ids = re.findall(r'(\d+)\s+RUNNING', result.stdout)
        if not ids:
            ids = re.findall(r'(\d+)\s+PENDING', result.stdout)
        return ids
    except Exception:
        return []


# ── Rendering ─────────────────────────────────────────────────────────────────

W = 64  # display width

def render_bar(step: int, max_steps: int, width: int = 30) -> str:
    frac = step / max(max_steps, 1)
    filled = int(frac * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {frac*100:.1f}%"


def render_ascii_curve(val_steps: list[dict], width: int = 50, height: int = 8) -> str:
    if len(val_steps) < 2:
        return "  (not enough val points yet)"

    losses = [s["val_loss"] for s in val_steps]
    steps  = [s["step"] for s in val_steps]

    y_max = max(losses) * 1.05
    y_min = min(PAPER_VAL_LOSS * 0.9, min(losses) * 0.95)

    def y_to_row(y):
        return int((y_max - y) / (y_max - y_min) * (height - 1))

    def x_to_col(x):
        return int((x - steps[0]) / max(steps[-1] - steps[0], 1) * (width - 1))

    grid = [[" "] * width for _ in range(height)]

    # Paper target line
    paper_row = y_to_row(PAPER_VAL_LOSS)
    if 0 <= paper_row < height:
        for c in range(width):
            grid[paper_row][c] = "·"

    # Actual loss curve
    for s in val_steps:
        r = y_to_row(s["val_loss"])
        c = x_to_col(s["step"])
        if 0 <= r < height and 0 <= c < width:
            grid[r][c] = "█"

    lines = []
    for i, row in enumerate(grid):
        y_val = y_max - (y_max - y_min) * i / (height - 1)
        marker = " "
        row_str = "".join(row)
        if "·" in row_str and "█" not in row_str:
            marker = "←paper"
            row_str = row_str.replace("·", "─")
        lines.append(f"  {y_val:4.2f} │{''.join(row_str)}{' ' + marker if marker != ' ' else ''}")

    lines.append(f"       └{'─'*width}")
    lines.append(f"       {steps[0]:>7,}{' '*int(width/2 - 6)}step{' '*4}{steps[-1]:>7,}")
    return "\n".join(lines)


def render_gap(current: float, target: float, higher_is_better: bool = False) -> str:
    gap = current - target
    if higher_is_better:
        icon = "▲" if gap >= -0.02 else "▼"
    else:
        icon = "▲" if gap <= 0.05 else "▼"
    return f"{icon} ({gap:+.3f} vs paper)"


def render_eta(step: int, max_steps: int, elapsed_min: float) -> str:
    if step == 0 or elapsed_min == 0:
        return "unknown"
    rate = step / elapsed_min
    remaining = (max_steps - step) / max(rate, 1e-9)
    return str(timedelta(minutes=int(remaining))).split(".")[0]


def render_dashboard(data: dict, job_id: str = "?", gpus: str = "?") -> str:
    sep = "═" * W
    thin = "─" * W
    lines = [f"\n{sep}",
             f"  Training Dashboard  |  job {job_id}  |  {gpus} GPU(s)",
             thin]

    latest = data["latest"]
    latest_val = data["latest_val"]

    if not latest:
        lines.append("  No training steps found in log yet.")
        lines.append(sep)
        return "\n".join(lines)

    step = latest["step"]
    max_steps = latest.get("max_steps", MAX_STEPS)
    elapsed_min = latest.get("elapsed_min", 0)

    lines.append(f"  Progress: {render_bar(step, max_steps)}  step {step:,}/{max_steps:,}")
    lines.append(f"  Elapsed:  {timedelta(minutes=int(elapsed_min))}  |  "
                 f"ETA: {render_eta(step, max_steps, elapsed_min)}")
    lines.append(f"  LR:       {latest.get('lr', '?')}  |  "
                 f"train_loss: {latest.get('loss', 0):.4f}")
    lines.append(thin)

    if latest_val:
        vl = latest_val["val_loss"]
        acc = latest_val["token_acc"]
        lines.append(f"  val_loss  (EMA): {vl:.4f}  [target @62500: {PAPER_VAL_LOSS}]  "
                     f"{render_gap(vl, PAPER_VAL_LOSS)}")
        lines.append(f"  token_acc (EMA): {acc:.3f}   [target @62500: {PAPER_TOKEN_ACC}]  "
                     f"{render_gap(acc, PAPER_TOKEN_ACC, higher_is_better=True)}")

        best = data["best_val"]
        if best and best["step"] != latest_val["step"]:
            lines.append(f"  Best so far: val_loss={best['val_loss']:.4f}  "
                         f"token_acc={best['token_acc']:.3f}  @ step {best['step']:,}")
    else:
        lines.append("  No val evaluations yet (first eval at step 500)")

    lines.append(thin)

    # Curve
    val_steps = data["val_steps"]
    if len(val_steps) >= 2:
        lines.append(f"  val_loss curve (·=paper target {PAPER_VAL_LOSS}):")
        lines.append(render_ascii_curve(val_steps[-50:]))  # last 50 evals
    else:
        lines.append("  (waiting for first val evals…)")

    # Anomalies
    anomalies = data["anomalies"]
    if anomalies:
        lines.append(thin)
        lines.append(f"  Recent events ({len(anomalies)}):")
        for kind, msg in anomalies[-5:]:
            icon = "🚨" if kind == "ERROR" else "⭐"
            lines.append(f"    {icon} {msg[:W-6]}")
    else:
        lines.append(f"\n  Recent anomalies: none")

    lines.append(sep)
    lines.append(f"  Refresh: python3 training_dashboard.py --job {job_id}")
    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Live training dashboard for CMDLM v2")
    parser.add_argument("--job", help="SLURM job ID (2-GPU job)")
    parser.add_argument("--log", help="Local log file path")
    parser.add_argument("--1gpu", action="store_true", dest="one_gpu",
                        help="Use 1-GPU log suffix (-1gpu.out)")
    args = parser.parse_args()

    log_text = None

    if args.log:
        with open(args.log, 'r', errors='replace') as f:
            log_text = f.read()
        job_id = os.path.basename(args.log).replace("slurm-", "").replace(".out", "").replace("-1gpu", "")
        gpus = "1" if args.one_gpu else "2"

    elif args.job:
        job_id = args.job
        gpus = "1" if args.one_gpu else "2"
        print(f"Fetching log for job {job_id} from {SSH_HOST}…", end=" ", flush=True)
        log_text = fetch_log_ssh(job_id, is_1gpu=args.one_gpu)
        if log_text:
            print("OK")
        else:
            print("FAILED")
            print(f"Could not fetch log. Is the job running? Try: ssh {SSH_HOST} 'ls {LOG_DIR_REMOTE}/slurm-{job_id}*.out'")
            sys.exit(1)

    else:
        # Auto-discover
        print(f"Discovering jobs on {SSH_HOST}…", end=" ", flush=True)
        job_ids = discover_job_ids()
        if not job_ids:
            print("none found")
            print(f"No RUNNING/PENDING jobs. Provide --job JOB_ID or --log PATH.")
            sys.exit(1)
        job_id = job_ids[0]
        gpus = "2"
        print(f"found job {job_id}")
        print(f"Fetching log…", end=" ", flush=True)
        log_text = fetch_log_ssh(job_id)
        if log_text:
            print("OK")
        else:
            print("log not available yet (job may still be PENDING)")
            sys.exit(0)

    data = parse_log(log_text)
    print(render_dashboard(data, job_id=job_id, gpus=gpus))


if __name__ == "__main__":
    main()
