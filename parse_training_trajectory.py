#!/usr/bin/env python3
"""
Trajectory early-warning system for CMDLM v2 training.

Parses SLURM log files, fits a decay curve to val_loss and token_acc,
and alerts via Telegram if the projected trajectory is off-track vs.
paper Table 1 (jina-v3: val_loss=1.60, token_acc=0.76 @ step 62,500).

Usage:
    python3 parse_training_trajectory.py slurm-11108132.out
    python3 parse_training_trajectory.py slurm-11108132.out --telegram
    python3 parse_training_trajectory.py slurm-11108132.out --json
"""

import argparse
import math
import os
import re
import subprocess
import sys
from typing import Optional

# Paper reference (Table 1, jina-v3 8-layer)
REFERENCE = {
     5_000: {"val_loss": 3.80, "token_acc": 0.20},
    10_000: {"val_loss": 2.80, "token_acc": 0.40},
    20_000: {"val_loss": 2.20, "token_acc": 0.55},
    40_000: {"val_loss": 1.85, "token_acc": 0.65},
    62_500: {"val_loss": 1.60, "token_acc": 0.76},
}
TARGET_STEP = 62_500
ALERT_VAL_LOSS_THRESHOLD = 1.80   # projected val_loss @62500 above this → alert
ALERT_TOKEN_ACC_THRESHOLD = 0.70  # projected token_acc @62500 below this → alert
MIN_POINTS_FOR_FIT = 5            # need at least this many val points to project


def parse_log(path: str) -> list[dict]:
    """Return list of {step, loss, val_loss, token_acc} from SLURM log."""
    step_re    = re.compile(r'^step (\d+)/\d+ \| loss ([\d.]+) \| acc ([\d.]+)')
    val_re     = re.compile(r'val_loss \(ema\): ([\d.]+)')
    acc_re     = re.compile(r'token_acc \(EMA, 100% mask\): ([\d.]+)')

    records = []
    current_step = None
    current_loss = None
    pending_val = None

    with open(path, 'r', errors='replace') as f:
        for line in f:
            m = step_re.match(line)
            if m:
                current_step = int(m.group(1))
                current_loss = float(m.group(2))
                pending_val = None
                continue

            m = val_re.search(line)
            if m and current_step is not None:
                pending_val = float(m.group(1))
                continue

            m = acc_re.search(line)
            if m and pending_val is not None and current_step is not None:
                records.append({
                    "step": current_step,
                    "loss": current_loss,
                    "val_loss": pending_val,
                    "token_acc": float(m.group(1)),
                })
                pending_val = None

    return records


def _fit_decay(xs: list[float], ys: list[float]) -> Optional[tuple]:
    """Fit y = a * exp(-b * x) + c. Returns (a, b, c) or None on failure."""
    try:
        from scipy.optimize import curve_fit
        import numpy as np

        x = np.array(xs, dtype=float)
        y = np.array(ys, dtype=float)

        def model(x, a, b, c):
            return a * np.exp(-b * x) + c

        y_range = y.max() - y.min()
        p0 = [y_range, 1e-5, y.min()]
        bounds = ([0, 0, 0], [100, 1, 20])
        popt, _ = curve_fit(model, x, y, p0=p0, bounds=bounds, maxfev=5000)
        return tuple(popt)
    except Exception:
        return None


def _project_linear(xs: list[float], ys: list[float], target_x: float) -> Optional[float]:
    """Linear extrapolation on log-transformed y as fallback."""
    try:
        import numpy as np
        x = np.array(xs[-10:])
        y = np.log(np.maximum(ys[-10:], 1e-9))
        coeffs = np.polyfit(x, y, 1)
        return float(math.exp(coeffs[0] * target_x + coeffs[1]))
    except Exception:
        return None


def project_at_step(records: list[dict], target_step: int, field: str) -> Optional[float]:
    """Project field value at target_step using curve fitting or linear fallback.

    token_acc is increasing, so we fit its complement (1 - acc) as a decay
    curve and invert back. val_loss is already decreasing so fits directly.
    """
    xs = [r["step"] for r in records if r[field] is not None]
    ys_raw = [r[field] for r in records if r[field] is not None]

    if len(xs) < MIN_POINTS_FOR_FIT:
        return None

    is_acc = (field == "token_acc")
    ys = [1.0 - y for y in ys_raw] if is_acc else ys_raw

    # Refuse to project a near-constant series — the decay fit will collapse to
    # the constant and produce meaningless projections (e.g. token_acc=0 across
    # all early val points → "projected acc = 0" forever).
    y_min, y_max = min(ys), max(ys)
    if y_max - y_min < 1e-4:
        return None

    params = _fit_decay(xs, ys)
    if params is not None:
        a, b, c = params
        projected = a * math.exp(-b * target_step) + c
    else:
        projected = _project_linear(xs, ys, target_step)

    if projected is None:
        return None
    return max(0.0, min(1.0, 1.0 - projected)) if is_acc else max(0.0, projected)


def find_reference_gap(records: list[dict]) -> list[str]:
    """Compare current values against reference checkpoints."""
    gaps = []
    current_step = records[-1]["step"] if records else 0

    for ref_step, refs in sorted(REFERENCE.items()):
        if ref_step > current_step:
            break
        # Find closest actual record to this reference step
        closest = min(records, key=lambda r: abs(r["step"] - ref_step))
        if abs(closest["step"] - ref_step) > 2000:
            continue

        vl_actual = closest["val_loss"]
        acc_actual = closest["token_acc"]
        vl_ref = refs["val_loss"]
        acc_ref = refs["token_acc"]

        vl_status = "OK" if vl_actual <= vl_ref * 1.15 else "BEHIND"
        acc_status = "OK" if acc_actual >= acc_ref * 0.85 else "BEHIND"

        gaps.append(
            f"  step {ref_step:>6,}: val_loss={vl_actual:.3f} (ref≤{vl_ref}) [{vl_status}]  "
            f"token_acc={acc_actual:.3f} (ref≥{acc_ref}) [{acc_status}]"
        )

    return gaps


def send_telegram(msg: str):
    """Send a message via Telegram using env vars BOT_TOKEN and CHAT_ID."""
    token = os.environ.get("BOT_TOKEN") or os.environ.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get("CHAT_ID") or os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("[telegram] BOT_TOKEN or CHAT_ID not set — skipping", file=sys.stderr)
        return
    try:
        import urllib.request, json
        payload = json.dumps({"chat_id": chat_id, "text": msg}).encode()
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=15)
    except Exception as e:
        print(f"[telegram] error: {e}", file=sys.stderr)


def check_trajectory(log_path: str, use_telegram: bool = False, as_json: bool = False) -> dict:
    """Main check: parse log, project trajectory, return status dict."""
    records = parse_log(log_path)

    if not records:
        result = {"status": "NO_DATA", "message": "No val records found in log yet"}
        if as_json:
            import json; print(json.dumps(result))
        else:
            print(result["message"])
        return result

    current_step = records[-1]["step"]
    current_val_loss = records[-1]["val_loss"]
    current_token_acc = records[-1]["token_acc"]

    proj_val_loss = project_at_step(records, TARGET_STEP, "val_loss")
    proj_token_acc = project_at_step(records, TARGET_STEP, "token_acc")
    ref_gaps = find_reference_gap(records)

    alerts = []
    if proj_val_loss is not None and proj_val_loss > ALERT_VAL_LOSS_THRESHOLD:
        alerts.append(
            f"⚠️ TRAJECTORY WARNING: projected val_loss @{TARGET_STEP:,} = {proj_val_loss:.3f} "
            f"(target ≤ {ALERT_VAL_LOSS_THRESHOLD}, paper = 1.60)"
        )
    if proj_token_acc is not None and proj_token_acc < ALERT_TOKEN_ACC_THRESHOLD:
        alerts.append(
            f"⚠️ TRAJECTORY WARNING: projected token_acc @{TARGET_STEP:,} = {proj_token_acc:.3f} "
            f"(target ≥ {ALERT_TOKEN_ACC_THRESHOLD}, paper = 0.76)"
        )

    result = {
        "status": "ALERT" if alerts else "OK",
        "current_step": current_step,
        "current_val_loss": current_val_loss,
        "current_token_acc": current_token_acc,
        "proj_val_loss_at_62500": proj_val_loss,
        "proj_token_acc_at_62500": proj_token_acc,
        "num_val_points": len(records),
        "alerts": alerts,
        "ref_gaps": ref_gaps,
    }

    if as_json:
        import json
        print(json.dumps(result, indent=2))
        return result

    # Human-readable output
    print(f"\n{'─'*60}")
    print(f"Trajectory Check  |  step {current_step:,}  |  {len(records)} val points")
    print(f"{'─'*60}")
    print(f"  Current:   val_loss={current_val_loss:.4f}  token_acc={current_token_acc:.3f}")
    if proj_val_loss is not None:
        gap_vl = proj_val_loss - 1.60
        vl_icon = "✓" if proj_val_loss <= ALERT_VAL_LOSS_THRESHOLD else "✗"
        print(f"  Projected: val_loss={proj_val_loss:.3f} (Δ{gap_vl:+.3f} vs paper) [{vl_icon}]")
    else:
        print(f"  Projected: val_loss not yet projectable (need {MIN_POINTS_FOR_FIT}+ val points)")

    if proj_token_acc is not None:
        gap_acc = proj_token_acc - 0.76
        acc_icon = "✓" if proj_token_acc >= ALERT_TOKEN_ACC_THRESHOLD else "✗"
        print(f"             token_acc={proj_token_acc:.3f} (Δ{gap_acc:+.3f} vs paper) [{acc_icon}]")
    else:
        # Common when EMA token_acc is still 0.000 across all val points (early
        # warmup, EMA half-life ~6932 steps): the decay fit can't latch onto a
        # constant series. Skip the line; will become projectable once acc moves.
        print(f"             token_acc: not yet projectable (constant or insufficient signal)")

    if ref_gaps:
        print(f"\n  Reference comparison:")
        for g in ref_gaps:
            print(g)

    if alerts:
        print(f"\n{'!'*60}")
        for a in alerts:
            print(f"  {a}")
        print(f"{'!'*60}")
        if use_telegram:
            msg = "\n".join(alerts) + f"\nCurrent step: {current_step:,}"
            send_telegram(msg)
    else:
        print(f"\n  Status: ON TRACK")

    return result


def main():
    parser = argparse.ArgumentParser(description="Trajectory early-warning for CMDLM training")
    parser.add_argument("log", help="Path to SLURM log file")
    parser.add_argument("--telegram", action="store_true", help="Send Telegram alert on deviation")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of human-readable")
    args = parser.parse_args()

    if not os.path.exists(args.log):
        print(f"Log not found: {args.log}", file=sys.stderr)
        sys.exit(1)

    result = check_trajectory(args.log, use_telegram=args.telegram, as_json=args.json)
    sys.exit(0 if result["status"] != "ALERT" else 2)


if __name__ == "__main__":
    main()
