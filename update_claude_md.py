#!/usr/bin/env python3
"""
Update the "Current Training State" section in CLAUDE.md with live cluster state.

Reads job IDs from .current_jobs (written by sbatch_script.sh), fetches current
step/val_loss from SLURM logs, and rewrites the section in-place.

Usage:
    python3 update_claude_md.py
    python3 update_claude_md.py --dry-run   # print changes without writing
"""

import argparse
import os
import re
import subprocess
import sys
from datetime import date

SSH_HOST   = "faretra"
PROJ_DIR   = os.path.dirname(os.path.abspath(__file__))
CLAUDE_MD  = os.path.join(PROJ_DIR, "CLAUDE.md")
REMOTE_DIR = "~/embedding-inversion-demo"


def _ssh(cmd: str, timeout: int = 20) -> str:
    try:
        r = subprocess.run(["ssh", SSH_HOST, cmd], capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip()
    except Exception:
        return ""


def read_current_jobs() -> dict:
    """Read .current_jobs written by sbatch_script.sh on the cluster."""
    raw = _ssh(f"cat {REMOTE_DIR}/.current_jobs 2>/dev/null")
    jobs = {}
    for line in raw.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            jobs[k.strip()] = v.strip()
    return jobs


def get_squeue() -> dict[str, str]:
    """Return {job_id: state} for mazzacano's current jobs."""
    out = _ssh("squeue -u mazzacano --format='%.10i %.8T' --noheader")
    result = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            result[parts[0].strip()] = parts[1].strip()
    return result


def get_latest_step(job_id: str, suffix: str = "") -> str:
    log = f"{REMOTE_DIR}/slurm-{job_id}{suffix}.out"
    out = _ssh(f"grep -a '^step ' {log} 2>/dev/null | tail -1")
    m = re.search(r"step (\d+)/(\d+)", out)
    if m:
        return f"step {int(m.group(1)):,}/{int(m.group(2)):,}"
    return "no log yet"


def get_best_checkpoint() -> str:
    out = _ssh(
        f"python3 -c \""
        f"import torch, os; "
        f"p = os.path.expanduser('{REMOTE_DIR}/checkpoints_v2_jinav3/best.pt'); "
        f"ckpt = torch.load(p, map_location='cpu', weights_only=False) if os.path.exists(p) else {{}}; "
        f"s = ckpt.get('step', 0); bl = ckpt.get('best_val_loss', float('inf')); "
        f"print(f'step {{s:,}}, val_loss={{bl:.4f}}' if s else 'none')"
        f"\" 2>/dev/null"
    )
    return out if out else "none"


def build_section(jobs_meta: dict, squeue: dict[str, str]) -> str:
    today = date.today().isoformat()
    two = jobs_meta.get("TWO_GPU_JOB", "")
    one = jobs_meta.get("ONE_GPU_JOB", "")
    submitted = jobs_meta.get("SUBMITTED", "")

    lines = []

    # Job states
    entries = []
    if two:
        state = squeue.get(two, "DONE/CANCELLED")
        step = get_latest_step(two) if state == "RUNNING" else state
        entries.append(f"{two} (2-GPU, 96h) — {step}")
    if one:
        state = squeue.get(one, "DONE/CANCELLED")
        step = get_latest_step(one, suffix="-1gpu") if state == "RUNNING" else state
        entries.append(f"{one} (1-GPU backup, 7d) — {step}")

    if entries:
        lines.append(f"- **Active jobs**: {', '.join(entries)} — as of {today}")
    else:
        lines.append(f"- **Active jobs**: none as of {today}")

    if submitted:
        lines.append(f"- **Last submitted**: {submitted}")

    # Best checkpoint
    best = get_best_checkpoint()
    lines.append(f"- **Best checkpoint**: {best}")
    lines.append(
        "- **Milestones**: step 5K (sanity check), 10K (convergence), 40K (trend), 62,500 (paper target)"
    )
    return "\n".join(lines)


def update_section(content: str, new_body: str) -> str:
    marker = "## Current Training State"
    start_idx = content.find(marker)
    if start_idx == -1:
        print(f"Section '{marker}' not found in CLAUDE.md", file=sys.stderr)
        return content

    header_end = content.index("\n", start_idx) + 1
    # Find next ## section or EOF
    m = re.search(r"\n## ", content[header_end:])
    body_end = header_end + m.start() if m else len(content)

    return content[:header_end] + new_body + "\n" + content[body_end:]


def main():
    parser = argparse.ArgumentParser(description="Update CLAUDE.md Current Training State")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("Fetching cluster state…", flush=True)
    jobs_meta = read_current_jobs()
    squeue = get_squeue()

    if not jobs_meta:
        print("  .current_jobs not found on cluster — run sbatch_script.sh first")
        print("  Showing squeue directly:")
        for jid, state in squeue.items():
            print(f"    {jid}: {state}")
    else:
        print(f"  TWO_GPU_JOB={jobs_meta.get('TWO_GPU_JOB')}  ONE_GPU_JOB={jobs_meta.get('ONE_GPU_JOB')}")

    new_body = build_section(jobs_meta, squeue)
    print(f"\nNew section:\n{new_body}\n")

    if args.dry_run:
        print("(dry-run — not writing)")
        return

    with open(CLAUDE_MD) as f:
        content = f.read()

    updated = update_section(content, new_body)
    if updated == content:
        print("No changes.")
        return

    with open(CLAUDE_MD, "w") as f:
        f.write(updated)
    print(f"Updated {CLAUDE_MD}")


if __name__ == "__main__":
    main()
