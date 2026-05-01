#!/usr/bin/env python3
"""
MCP server for CMDLM v2 training status.

Exposes 5 tools to Claude Code via the MCP stdio transport:
  - get_training_status   → current step, loss, token_acc, ETA
  - tail_log              → last N lines from a SLURM log
  - list_checkpoints      → checkpoint files with metadata
  - get_trajectory        → projected val_loss/token_acc at step 62500
  - get_queue_status      → squeue output for mazzacano's jobs

Install:
    pip install mcp
    Add to ~/.claude/settings.json under "mcpServers":
    {
      "training": {
        "command": "python3",
        "args": ["/home/mazza/Documents/embedding-inversion-demo/training_mcp_server.py"]
      }
    }

Requires:
    SSH access to faretra (key-based, no password prompt)
"""

import asyncio
import json
import re
import subprocess
import sys
from typing import Any

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent, INVALID_PARAMS, INTERNAL_ERROR
except ImportError:
    print("mcp package not installed. Run: pip install mcp", file=sys.stderr)
    sys.exit(1)

SSH_HOST     = "faretra"
LOG_DIR      = "~/embedding-inversion-demo"
CKPT_DIR     = "~/embedding-inversion-demo/checkpoints_v2_jinav3"
SLURM_USER   = "mazzacano"
MAX_STEPS    = 200_000
PAPER_TARGET = {"step": 62_500, "val_loss": 1.60, "token_acc": 0.76}

app = Server("training")


def _ssh(cmd: str, timeout: int = 20) -> str:
    result = subprocess.run(
        ["ssh", SSH_HOST, cmd],
        capture_output=True, text=True, timeout=timeout
    )
    return result.stdout


def _parse_latest_step(log_text: str) -> dict:
    step_re  = re.compile(r'^step (\d+)/(\d+) \| loss ([\d.]+) \| acc ([\d.]+) \| lr ([\S]+).*elapsed ([\d.]+)min', re.M)
    val_re   = re.compile(r'val_loss \(ema\): ([\d.]+)')
    acc_re   = re.compile(r'token_acc \(EMA, 100% mask\): ([\d.]+)')
    best_re  = re.compile(r'best val_loss.*?(\d+\.\d+).*?step (\d+)', re.I)

    step_m = None
    for m in step_re.finditer(log_text):
        step_m = m

    if not step_m:
        return {}

    val_m  = None
    for m in val_re.finditer(log_text):
        val_m = m
    acc_m  = None
    for m in acc_re.finditer(log_text):
        acc_m = m

    step = int(step_m.group(1))
    max_steps = int(step_m.group(2))
    elapsed_min = float(step_m.group(6))
    rate = step / max(elapsed_min, 1e-9)
    remaining_min = (max_steps - step) / max(rate, 1e-9)

    return {
        "step": step,
        "max_steps": max_steps,
        "progress_pct": round(step / max_steps * 100, 1),
        "train_loss": float(step_m.group(3)),
        "train_acc": float(step_m.group(4)),
        "lr": step_m.group(5),
        "elapsed_min": round(elapsed_min, 1),
        "eta_hours": round(remaining_min / 60, 1),
        "val_loss": float(val_m.group(1)) if val_m else None,
        "token_acc": float(acc_m.group(1)) if acc_m else None,
        "paper_target": PAPER_TARGET,
    }


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="get_training_status",
            description=(
                "Get current training status: step, loss, val_loss, token_acc, LR, elapsed, ETA. "
                "Auto-discovers the active SLURM job."
            ),
            inputSchema={"type": "object", "properties": {
                "job_id": {"type": "string", "description": "Optional job ID; auto-discovered if omitted"}
            }},
        ),
        Tool(
            name="tail_log",
            description="Return last N lines from a SLURM training log on faretra.",
            inputSchema={"type": "object", "properties": {
                "job_id": {"type": "string", "description": "SLURM job ID"},
                "n_lines": {"type": "integer", "description": "Number of lines to return (default 50)", "default": 50},
                "is_1gpu": {"type": "boolean", "description": "Use 1-GPU log suffix (-1gpu.out)", "default": False},
            }, "required": ["job_id"]},
        ),
        Tool(
            name="list_checkpoints",
            description="List checkpoint files in checkpoints_v2_jinav3/ with file sizes and modification times.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="get_trajectory",
            description=(
                "Parse val_loss history from log, fit a decay curve, and project val_loss "
                "and token_acc at step 62,500. Returns ON_TRACK or ALERT status."
            ),
            inputSchema={"type": "object", "properties": {
                "job_id": {"type": "string", "description": "SLURM job ID; auto-discovered if omitted"}
            }},
        ),
        Tool(
            name="get_queue_status",
            description="Show squeue output for mazzacano's jobs on faretra.",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    try:
        if name == "get_training_status":
            return await _get_training_status(arguments)
        elif name == "tail_log":
            return await _tail_log(arguments)
        elif name == "list_checkpoints":
            return await _list_checkpoints(arguments)
        elif name == "get_trajectory":
            return await _get_trajectory(arguments)
        elif name == "get_queue_status":
            return await _get_queue_status(arguments)
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {e}")]


async def _get_training_status(args: dict) -> list[TextContent]:
    job_id = args.get("job_id")

    if not job_id:
        queue = _ssh(f"squeue -u {SLURM_USER} --format='%.10i %.8T' --noheader")
        running = re.findall(r'(\d+)\s+RUNNING', queue)
        if not running:
            running = re.findall(r'(\d+)\s+PENDING', queue)
        if not running:
            return [TextContent(type="text", text="No active jobs found for mazzacano")]
        job_id = running[0]

    log = _ssh(f"cat {LOG_DIR}/slurm-{job_id}.out 2>/dev/null || echo ''")
    if not log.strip():
        return [TextContent(type="text", text=f"Log not found or empty for job {job_id}")]

    status = _parse_latest_step(log)
    if not status:
        return [TextContent(type="text", text=f"Job {job_id}: log exists but no step lines yet (still starting?)")]

    status["job_id"] = job_id
    lines = [
        f"Job {job_id} — step {status['step']:,}/{status['max_steps']:,} ({status['progress_pct']}%)",
        f"  train_loss: {status['train_loss']:.4f}  train_acc: {status['train_acc']:.3f}  lr: {status['lr']}",
        f"  elapsed: {status['elapsed_min']:.0f}min  ETA: {status['eta_hours']:.1f}h",
    ]
    if status["val_loss"] is not None:
        vl_gap = status["val_loss"] - PAPER_TARGET["val_loss"]
        acc_gap = (status["token_acc"] or 0) - PAPER_TARGET["token_acc"]
        lines.append(f"  val_loss (EMA): {status['val_loss']:.4f}  (Δ{vl_gap:+.3f} vs paper target {PAPER_TARGET['val_loss']})")
        lines.append(f"  token_acc (EMA): {status['token_acc']:.3f}  (Δ{acc_gap:+.3f} vs paper target {PAPER_TARGET['token_acc']})")
    else:
        lines.append("  val_loss: not available yet")

    return [TextContent(type="text", text="\n".join(lines))]


async def _tail_log(args: dict) -> list[TextContent]:
    job_id = args["job_id"]
    n = args.get("n_lines", 50)
    suffix = "-1gpu" if args.get("is_1gpu") else ""
    log = _ssh(f"tail -n {n} {LOG_DIR}/slurm-{job_id}{suffix}.out 2>/dev/null")
    if not log.strip():
        return [TextContent(type="text", text=f"Log not found: slurm-{job_id}{suffix}.out")]
    return [TextContent(type="text", text=log)]


async def _list_checkpoints(_args: dict) -> list[TextContent]:
    out = _ssh(f"ls -lh {CKPT_DIR}/ 2>/dev/null | grep -v '^total'")
    if not out.strip():
        return [TextContent(type="text", text=f"No checkpoints found in {CKPT_DIR}/")]
    return [TextContent(type="text", text=out.strip())]


async def _get_trajectory(args: dict) -> list[TextContent]:
    job_id = args.get("job_id")
    if not job_id:
        queue = _ssh(f"squeue -u {SLURM_USER} --format='%.10i %.8T' --noheader")
        running = re.findall(r'(\d+)\s+RUNNING', queue)
        if not running:
            running = re.findall(r'(\d+)\s+PENDING', queue)
        if not running:
            return [TextContent(type="text", text="No active jobs found")]
        job_id = running[0]

    out = _ssh(
        f"python3 {LOG_DIR}/parse_training_trajectory.py "
        f"{LOG_DIR}/slurm-{job_id}.out --json 2>/dev/null"
    )
    if not out.strip():
        return [TextContent(type="text", text="parse_training_trajectory.py not available or log empty")]

    try:
        data = json.loads(out)
        status = data.get("status", "UNKNOWN")
        step = data.get("current_step", "?")
        proj_vl = data.get("proj_val_loss_at_62500")
        proj_acc = data.get("proj_token_acc_at_62500")
        num_pts = data.get("num_val_points", 0)
        alerts = data.get("alerts", [])

        lines = [f"Trajectory [{status}]  |  current step: {step:,}  |  {num_pts} val points"]
        if proj_vl:
            lines.append(f"  Projected @62500: val_loss={proj_vl:.3f}  token_acc={proj_acc:.3f}")
            lines.append(f"  Paper target:     val_loss=1.60  token_acc=0.76")
        for g in data.get("ref_gaps", []):
            lines.append(g)
        if alerts:
            lines.extend(alerts)
        return [TextContent(type="text", text="\n".join(lines))]
    except Exception:
        return [TextContent(type="text", text=out.strip())]


async def _get_queue_status(_args: dict) -> list[TextContent]:
    out = _ssh(f"squeue -u {SLURM_USER} --format='%.10i %.30j %.8T %.10M %R' 2>/dev/null")
    if not out.strip():
        return [TextContent(type="text", text="No jobs in queue (or squeue unavailable)")]
    return [TextContent(type="text", text=out.strip())]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
