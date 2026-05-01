# Training Tools Reference

Six scripts that compress feedback loops while training CMDLM v2 toward the paper target
(76.0% token accuracy, val_loss ≤ 1.60 at step 62,500 of 200K).

---

## Quick Start

```bash
# 1. Install local deps (first time only)
pip install -r requirements.txt

# 2. Install cluster deps (first time only, run on faretra)
pip3 install --user anthropic scipy

# 3. Set env vars (see .env.example)
export ANTHROPIC_API_KEY="sk-ant-..."   # for haiku_log_analyst.py + MCP server

# 4. Run the test suite
bash run_tests.sh

# 5. Check training status from anywhere
python3 training_dashboard.py
```

---

## Tool 1 — CLAUDE.md

**Purpose:** Compressed project context loaded automatically at every Claude Code session start.
Eliminates ~2K tokens of re-orientation per conversation.

**Usage:**
```bash
# Claude Code reads this automatically — no action needed.

# Keep "Current Training State" section current after each job submission:
python3 update_claude_md.py            # fetches live cluster state
python3 update_claude_md.py --dry-run  # preview changes only
```

**Sections:** Architecture (params, shapes), paper target, critical files, cluster commands,
bug history summary (18 fixes), current training state.

---

## Tool 2 — Pre-flight Check (`train.sh` + `training_health_check.py`)

**Purpose:** Catch architecture bugs, checkpoint corruption, and config mismatches before
wasting GPU-hours. Runs automatically when training starts.

**How it works:**
- `train.sh` calls `training_health_check.py --tests 1,2,3,4,5,6 --cpu` before `torchrun`
- If any test FAILs, training aborts with exit code 1
- `cluster_monitor.sh` detects "FAILED" in the log and sends a Telegram alert
- Skipped on fresh start (no checkpoint)

**Manual usage:**
```bash
# Run all 7 tests (needs GPU + data dir for test 7):
python3 training_health_check.py --checkpoint checkpoints_v2_jinav3/latest.pt

# CPU-only subset (used by pre-flight, ~15s):
python3 training_health_check.py --checkpoint checkpoints_v2_jinav3/latest.pt --cpu --tests 1,2,3,4,5,6

# With live val loss (needs data dir, GPU):
python3 training_health_check.py --checkpoint checkpoints_v2_jinav3/latest.pt --data-dir data_jinav3
```

**Tests:**
| # | Name | Deps | What it checks |
|---|------|------|----------------|
| 1 | Checkpoint integrity | checkpoint | keys, step, best_val_loss |
| 2 | Architecture | checkpoint | layers, heads, hidden, tied weights |
| 3 | EMA health | checkpoint | EMA weights diverged from raw by ≥ε |
| 4 | Noise schedule | model | mask_ratio = 1-exp(-5t), t in [0.02,1] |
| 5 | Loss formula | model | loss = sum(CE)/t, not mean CE |
| 6 | Trajectory | checkpoint | val_loss vs paper reference thresholds |
| 7 | Live val | data+GPU | actual val_loss + token_acc on real data |

---

## Tool 3 — Trajectory Warning (`parse_training_trajectory.py`)

**Purpose:** Early alert if training is diverging from paper curve. Fires when projected
val_loss at step 62,500 exceeds 1.80 (12% above paper target 1.60).

**How it works:** Fits `y = a·exp(-b·x) + c` to val_loss history (token_acc uses complement
`1 - acc` to handle the increasing function). Runs from `cluster_monitor.sh` every 10 val evals.

**Usage:**
```bash
# Check trajectory from local log file:
python3 parse_training_trajectory.py slurm-11108132.out

# JSON output (for scripting):
python3 parse_training_trajectory.py slurm-11108132.out --json

# With Telegram alert on deviation:
python3 parse_training_trajectory.py slurm-11108132.out --telegram
```

**Output example:**
```
Trajectory Check  |  step 30,000  |  15 val points
  Current:   val_loss=2.1500  token_acc=0.550
  Projected: val_loss=1.72 (Δ+0.12 vs paper) [✓]
             token_acc=0.73 (Δ-0.03 vs paper) [✓]
  Status: ON TRACK
```

**Alert thresholds:**
- `proj_val_loss @62500 > 1.80` → ALERT
- `proj_token_acc @62500 < 0.70` → ALERT

**Note:** In early training (< 20K steps), curve fitting is noisy and may fire false positives.
Use the `ref_gaps` section (actual vs. paper reference at past milestones) as the primary signal.

---

## Tool 4 — Training MCP Server (`training_mcp_server.py`)

**Purpose:** Give Claude Code structured access to training state without manual SSH.
Registered in `.mcp.json` and activated automatically in Claude Code sessions.

**Requires:** `pip install mcp` (local machine only)

**Available tools:**

| Tool | What it returns |
|------|----------------|
| `get_training_status` | Current step, loss, val_loss, token_acc, LR, ETA |
| `tail_log` | Last N lines of SLURM log (default 50) |
| `list_checkpoints` | Files in `checkpoints_v2_jinav3/` with sizes |
| `get_trajectory` | Projected val_loss/token_acc at step 62,500 |
| `get_queue_status` | `squeue` output for mazzacano |

**Setup:**
The server is pre-configured in `.mcp.json`. Claude Code picks it up automatically.
Requires key-based SSH access to `faretra` (no password prompt).

**Manual test:**
```bash
python3 -c "
import asyncio
from training_mcp_server import list_tools
async def t(): return [x.name for x in await list_tools()]
print(asyncio.run(t()))
"
```

---

## Tool 5 — Haiku Log Analyst (`haiku_log_analyst.py`)

**Purpose:** AI-generated training commentary every 20 eval checkpoints via Telegram.
Claude Haiku reads the last 100 log lines and reports trend, anomalies, and a verdict.

**Cost:** ~$0.08 for a full 200K step run (100 lines × 400 evals × $0.25/MTok).

**Requires:** `ANTHROPIC_API_KEY` environment variable (on the cluster).

**Usage:**
```bash
# Manual analysis (prints to stdout):
python3 haiku_log_analyst.py --log slurm-11108132.out --last 100

# With Telegram notification:
python3 haiku_log_analyst.py --log slurm-11108132.out --last 100 --telegram --label "2-GPU #11108132"
```

**Telegram setup on cluster:**
```bash
# Add to ~/.bashrc or ~/.profile on faretra:
export ANTHROPIC_API_KEY="sk-ant-..."
```

**System prompt (condensed):** MDLM training monitor with paper reference curve, reports
trend vs. reference, anomalies, and ONE_OF[ON TRACK / BEHIND / CRITICAL].

---

## Tool 6 — Training Dashboard (`training_dashboard.py`)

**Purpose:** At-a-glance training summary with ASCII loss curve, paper gap indicators,
progress bar, and ETA. Run from your local machine — SSH-es to faretra automatically.

**Usage:**
```bash
# Auto-discover current job from squeue:
python3 training_dashboard.py

# Specific job:
python3 training_dashboard.py --job 11108132

# From local file:
python3 training_dashboard.py --log slurm-11108132.out

# 1-GPU backup job:
python3 training_dashboard.py --job 11108131 --1gpu
```

**Output:**
```
════════════════════════════════════════════════════════════════
  Training Dashboard  |  job 11108132  |  2 GPU(s)
────────────────────────────────────────────────────────────────
  Progress: [████████░░░░░░░░░░░░░░░░░░░░░░] 20.0%  step 40,000/200,000
  Elapsed:  3:00:00  |  ETA: 12:00:00
  LR:       7.50e-05  |  train_loss: 1.8800
────────────────────────────────────────────────────────────────
  val_loss  (EMA): 1.8600  [target @62500: 1.6]  ▲ (+0.260 vs paper)
  token_acc (EMA): 0.650   [target @62500: 0.76]  ▼ (-0.110 vs paper)
────────────────────────────────────────────────────────────────
  val_loss curve (·=paper target 1.6):
  ...
```

---

## Test Suite

```bash
bash run_tests.sh          # canonical v2 tests (test_training_correctness.py + test_v2_audit2.py)
bash run_tests.sh -v       # verbose
bash run_tests.sh --quick  # audit tests only (faster)
```

**Active test files:**
- `test_training_correctness.py` — 88 tests covering all model + training invariants
- `test_v2_audit2.py` — 25 tests from audit round 2 (issues 1-9)

**Legacy test files** (mmBERT path, no longer the active training path):
`test_bugs.py`, `test_ema.py`, `test_ema_v2.py`, `test_floats.py`, `test_floats2.py`,
`test_nli_limits.py`, `test_optimizer_reset.py`, `test_pos.py`, `test_pos_v2.py`

---

## Workflow: Checking a Running Training Job

```bash
# 1. Quick status check
python3 training_dashboard.py

# 2. Trajectory projection
python3 parse_training_trajectory.py slurm-11108132.out

# 3. Deep health check (on cluster, needs checkpoint)
ssh faretra "cd ~/embedding-inversion-demo && \
  python3 training_health_check.py \
    --checkpoint checkpoints_v2_jinav3/latest.pt --cpu --tests 1,2,3,4,5,6"

# 4. Run Haiku analysis manually
python3 haiku_log_analyst.py \
  --log <(ssh faretra "tail -100 ~/embedding-inversion-demo/slurm-11108132.out") \
  --last 100
```

## Workflow: After Submitting a New Job

```bash
# On cluster:
bash sbatch_script.sh

# Locally:
python3 update_claude_md.py    # update CLAUDE.md with new job IDs + step
bash run_tests.sh              # verify nothing regressed
```
