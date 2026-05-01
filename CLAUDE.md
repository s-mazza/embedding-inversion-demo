# CMDLM v2 Training — Project Context

## Architecture (model.py · ConditionalMDLM · _from_scratch=True)
- **Params**: ~86M tied / ~278M untied | hidden=768, heads=12, blocks=8, vocab=250002, L=32
- **Conditioning**: `t_embed: Linear(1,768)` shared → per-layer `AdaLNZeroSplit(768)` (c_proj + t_proj, zero-init, paper Eq. 6-9)
- **Identity at init**: AdaLN zero-init → all blocks identity at step 0; `embed_norm` LayerNorm after token+pos embeddings
- **Tied weights**: `output_proj.weight ≡ token_embed.weight` (saves 192M params)
- **Loss**: `mean(sum_CE_per_sample / t)` — NOT mean CE; noise schedule: `mask_ratio = 1 - exp(-5t)`, `t ~ U[0.02,1]`

## Paper Target (Table 1, jina-v3, paper 2602.11047v3)
- **76.0% token_acc** | **val_loss ≈ 1.60** — both at step 62,500
- Final (~80K steps): token_acc ~0.81, val_loss ~1.32
- Val metric: EMA model, 100% masking (all content tokens), 50 batches × 200 = 10K samples

## Critical Files
| File | Purpose |
|------|---------|
| `model.py` | ConditionalMDLM, AdaLNZeroSplit, TransformerBlock |
| `train.py` | Training loop, loss, EMA, checkpointing, val loop |
| `dataset.py` | EmbeddingDataset, apply_mask |
| `configs/v2_jinav3.yaml` | Active config: bs=200, grad_accum=4, lr=1e-4, max_steps=200K |
| `train.sh` | Docker entry point (pre-flight + torchrun inside container) |
| `sbatch_script.sh` | SLURM submission: 2-GPU + 1-GPU jobs, kills/restarts tmux monitor |
| `findings.md` | All confirmed bugs + fixes (18 total, 2 rounds) |
| `training_health_check.py` | 7 tests: checkpoint, arch, EMA, noise, loss, trajectory, live val |

## Cluster (faretra)
- **SSH**: `ssh faretra` (only host with SLURM+GPU; moro43 is login node, requires password)
- **Jobs**: `squeue -u mazzacano --format='%.10i %.30j %.8T %.10M %R'`
- **Submit**: `ssh faretra "cd ~/embedding-inversion-demo && bash sbatch_script.sh"` (kills old monitor, submits 2 jobs)
- **Checkpoints**: `~/embedding-inversion-demo/checkpoints_v2_jinav3/` (`latest.pt`, `best.pt`, `best_ema.safetensors`)
- **Monitor**: `ssh faretra "tmux attach -t monitor"` — restarts automatically on each sbatch submit
- **Logs**: `slurm-<JOB>.out` (2-GPU), `slurm-<JOB>-1gpu.out` (1-GPU backup)
- **Background processes**: always use `tmux new-session -d -s <name> '...'` — never `nohup ... &` (hangs SSH)

## Bug History (18 fixes across 2 rounds — see findings.md for full details)
**Round 1 (9 bugs):** embedding init std=0.02, EMA fp32 deepcopy, loss formula (sum CE/t not mean CE), AdaLN-Zero gate α, padding mask in attention, t estimation bias (divide by content_len not L), val sampler sharding, mixed_precision flag, checkpoint atomic write

**Round 2 (9 bugs):** per-layer conditioning (paper Eq. 6-9 → AdaLNZeroSplit), embed_norm LayerNorm post-embedding, final_norm → AdaLNZero, GradScaler disabled for BF16, epoch saved/restored in checkpoint, t_proj weight decay, token_acc (EMA, 100% mask) logged in val, val comment fix (50×200=10K), val RNG seed

## Current Training State (update after each job)
- **Active jobs**: 11108132 (2-GPU, 96h), 11108131 (1-GPU backup, 7d) — PENDING as of 2026-05-01
- **Start step**: 0 (fresh start after Round 2 fixes, commit b9ba9e7)
- **Milestones**: step 5K (sanity check), 10K (convergence), 40K (trend), 62,500 (paper target)

## Tooling
| Script | Purpose |
|--------|---------|
| `training_health_check.py` | Pre-flight + post-checkpoint health; `--tests 1,2,3,4,5,6` for CPU-only |
| `training_dashboard.py` | Live ASCII dashboard + paper gap; `python3 training_dashboard.py` |
| `parse_training_trajectory.py` | Curve fit vs paper reference; alerts if projected loss > 1.80 @62500 |
| `haiku_log_analyst.py` | Claude Haiku log analysis → Telegram; `--log LOG --last 100 --telegram` |
| `training_mcp_server.py` | MCP server: `get_training_status`, `tail_log`, `list_checkpoints`, etc. |
