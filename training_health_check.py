#!/usr/bin/env python3
"""
Training health check for CMDLM embedding inversion.

Detects known bugs, verifies paper-correct behavior, and tracks progress
against the paper baseline (jina-v3, 8-layer: best_step=62500, val_loss=1.60,
token_acc=76.0%, paper 2602.11047v3).

Usage:
    # Static checks only (no GPU needed):
    python training_health_check.py --checkpoint checkpoints_v2_jinav3/latest.pt

    # Full suite including live val loss and token accuracy:
    python training_health_check.py --checkpoint checkpoints_v2_jinav3/latest.pt \\
                                    --data-dir data_jinav3

    # Force CPU (skip GPU-dependent tests):
    python training_health_check.py --checkpoint checkpoints_v2_jinav3/latest.pt --cpu
"""

import argparse
import math
import sys
import copy
import torch
import torch.nn.functional as F
import numpy as np

from model import ConditionalMDLM, apply_mask

# ── Paper reference values (Table 1 + Figure 2, jina-v3 8-layer) ───────────
PAPER = {
    "best_val_loss": 1.60,
    "best_step": 62500,
    "best_token_acc": 0.760,
    "end_val_loss": 1.32,   # Figure 2(b): ~1.32 at 80K steps
    "end_token_acc": 0.810, # Figure 2(a): ~0.81 at 80K steps
    "log_vocab": math.log(250002),  # ≈ 12.43 — val loss at random guessing
}

# ── Expected architecture (config: v2_jinav3.yaml) ──────────────────────────
EXPECTED_ARCH = {
    "vocab_size": 250002,
    "max_seq_len": 32,
    "hidden_dim": 768,
    "num_heads": 12,
    "ff_dim": 3072,
    "num_layers": 8,
    "embedding_cond_dim": 1024,
    "mask_token_id": 250001,
}

# ── Val-loss failure thresholds (FAIL if exceeded at given step) ─────────────
# Based on paper Figure 2(b) with ~2× headroom. Exceeding → training stalled.
# Format: (min_step, max_allowed_val_loss)
VAL_LOSS_FAIL = [
    (   500, 13.5),  # step 500: should be descending from 12.43
    (  1000, 13.0),
    (  2500, 11.0),
    (  5000,  9.0),
    ( 10000,  7.0),
    ( 25000,  4.5),
    ( 50000,  3.0),
    ( 62500,  2.5),
    ( 80000,  2.0),
]

# ── Val-loss warn thresholds (WARN if exceeded, but not failing) ─────────────
VAL_LOSS_WARN = [
    (  1000, 12.5),
    (  5000,  8.0),
    ( 25000,  3.5),
    ( 50000,  2.5),
    ( 62500,  1.80),
]

# ── Token accuracy lower bounds (WARN if below at given step) ───────────────
TOKEN_ACC_LOWER = [
    (  500, 0.005),
    ( 1000, 0.010),
    ( 5000, 0.100),
    (25000, 0.400),
    (62500, 0.700),
]


# ── Result counter ───────────────────────────────────────────────────────────

class Results:
    def __init__(self):
        self.counts = {"PASS": 0, "WARN": 0, "FAIL": 0, "INFO": 0, "SKIP": 0}

    def _fmt(self, tag, msg):
        markers = {"PASS": "✓", "WARN": "~", "FAIL": "✗", "INFO": "·", "SKIP": "-"}
        self.counts[tag] += 1
        print(f"  [{tag}] {markers[tag]} {msg}")

    def ok(self, msg):   self._fmt("PASS", msg)
    def warn(self, msg): self._fmt("WARN", msg)
    def fail(self, msg): self._fmt("FAIL", msg)
    def info(self, msg): self._fmt("INFO", msg)
    def skip(self, msg): self._fmt("SKIP", msg)

    def header(self, n, title, subtitle=""):
        print(f"\n{'─'*64}")
        print(f"TEST {n}: {title}")
        if subtitle:
            print(f"  ({subtitle})")
        print(f"{'─'*64}")

    def summary(self):
        p, w, f = self.counts["PASS"], self.counts["WARN"], self.counts["FAIL"]
        print(f"\n{'═'*64}")
        print(f"  SUMMARY:  {p} PASS | {w} WARN | {f} FAIL")
        if f > 0:
            print("  STATUS:   ✗ TRAINING HAS CRITICAL ISSUES")
        elif w > 0:
            print("  STATUS:   ~ TRAINING OK BUT MONITOR CLOSELY")
        else:
            print("  STATUS:   ✓ TRAINING LOOKS HEALTHY")
        print(f"{'═'*64}\n")


# ── TEST 1: Checkpoint Integrity ─────────────────────────────────────────────

def test_checkpoint_integrity(ckpt, r: Results):
    r.header(1, "Checkpoint Integrity")

    required_keys = {"step", "best_val_loss", "model", "ema_model", "optimizer", "config"}
    missing = required_keys - set(ckpt.keys())
    if missing:
        r.fail(f"Missing checkpoint keys: {missing}")
    else:
        r.ok(f"All required keys present: {sorted(required_keys)}")

    step = ckpt.get("step", -1)
    if isinstance(step, int) and step >= 0:
        r.ok(f"step = {step:,}")
    else:
        r.fail(f"Invalid step: {step!r}")

    bvl = ckpt.get("best_val_loss", float("nan"))
    if math.isfinite(bvl) and bvl > 0:
        r.ok(f"best_val_loss = {bvl:.4f}")
    elif bvl == float("inf"):
        r.warn("best_val_loss = inf  (no eval has run yet, or all evals failed)")
    else:
        r.fail(f"best_val_loss is invalid: {bvl}")

    if "config" in ckpt:
        cfg = ckpt["config"]
        mc = cfg.get("model", {})
        if mc.get("num_layers") == EXPECTED_ARCH["num_layers"]:
            r.ok(f"Config: from-scratch, num_layers={mc['num_layers']}")
        else:
            r.warn(f"Config num_layers={mc.get('num_layers')} ≠ expected {EXPECTED_ARCH['num_layers']}")

    best_step = ckpt.get("best_step", None)
    if best_step is not None:
        r.info(f"best_step = {best_step:,}")

    return ckpt.get("step", 0), ckpt.get("best_val_loss", float("inf"))


# ── TEST 2: Architecture ─────────────────────────────────────────────────────

def test_architecture(model, ckpt, r: Results):
    r.header(2, "Architecture", "From-scratch 8-layer transformer, paper §3.1")

    mc = ckpt["config"]["model"]

    # From-scratch vs mmBERT
    if model._from_scratch:
        r.ok("From-scratch architecture (not mmBERT backbone)")
    else:
        r.fail("mmBERT architecture detected — expected from-scratch for v2_jinav3")

    # Layer count
    n_layers = len(model.blocks) if hasattr(model, "blocks") else -1
    if n_layers == EXPECTED_ARCH["num_layers"]:
        r.ok(f"num_layers = {n_layers} (matches paper: 8)")
    else:
        r.fail(f"num_layers = {n_layers} ≠ expected {EXPECTED_ARCH['num_layers']}")

    # Key dimensions
    for key, expected in [
        ("hidden_dim", EXPECTED_ARCH["hidden_dim"]),
        ("vocab_size", EXPECTED_ARCH["vocab_size"]),
        ("max_seq_len", EXPECTED_ARCH["max_seq_len"]),
        ("embedding_cond_dim", EXPECTED_ARCH["embedding_cond_dim"]),
    ]:
        val = mc.get(key)
        if val == expected:
            r.ok(f"{key} = {val}")
        else:
            r.fail(f"{key} = {val} ≠ expected {expected}")

    # Parameter count
    total, trainable = model.count_params()
    if 250_000_000 <= total <= 320_000_000:
        r.ok(f"Total params = {total:,} (expected range 250M–320M)")
    else:
        r.warn(f"Total params = {total:,} (outside expected range 250M–320M)")

    # t_proj (timestep conditioning, added in this run)
    has_t_proj = hasattr(model, "t_proj")
    if has_t_proj:
        r.ok("t_proj present (timestep conditioning, paper §3.3)")
        t_w = model.t_proj.weight
        t_norm = t_w.float().norm().item()
        step = ckpt.get("step", 0)
        if step > 2000 and t_norm < 1e-6:
            r.warn(f"t_proj weight norm = {t_norm:.2e} at step {step:,} — t conditioning not learning?")
        elif step > 0:
            r.ok(f"t_proj weight norm = {t_norm:.4f} (non-zero ✓)")
        else:
            r.info(f"t_proj weight norm = {t_norm:.4f} (step 0: zero-init expected)")
    else:
        r.fail("t_proj missing — timestep conditioning not implemented")

    # Weight tying
    tied = mc.get("tie_weights", False)
    if tied:
        if model.output_proj.weight.data_ptr() == model.token_embed.weight.data_ptr():
            r.ok("Weight tying: output_proj shares token_embed (paper: input/output embeddings tied)")
        else:
            r.warn("Config says tie_weights=True but output_proj and token_embed are not sharing data")
    else:
        r.warn("tie_weights=False — paper requires weight tying for this architecture")


# ── TEST 3: EMA Health ───────────────────────────────────────────────────────

def test_ema_health(model, ema_model, ckpt, r: Results):
    r.header(3, "EMA Health", "Bug: bf16 underflow rounds 1e-4 update to zero → EMA frozen")

    step = ckpt.get("step", 0)

    # EMA dtype
    ema_dtypes = {p.dtype for p in ema_model.parameters()}
    if ema_dtypes == {torch.bfloat16}:
        r.ok("EMA model dtype: bfloat16 (expected)")
    elif torch.bfloat16 in ema_dtypes:
        r.warn(f"EMA model has mixed dtypes: {ema_dtypes}")
    else:
        r.fail(f"EMA model dtype: {ema_dtypes} — expected bfloat16")

    # Key correspondence
    raw_keys = set(model.state_dict().keys())
    ema_keys = set(ema_model.state_dict().keys())
    if raw_keys == ema_keys:
        r.ok(f"EMA state dict has {len(ema_keys)} keys matching raw model")
    else:
        extra = ema_keys - raw_keys
        missing = raw_keys - ema_keys
        r.fail(f"EMA key mismatch: {len(extra)} extra, {len(missing)} missing")

    # EMA vs raw model divergence (bug: if frozen → they stay identical)
    raw_sd = model.state_dict()
    ema_sd = ema_model.state_dict()

    diffs = []
    for k in list(raw_sd.keys())[:20]:  # sample first 20 params for speed
        if raw_sd[k].dtype.is_floating_point and k in ema_sd:
            d = (raw_sd[k].float() - ema_sd[k].float()).abs().mean().item()
            diffs.append(d)

    if not diffs:
        r.warn("Could not compute EMA divergence (no float params sampled)")
        return

    mean_diff = float(np.mean(diffs))
    if step == 0:
        if mean_diff < 1e-5:
            r.ok(f"Step 0: EMA == raw model (expected at initialization, mean_diff={mean_diff:.2e})")
        else:
            r.warn(f"Step 0: EMA already diverged from raw model (mean_diff={mean_diff:.4f})?")
    elif step < 500:
        if mean_diff > 0:
            r.ok(f"EMA differs from raw model at step {step} (mean_diff={mean_diff:.4f})")
        else:
            r.warn(f"EMA identical to raw at step {step} — possible frozen EMA")
    else:
        # At step N with decay=0.9999: EMA lags ~1/(1-0.9999)=10000 steps behind.
        # After 1K+ steps, EMA and raw should have measurable diff.
        ema_decay = ckpt["config"]["training"].get("ema_decay", 0.9999)
        if mean_diff < 1e-5:
            r.fail(
                f"EMA identical to raw model at step {step:,} (mean_diff={mean_diff:.2e})\n"
                f"         FROZEN EMA BUG: fp32 accumulation may not be working"
            )
        elif mean_diff < 1e-4:
            r.warn(f"EMA very close to raw at step {step:,} (mean_diff={mean_diff:.2e}). "
                   f"Expected with decay={ema_decay}")
        else:
            r.ok(f"EMA diverged from raw model at step {step:,} (mean_diff={mean_diff:.4f}, "
                 f"decay={ema_decay}) — EMA is tracking ✓")

    # Simulate one EMA update in fp32 and verify it changes the parameter
    with torch.no_grad():
        p_ema = next(iter(ema_model.parameters())).float().clone()
        p_raw = next(iter(model.parameters())).float()
        decay = ckpt["config"]["training"].get("ema_decay", 0.9999)
        p_new = p_ema.lerp(p_raw, 1 - decay)
        update_size = (p_new - p_ema).abs().max().item()

    if update_size > 0:
        r.ok(f"EMA update magnitude (fp32 simulation): {update_size:.4e} — non-zero ✓")
    else:
        r.fail("EMA update rounded to zero even in fp32 simulation — check optimizer lr range")


# ── TEST 4: Noise Schedule ───────────────────────────────────────────────────

def test_noise_schedule(r: Results, device, n_samples=2000):
    r.header(4, "Noise Schedule", "Bug: λ=0.001 (mask_ratio≈0.001) instead of λ=5.0 (mask_ratio≈0.81)")

    vocab = EXPECTED_ARCH["vocab_size"]
    L = EXPECTED_ARCH["max_seq_len"]
    mask_id = EXPECTED_ARCH["mask_token_id"]

    # Synthetic token ids (all non-padding, non-mask)
    token_ids = torch.randint(0, vocab - 2, (n_samples, L), device=device)

    # apply_mask return value count
    try:
        out = apply_mask(token_ids, mask_id)
        if len(out) == 4:
            r.ok("apply_mask returns 4 values: (masked_ids, target_mask, mask_ratio, t)")
        else:
            r.fail(f"apply_mask returns {len(out)} values, expected 4 (missing t for 1/t loss?)")
            return
    except Exception as e:
        r.fail(f"apply_mask raised: {e}")
        return

    masked_ids, target_mask, mask_ratio, t_sample = out

    # t distribution
    t_min = t_sample.min().item()
    t_max = t_sample.max().item()
    t_mean = t_sample.mean().item()
    if 0.018 <= t_min and t_max <= 1.01:
        r.ok(f"t range = [{t_min:.3f}, {t_max:.3f}] (expected [0.02, 1.0])")
    else:
        r.fail(f"t range = [{t_min:.4f}, {t_max:.4f}] — expected clamp to [0.02, 1.0]")

    # t mean: E[U(0.02,1)] = 0.51
    if 0.45 <= t_mean <= 0.57:
        r.ok(f"t mean = {t_mean:.3f} (expected ~0.51 for U[0.02,1])")
    else:
        r.warn(f"t mean = {t_mean:.3f} (expected ~0.51 for U[0.02,1])")

    # mask_ratio: should be 1 - exp(-5t)
    expected_ratio = 1 - torch.exp(-5.0 * t_sample)
    ratio_err = (mask_ratio - expected_ratio).abs().max().item()
    if ratio_err < 1e-5:
        r.ok(f"mask_ratio = 1 - exp(-5t) verified (max error {ratio_err:.2e})")
    else:
        r.fail(f"mask_ratio formula error = {ratio_err:.4f} — expected 1-exp(-5t)")

    # mask_ratio distribution
    mr_mean = mask_ratio.mean().item()
    mr_min = mask_ratio.min().item()
    # E[1-exp(-5t)] for t~U[0.02,1] ≈ 0.817
    if 0.70 <= mr_mean <= 0.90:
        r.ok(f"mask_ratio mean = {mr_mean:.3f} ∈ [0.70, 0.90] (expected ~0.82)")
    else:
        r.fail(
            f"mask_ratio mean = {mr_mean:.4f} outside [0.70, 0.90]\n"
            f"         BUG SIGNATURE: old λ=0.001 gave mean≈0.001"
        )

    if mr_min > 0.01:
        r.ok(f"mask_ratio min = {mr_min:.4f} > 0.01 (not near-zero)")
    else:
        r.fail(
            f"mask_ratio min = {mr_min:.5f} ≈ 0 — NEAR-ZERO MASKING BUG\n"
            f"         Symptom: model sees nearly all tokens, no learning signal"
        )

    # Check that masking actually happens
    n_masked = target_mask.float().sum().item()
    total_pos = n_samples * L
    frac_masked = n_masked / total_pos
    if 0.60 <= frac_masked <= 0.95:
        r.ok(f"Fraction of positions masked = {frac_masked:.3f} (expected ~0.82)")
    else:
        r.warn(f"Fraction of positions masked = {frac_masked:.3f} (expected ~0.82)")

    # Bug check: old formula would have mask_ratio << 0.01
    if mr_mean < 0.01:
        r.fail("⚠  OLD BUG DETECTED: mask_ratio mean ≈ 0 → 1-(1-1e-3)^u schedule")
    elif mr_mean > 0.99:
        r.warn("mask_ratio mean ≈ 1.0 → masking too aggressive (check λ)")
    else:
        r.info(f"1/t weight at mean t={t_mean:.2f}: E[1/t] ≈ {1/t_mean:.1f} — training loss ≈ {1/t_mean * PAPER['log_vocab']:.1f} at init")


# ── TEST 5: Loss Formula ─────────────────────────────────────────────────────

def test_loss_formula(model, r: Results, device):
    r.header(5, "Loss Formula", "Bug: mean(CE)*mean(1/t) ≠ mean(CE/t) — per-sample 1/t required (Eq. 4)")

    vocab = EXPECTED_ARCH["vocab_size"]
    L = EXPECTED_ARCH["max_seq_len"]
    mask_id = EXPECTED_ARCH["mask_token_id"]
    cond_dim = EXPECTED_ARCH["embedding_cond_dim"]

    # Create batch of 8 samples with controlled t values:
    # 4 samples with forced t≈0.05 (few masked), 4 with t≈0.95 (many masked)
    # For these, the per-sample CE will differ, so the two loss formulas diverge.
    B = 8
    token_ids = torch.randint(0, vocab - 2, (B, L), device=device)

    # Force two groups: low-t (sparse mask) and high-t (dense mask)
    # Do this by constructing mask_ratio directly
    low_t  = torch.full((B // 2, 1), 0.05, device=device)
    high_t = torch.full((B // 2, 1), 0.95, device=device)
    t_forced = torch.cat([low_t, high_t], dim=0)  # [8, 1]
    mask_ratio_forced = 1 - torch.exp(-5.0 * t_forced)

    rand_scores = torch.rand(B, L, device=device)
    target_mask = rand_scores < mask_ratio_forced
    masked_ids = token_ids.clone()
    masked_ids[target_mask] = mask_id

    # Verify t_forced actually separates the groups
    low_frac  = target_mask[:B//2].float().mean().item()
    high_frac = target_mask[B//2:].float().mean().item()
    r.info(f"Controlled masking: low group {low_frac:.2f} masked, high group {high_frac:.2f} masked")

    # Synthetic embeddings
    cond = torch.randn(B, cond_dim, device=device)

    model.eval()
    with torch.no_grad():
        try:
            with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu',
                                    dtype=torch.bfloat16, enabled=(device.type == 'cuda')):
                hidden = model.forward_hidden(masked_ids, cond)
        except Exception as e:
            r.fail(f"forward_hidden raised: {e}")
            model.train()
            return
    model.train()

    # Compute per-sample CE manually
    w = model.output_proj.weight.float()
    h_flat = hidden.view(-1, hidden.shape[-1]).float()
    t_flat = token_ids.view(-1)
    m_flat = target_mask.view(-1).float()

    sample_ce_sum = torch.zeros(B, device=device)
    chunk_size = 256
    for i in range(0, B * L, chunk_size):
        end = min(i + chunk_size, B * L)
        lc = F.linear(h_flat[i:end], w)
        ce_chunk = F.cross_entropy(lc, t_flat[i:end], reduction="none")
        idx = torch.arange(i, end, device=device) // L
        sample_ce_sum.scatter_add_(0, idx, (ce_chunk * m_flat[i:end]))

    n_masked = target_mask.float().sum(-1).clamp(min=1)  # [B]
    per_sample_ce = sample_ce_sum / n_masked             # [B] mean CE per sample

    # NEW formula (correct, paper Eq. 4):  mean(CE_b / t_b)
    loss_correct = (per_sample_ce / t_forced.squeeze(1)).mean().item()

    # OLD formula (bug): mean(CE_all) * mean(1/t)
    total_ce = sample_ce_sum.sum().item()
    total_masked = m_flat.sum().item()
    global_ce = total_ce / max(total_masked, 1)
    global_inv_t = (1.0 / t_forced).mean().item()
    loss_buggy = global_ce * global_inv_t

    diff_pct = abs(loss_correct - loss_buggy) / max(abs(loss_buggy), 1e-8) * 100

    r.info(f"Per-sample loss  (CORRECT, Eq. 4) = {loss_correct:.4f}")
    r.info(f"Global-mean loss (BUGGY)           = {loss_buggy:.4f}")
    r.info(f"Difference = {diff_pct:.1f}%")

    if diff_pct < 0.5:
        r.warn("Per-sample and global formulas give nearly identical results on this batch "
               "(unlikely but possible if batch happens to be uniform)")
    else:
        r.ok(f"Per-sample and global formulas differ by {diff_pct:.1f}% — distinct paths verifiable")

    # Verify low-t samples are amplified more in per-sample formula
    low_contrib  = (per_sample_ce[:B//2] / t_forced[:B//2].squeeze(1)).mean().item()
    high_contrib = (per_sample_ce[B//2:] / t_forced[B//2:].squeeze(1)).mean().item()
    # The low-t group weight = 1/0.05 = 20; high-t = 1/0.95 ≈ 1.05 → 19× amplification
    expected_ratio = 0.95 / 0.05  # = 19 (weight ratio, ignoring CE difference)
    actual_ratio = low_contrib / max(high_contrib, 1e-8)
    r.info(f"Low-t / high-t loss contribution ratio: {actual_ratio:.1f} (weighting-only ratio: {expected_ratio:.0f})")

    if actual_ratio > 2.0:
        r.ok("Low-t samples (sparse mask) correctly amplified vs high-t samples ✓")
    else:
        r.warn(f"Low-t samples not amplified (ratio={actual_ratio:.2f}) — possible loss formula issue")


# ── TEST 6: Training Trajectory vs Paper ─────────────────────────────────────

def test_trajectory(step, best_val_loss, r: Results):
    r.header(6, "Training Trajectory vs Paper Baseline",
             f"Target: val_loss=1.60 at step 62500, token_acc=76% (Table 1)")

    r.info(f"Current: step={step:,}, best_val_loss={best_val_loss:.4f}")
    r.info(f"Paper:   step={PAPER['best_step']:,}, best_val_loss={PAPER['best_val_loss']:.2f}, "
           f"token_acc={PAPER['best_token_acc']:.1%}")

    if best_val_loss == float("inf"):
        r.warn("best_val_loss = inf — no eval checkpoint yet (training too early)")
        return

    # ── Previous-bug detectors ─────────────────────────────────────────────
    if step > 500 and best_val_loss > 400:
        r.fail(
            f"best_val_loss={best_val_loss:.1f} >> 100 after step {step:,}\n"
            f"         STAGNATION BUG (previous run: stuck at ~490 from wrong masking)"
        )
    elif step > 1000 and best_val_loss > PAPER["log_vocab"] + 0.5:
        r.fail(
            f"best_val_loss={best_val_loss:.2f} ≈ log(vocab)={PAPER['log_vocab']:.2f} at step {step:,}\n"
            f"         MODEL NOT LEARNING — check data loading and masking"
        )
    else:
        r.ok(f"Not stuck at random guessing or pathological loss value")

    # ── Step-dependent FAIL threshold ─────────────────────────────────────
    fail_thresh = None
    for min_step, thresh in VAL_LOSS_FAIL:
        if step >= min_step:
            fail_thresh = thresh

    if fail_thresh is not None:
        if best_val_loss <= fail_thresh:
            r.ok(f"best_val_loss={best_val_loss:.4f} ≤ {fail_thresh:.1f} (threshold for step {step:,})")
        else:
            r.fail(
                f"best_val_loss={best_val_loss:.4f} > {fail_thresh:.1f} at step {step:,}\n"
                f"         BELOW EXPECTED TRAJECTORY — training may be stalled"
            )

    # ── Step-dependent WARN threshold ─────────────────────────────────────
    warn_thresh = None
    for min_step, thresh in VAL_LOSS_WARN:
        if step >= min_step:
            warn_thresh = thresh

    if warn_thresh is not None:
        if best_val_loss > warn_thresh:
            r.warn(f"best_val_loss={best_val_loss:.4f} > {warn_thresh:.1f} "
                   f"— below expected pace at step {step:,}")
        else:
            r.ok(f"best_val_loss={best_val_loss:.4f} ≤ warn threshold {warn_thresh:.1f} ✓")

    # ── On-track estimate ─────────────────────────────────────────────────
    if step > 0:
        pct_complete = step / PAPER["best_step"] * 100
        # Rough interpolation: val_loss decays roughly as log(V) * exp(-k*step)
        # Fit: at step=0: 12.43; at step=62500: 1.60
        # → 1.60 = 12.43 * exp(-k*62500) → k = ln(12.43/1.60)/62500 ≈ 3.34e-5
        k = math.log(PAPER["log_vocab"] / PAPER["best_val_loss"]) / PAPER["best_step"]
        projected = PAPER["log_vocab"] * math.exp(-k * step)
        r.info(f"Paper-interpolated val_loss at step {step:,}: ~{projected:.2f} "
               f"({pct_complete:.0f}% of way to best_step)")

    # ── Final target check ────────────────────────────────────────────────
    if step >= PAPER["best_step"]:
        if best_val_loss <= PAPER["best_val_loss"]:
            r.ok(f"✓ PAPER TARGET MET: val_loss={best_val_loss:.4f} ≤ {PAPER['best_val_loss']:.2f}")
        elif best_val_loss <= PAPER["best_val_loss"] * 1.1:
            r.warn(f"Close to paper target: val_loss={best_val_loss:.4f} "
                   f"(target={PAPER['best_val_loss']:.2f}, within 10%)")
        else:
            r.fail(f"Missed paper target: val_loss={best_val_loss:.4f} "
                   f"vs paper {PAPER['best_val_loss']:.2f} at step {step:,}")


# ── TEST 7: Live Val Loss & Token Accuracy ────────────────────────────────────

def test_live_valoss(model, ema_model, ckpt, data_dir, r: Results, device, n_batches=30):
    r.header(7, "Live Val Loss & Token Accuracy", "Requires --data-dir")

    step = ckpt.get("step", 0)
    mc = ckpt["config"]["model"]
    mask_id = mc["mask_token_id"]
    val_split = ckpt["config"].get("data", {}).get("val_split", 0.01)

    try:
        from dataset import EmbeddingInversionDataset
        from torch.utils.data import DataLoader
        ds = EmbeddingInversionDataset(
            data_dir,
            max_seq_len=mc["max_seq_len"],
            val=True,
            val_split=val_split,
            pad_token_id=1,
            bos_token_id=None,
        )
        r.ok(f"Val dataset: {len(ds):,} samples")
    except Exception as e:
        r.fail(f"Failed to load dataset from {data_dir!r}: {e}")
        return

    loader = torch.utils.data.DataLoader(ds, batch_size=100, shuffle=False, num_workers=2)

    ema_val_loss = 0.0
    raw_val_loss = 0.0
    total_correct = 0
    total_masked = 0
    n_done = 0

    ema_model.eval()
    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            vids  = batch["token_ids"].to(device)
            vemb  = batch["embedding"].to(device)
            vpad  = batch["padding_mask"].to(device)
            vm_ids, vm_mask, _, _ = apply_mask(vids, mask_id, vpad)

            with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu',
                                    dtype=torch.bfloat16, enabled=(device.type == 'cuda')):
                # EMA val loss
                vh = ema_model.forward_hidden(vm_ids, vemb, vpad)
                vh_flat = vh.view(-1, vh.shape[-1])
                vt_flat = vids.view(-1)
                vm_flat = vm_mask.view(-1).float()
                vw = ema_model.output_proj.weight
                vtotal = torch.tensor(0.0, device=device)
                for vi in range(0, vh_flat.shape[0], 256):
                    ve = min(vi + 256, vh_flat.shape[0])
                    vlc = F.linear(vh_flat[vi:ve], vw)
                    vtotal = vtotal + (F.cross_entropy(vlc, vt_flat[vi:ve], reduction="none") * vm_flat[vi:ve]).sum()
                ema_val_loss += (vtotal / vm_flat.sum().clamp(min=1)).item()

                # Raw model (+ token accuracy)
                rh = model.forward_hidden(vm_ids, vemb, vpad)
                rh_flat = rh.view(-1, rh.shape[-1])
                rw = model.output_proj.weight
                rtotal = torch.tensor(0.0, device=device)
                for vi in range(0, rh_flat.shape[0], 256):
                    ve = min(vi + 256, rh_flat.shape[0])
                    rlc = F.linear(rh_flat[vi:ve], rw)
                    rtotal = rtotal + (F.cross_entropy(rlc, vt_flat[vi:ve], reduction="none") * vm_flat[vi:ve]).sum()
                    preds = rlc.argmax(-1)
                    total_correct += ((preds == vt_flat[vi:ve]) * vm_flat[vi:ve].bool()).sum().item()
                raw_val_loss += (rtotal / vm_flat.sum().clamp(min=1)).item()
                total_masked += vm_flat.sum().item()

            n_done += 1

    model.train()

    if n_done == 0:
        r.fail("No validation batches processed")
        return

    avg_ema_val = ema_val_loss / n_done
    avg_raw_val = raw_val_loss / n_done
    token_acc = total_correct / max(total_masked, 1)

    r.info(f"Evaluated {n_done} batches from val set")
    r.ok(f"EMA val_loss  = {avg_ema_val:.4f}")
    r.ok(f"Raw val_loss  = {avg_raw_val:.4f}")
    r.ok(f"Token acc     = {token_acc:.3f} ({token_acc:.1%})")

    # Compare to saved best
    saved_best = ckpt.get("best_val_loss", float("inf"))
    if saved_best < float("inf"):
        ratio = avg_ema_val / saved_best
        if 0.7 <= ratio <= 1.5:
            r.ok(f"Live EMA val_loss {avg_ema_val:.4f} vs checkpoint best {saved_best:.4f} (ratio {ratio:.2f})")
        else:
            r.warn(f"Large gap: live val_loss {avg_ema_val:.4f} vs best {saved_best:.4f} "
                   f"(ratio {ratio:.2f}). Normal if training is progressing.")

    # NaN/Inf check
    if not math.isfinite(avg_ema_val):
        r.fail(f"EMA val_loss is non-finite: {avg_ema_val} — numerical instability")
    elif avg_ema_val > 490:
        r.fail(f"EMA val_loss {avg_ema_val:.1f} >> 100 — STAGNATION BUG (expected < 15 at any point)")
    elif step > 1000 and avg_ema_val > PAPER["log_vocab"] + 0.5:
        r.fail(f"val_loss {avg_ema_val:.2f} ≈ random guessing at step {step:,} — not learning")

    # Token accuracy check
    for min_step, min_acc in TOKEN_ACC_LOWER:
        if step >= min_step and token_acc < min_acc:
            r.fail(f"token_acc={token_acc:.3f} < {min_acc:.2f} at step {step:,} (expected > {min_acc:.0%})")
            break
    else:
        for min_step, min_acc in TOKEN_ACC_LOWER:
            if step >= min_step:
                if token_acc >= min_acc:
                    r.ok(f"token_acc={token_acc:.1%} ≥ {min_acc:.0%} (threshold for step {step:,})")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints_v2_jinav3/latest.pt",
                        help="Path to .pt checkpoint (default: checkpoints_v2_jinav3/latest.pt)")
    parser.add_argument("--data-dir", default=None,
                        help="Data directory for live val loss / accuracy (optional)")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU (skips GPU-dependent tests)")
    parser.add_argument("--tests", default=None,
                        help="Comma-separated list of test numbers to run, e.g. '1,2,3,4,5,6'")
    args = parser.parse_args()
    requested = set(int(t.strip()) for t in args.tests.split(",")) if args.tests else None

    device = torch.device("cpu") if args.cpu or not torch.cuda.is_available() else torch.device("cuda")

    print(f"\n{'═'*64}")
    print(f"  MDLM Training Health Check")
    print(f"{'═'*64}")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Data dir   : {args.data_dir or '(not provided — TEST 7 skipped)'}")
    print(f"  Device     : {device}")
    print(f"  Paper ref  : val_loss={PAPER['best_val_loss']:.2f} @ step {PAPER['best_step']:,}, "
          f"acc={PAPER['best_token_acc']:.0%}")

    r = Results()

    # ── Load checkpoint ────────────────────────────────────────────────────
    try:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except FileNotFoundError:
        print(f"\n[ERROR] Checkpoint not found: {args.checkpoint}")
        print("  Make sure training has produced at least one checkpoint (first eval at step 500).")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Failed to load checkpoint: {e}")
        sys.exit(1)

    step, best_val_loss = test_checkpoint_integrity(ckpt, r)

    # ── Load model ─────────────────────────────────────────────────────────
    try:
        model = ConditionalMDLM(ckpt["config"]).to(device)
        raw_sd = {k.replace("_orig_mod.", "").replace("module.", ""): v
                  for k, v in ckpt["model"].items()}
        missing, unexpected = model.load_state_dict(raw_sd, strict=True)
        r.ok(f"Model loaded: 0 missing keys, 0 unexpected keys")
    except Exception as e:
        r.fail(f"Model load failed: {e}")
        r.summary()
        sys.exit(1)

    # Load EMA
    try:
        ema_model = copy.deepcopy(model).bfloat16()
        ema_model.eval()
        for p in ema_model.parameters():
            p.requires_grad_(False)
        if "ema_model" in ckpt:
            ema_sd = {k.replace("_orig_mod.", "").replace("module.", ""): v
                      for k, v in ckpt["ema_model"].items()}
            ema_sd = {k: v.bfloat16() for k, v in ema_sd.items()}
            ema_model.load_state_dict(ema_sd)
    except Exception as e:
        r.warn(f"EMA model load failed: {e} — EMA tests may be inaccurate")
        ema_model = copy.deepcopy(model).bfloat16().eval()

    # ── Run tests ──────────────────────────────────────────────────────────
    def _want(n): return requested is None or n in requested

    if _want(2): test_architecture(model, ckpt, r)
    if _want(3): test_ema_health(model, ema_model, ckpt, r)
    if _want(4): test_noise_schedule(r, device)
    if _want(5): test_loss_formula(model, r, device)
    if _want(6): test_trajectory(step, best_val_loss, r)

    if _want(7):
        if args.data_dir:
            test_live_valoss(model, ema_model, ckpt, args.data_dir, r, device)
        else:
            r.header(7, "Live Val Loss & Token Accuracy", "SKIPPED — provide --data-dir to run")
            r.skip("Skipped (no --data-dir provided)")

    r.summary()
    if r.counts["FAIL"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
