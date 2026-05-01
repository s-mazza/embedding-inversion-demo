# Technical Findings: Embedding Inversion Implementation Audit

This document summarizes the findings of a deep-dive investigation into the alignment between the current codebase, the paper ("Embedding Inversion via Conditional Masked Diffusion Language Models"), and the project's README.

## 1. Architectural Discrepancies

### 1.1 Missing Timestep Conditioning (v3/mmBERT)
The paper (Section 3.3, Eq. 6-9) specifies that each transformer layer must be modulated by both the embedding vector $c$ and the diffusion timestep $t$.
- **Bug:** The `v3` (mmBERT) path in `model.py` (`_forward_mmbert`) projects only the embedding $c$ and ignores $t$.
- **Impact:** The model has no knowledge of the noise level during denoising.
- **Proof:** `test_bugs.py` showed a max logit difference of **0.0** when changing mask fractions.

### 1.2 Position Insensitivity (v3/mmBERT)
ModernBERT uses Rotary Positional Embeddings (RoPE) to handle sequence order.
- **Bug:** The wrapper `ModernBertLayerWithAdaLN` calls the attention module with `position_embeddings=None`.
- **Impact:** The model is effectively a **Bag-of-Words** model. It cannot distinguish between positions in a sequence.
- **Proof:** `test_pos.py` confirmed that predictions for all masked positions are identical when input tokens are identical.

### 1.3 AdaLN vs AdaLN-Zero (v2)
The paper claims "All models use the same ... MDLM backbone with AdaLN-Zero conditioning."
- **Bug:** The `v2` (from-scratch) architecture uses standard AdaLN (scale/shift) but lacks the gating parameter $\alpha$ initialized to zero.
- **Impact:** The model does not benefit from the "identity initialization" property of DiT-style architectures, which is critical for stable fine-tuning of large backbones.

## 2. Critical Implementation Bugs

### 2.1 Broken EMA Mechanism (Still Broken)
The recent "fix" in `5fc15f6` (calculating `lerp_` in `fp32`) is insufficient because the result is immediately copied back into a `bf16` tensor.
- **Bug:** `1.0 + 0.0001 * (1.1 - 1.0) = 1.00001`. The smallest representable increment in `bf16` around 1.0 is $\approx 0.0078$.
- **Impact:** The `copy_` operation rounds the update back to the old value. **The EMA model remains frozen at its initial state.**
- **Proof:** `test_ema_v2.py` confirmed that after 100 steps, the `bf16` EMA weights were unchanged, while `fp32` weights updated correctly.

### 2.2 Loss Weighting Discrepancy (Gradient Attenuation)
The implementation in `train.py` uses the **mean** cross-entropy per sample, whereas the paper's Equation 4 specifies a **sum** over masked positions.
- **Bug:** The code calculates `loss = (Mean_CE / t)`. The paper specifies `loss = (Sum_CE / t)`.
- **Impact:** Since `Sum_CE ≈ n_masked * Mean_CE`, and $n_{masked}$ is proportional to $t$, the paper's loss weighting is balanced across all $t$. The code's weighting, however, effectively applies a $1/t$ penalty that **attenuates gradients for high-noise samples ($t \approx 1$) by a factor of 30x**.
- **Result:** The model over-fits to the trivial low-noise regime and fails to learn the complex global reconstruction required for inversion.

### 2.3 Biased Timestep Estimation
- **Bug:** `_t_from_input` calculates the mask fraction over the *entire* sequence length (32), not just the non-padding tokens.
- **Impact:** For a 10-token sentence with 22 pads, $t$ will be reported as ~0.06 even if 50% of the actual content is masked. This creates a massive training/inference mismatch.

## 3. v2-Specific Architectural Audit

### 3.1 Missing Post-Embedding LayerNorm
Standard Transformer architectures (BERT, GPT, RoBERTa) apply a LayerNorm immediately after summing token and position embeddings.
- **Bug:** The `v2` path (`_forward_scratch`) skips this normalization.
- **Impact:** Increases the risk of internal covariate shift and makes training more sensitive to initialization and learning rate spikes.

### 3.2 Inference-Time Domain Shift
- **Bug:** The model is trained on sequences containing padding tokens (`[content...][pad...]`), but `invert.py` generates full-length 32-token sequences without any padding.
- **Impact:** The model "hallucinates" content in positions that should be padding, leading to lower reconstruction quality as the decoder tokenizer struggles to make sense of the noise.

### 3.3 Missing AdaLN-Zero in Output Layer
While `TransformerBlock` now uses `AdaLNZero`, the final normalization before the output projection (`self.final_norm`) uses a standard `AdaLN` (scale/shift).
- **Impact:** The "identity initialization" property is not perfectly preserved, as the final norm can shift features away from their backbone values at step 0.

### 3.4 Weight Decay on Non-Decayable Parameters
- **Bug:** `train.py` passes all model parameters directly to `AdamW`.
- **Impact:** Biases and LayerNorm weights are subjected to weight decay, which can lead to parameter collapse and training instability.
- **Proof:** `eval_v2_deep_audit.py` (Test 4) identified 29 such parameters being incorrectly decayed.

### 3.5 Gradient Imbalance (Mean vs Sum CE)
- **Bug:** Current implementation uses `Mean_CE / t`.
- **Impact:** Gradient magnitude at $t=0.05$ is **20.4x larger** than at $t=0.95$.
- **Result:** The model almost entirely ignores high-noise samples, which are the most critical for global text reconstruction.
- **Proof:** `eval_v2_deep_audit.py` (Test 3) quantified this exact ratio.

### 3.6 Sequence Length Limitation
- **Bug:** Uses fixed absolute position embeddings.
- **Impact:** Model crashes on any input longer than `max_seq_len` (32).
- **Proof:** `eval_v2_deep_audit.py` (Test 12) confirmed an "index out of range" error on 64-token sequences.

## 4. Experimental Results Audit

Results from running `eval_diagnostic.py` on the cluster checkpoint (mmBERT/v3):

| Metric | Current Code (v3) | Paper Claim |
|--------|-------------------|-------------|
| Token Accuracy (100% mask) | **3.2%** | **10-30%** (mmBERT) |
| Token Accuracy (README) | **N/A** | **77.3%** (jina-v3) |
| Val Loss (Recomputed) | **7.57** | **2.68** |
| Position Sensitivity | **None** | **Full (RoPE)** |

## 5. Summary & Verdict
The current implementation fails to replicate the paper's results due to several critical flaws:
1.  **v3 path is position-insensitive** (broken RoPE).
2.  **EMA mechanism is numerically broken** (bf16 rounding).
3.  **Loss weighting is mathematically inconsistent with Equation 4**, leading to gradient attenuation in the most difficult training regime (high noise).
4.  **Significant architectural omissions** (missing conditioning in v3, missing LayerNorm in v2).

**Conclusion:** The codebase is currently in a "broken" state relative to the published paper. While recent commits have attempted to address some issues, the core diffusion logic and transformer implementations remain flawed.

## Recommended Fixes
1.  **v3 RoPE:** Correctly compute and pass `position_embeddings` (RoPE) to the ModernBert layers.
2.  **EMA:** Store `ema_model` in `fp32` to allow tiny updates to accumulate correctly.
3.  **Loss:** Change `Mean_CE` to `Sum_CE` in Equation 4 implementation to balance gradients across timesteps.
4.  **v3 Conditioning:** Implement `t_proj` and AdaLN-Zero in the mmBERT path to match the paper's conditioning requirements.

## 6. Gemini-Proposed Findings — Independent Verification

The following three issues were proposed by Gemini. Each was independently re-verified against the actual code and assigned a verdict. Fixes applied where warranted.

### 6.1 ✅ CONFIRMED + FIXED: Distributed validation is rank-0 shard only
- **Bug:** `val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, ...)` shards the val set across all ranks, but validation runs only on rank 0 (`if is_main`). Rank 0 therefore iterates over only `1/world_size` of val data.
- **Impact:** Best-checkpoint selection uses a biased, noisy estimate of val loss. At world_size=2: 50% of val data; at world_size=4: 25%.
- **Fix applied:** `val_sampler = None` in `dataset.py` (unconditionally). Validation on rank 0 now always iterates the full val set. Test added: `TestDistributedCorrectness::test_val_loader_never_uses_distributed_sampler`.

### 6.2 ❌ NOT A BUG (Gemini overcalled): Training/validation objective use different loss formulas
- **Claim:** Training uses 1/t-weighted sum-CE (MDLM Eq. 4); validation uses unweighted mean-CE — therefore checkpoint selection is inconsistent with training.
- **Verdict after review:** This is STANDARD ML PRACTICE. The 1/t weight is an optimization device to balance gradient contributions across noise levels (high-t samples would otherwise dominate). For evaluation you want a clean, interpretable number independent of the t-distribution. Unweighted mean-CE is the correct evaluation metric. The two objectives are highly correlated and val mean-CE is a perfectly valid proxy for training quality. No fix needed.

### 6.3 ✅ CONFIRMED + FIXED (low severity): `mixed_precision` config flag is non-functional
- **Bug:** All configs define `training.mixed_precision: true` but `train.py` never reads this flag; `GradScaler` and all `autocast` calls were hardcoded to BF16.
- **Impact:** Low severity — hardcoded BF16 is correct for A100/H100. But the flag is a dead config key; setting `mixed_precision: false` has no effect.
- **Fix applied:** `train.py` now reads `use_amp = tc.get("mixed_precision", True)` and passes `enabled=use_amp` to all `GradScaler` and `autocast` calls. Test added: `TestDistributedCorrectness::test_mixed_precision_flag_read_from_config`.

### 6.4 Verification Summary
- 6.1: CONFIRMED real bug, fixed in `dataset.py`
- 6.2: FALSE POSITIVE — standard practice, no fix needed
- 6.3: CONFIRMED dead config key, fixed in `train.py`
- Test suite: 64/64 passed after fixes

---

## 7. Audit Round 2 — 9 New Issues (All Confirmed + Fixed)

Deep audit targeting paper §3.3 Eq. 5–9 and training infrastructure. All 9 issues
are data-proven against the live code. Test suite: **88/88 passed** after fixes.

### 7.1 ✅ CRITICAL + FIXED: Conditioning architecture deviates from paper Eq. 6–9 (per-layer)

**Paper (Eq. 6–7):** Each transformer layer ℓ has *independent* `MLP_t^(ℓ)` and `MLP_c^(ℓ)`.
**Code (before fix):** A single shared `t_proj = Linear(1, 768)` and `cond_proj` was computed
*once* before the block loop and reused unchanged across all 8 layers.
**Impact:** Layers could not independently learn t vs. c sensitivity. The single `t_proj`
accumulated gradients from all 8 blocks simultaneously.
**Fix:** New `AdaLNZeroSplit` class with independent `c_proj` and `t_proj` per block, combined
additively inside the block. Shared `t_embed = Linear(1, 768)` maps scalar t → hidden vector;
per-layer projections from there. `TransformerBlock.forward` updated to `(x, cond_c, cond_t, ...)`.

### 7.2 ✅ HIGH + FIXED (auto): `t_proj.weight` in wrong optimizer param group

**Before:** Old shared `t_proj.weight` was in `decay_params` (no "adaln"/"norm"/"bias" match).
**After Issue 7.1 fix:** Per-layer t_proj lives inside `adaln1.t_proj`, so `"adaln"` substring
match correctly routes it to `no_decay_params`. Auto-fixed by the Issue 7.1 architectural change.

### 7.3 ✅ HIGH + FIXED: Missing post-embedding LayerNorm (re-confirms findings.md §3.1)

**Bug:** `_forward_scratch` computed `x = token_embed + pos_embed` with no normalization before
the first transformer block.
**Fix:** Added `self.embed_norm = nn.LayerNorm(hidden_dim)` applied as `embed_norm(embed + pos)`.
Applied in both `_forward_scratch` and `forward_hidden`.

### 7.4 ✅ HIGH + FIXED: GradScaler is a no-op with BF16 + DDP divergence risk

**Bug:** `GradScaler('cuda', enabled=use_amp)` — BF16 has fp32 dynamic range; no underflow risk.
The initial scale (65536) never changes. `unscale_` → scale → no-op round-trip.
**DDP risk:** `_check_inf_per_device()` runs per-rank without synchronization; asymmetric inf
detection causes parameter divergence.
**Fix:** `scaler = GradScaler('cuda', enabled=False)`. All scaler API calls become no-ops;
`autocast` still controlled by `use_amp` from config.

### 7.5 ✅ MEDIUM + FIXED: Epoch counter not saved/restored on checkpoint resume

**Bug:** `epoch` (used as seed for `DistributedSampler.set_epoch`) was not in the checkpoint
dict. Every resume restarted from `epoch=0`, repeating the same shuffle orderings.
**Fix:** `save_checkpoint` now accepts `epoch=0` kwarg and writes `"epoch": epoch` to the dict.
Resume code reads `epoch = ckpt.get("epoch", 0)`. All 5 call sites updated.

### 7.6 ✅ MEDIUM + FIXED: `final_norm` was AdaLN not AdaLNZero (re-confirms §3.3)

**Bug:** `self.final_norm = AdaLN(hidden_dim, hidden_dim)` — scale+shift only, no gate α.
At step 0, AdaLN applies `LayerNorm(x) × 1 + 0 = LayerNorm(x)` which is non-trivial even
though all upstream blocks are identity (α=0). Breaks "identity at init" framing.
**Fix:** Changed to `AdaLNZero(hidden_dim, hidden_dim)`. Forward paths updated to unpack
`(x_normed, _) = self.final_norm(x, cond_c + cond_t)`.

### 7.7 ✅ MEDIUM + FIXED: Paper metric (76% token accuracy) never logged during training

**Bug:** `val_loss` (unweighted mean-CE) was the only logged metric. Paper Table 1 reports
*token accuracy at 100% masking*. Without logging it, there was no way to know during
training if the model was on track to hit 76%.
**Fix:** Added 100%-mask token accuracy loop after each val evaluation (10 batches of EMA model).
Prints `token_acc (EMA, 100% mask): X.XXX  [paper target: 0.760 @ step 62500]`.

### 7.8 ✅ MEDIUM + FIXED: Val comment said "5000 samples", actual count is 10000

**Bug:** Line 408: `# 50 batches = 5000 samples (lower noise)` — with `batch_size=200`,
50 × 200 = 10,000, not 5,000. Stale from an earlier configuration.
**Fix:** Updated comment to `# 50 batches = 10000 samples (batch_size=200)`.

### 7.9 ✅ LOW + FIXED: Non-deterministic val masking → noisy checkpoint selection

**Bug:** `apply_mask` in the val loop samples fresh `t ~ U[0.02, 1]` with no fixed seed.
The same model evaluated twice gives slightly different `val_loss`. Noisy best-checkpoint
selection — runs with unlucky t draws appear worse.
**Fix:** Wrapped val loop with `with torch.no_grad(), torch.random.fork_rng():` +
`torch.manual_seed(42)`. Training RNG state is restored after val; val masking is
reproducible across all eval calls.

### 7.10 Summary
| # | Severity | Issue | File(s) | Status |
|---|----------|-------|---------|--------|
| 7.1 | CRITICAL | Per-layer t/c conditioning (paper Eq. 6-9) | model.py | ✅ FIXED |
| 7.2 | HIGH | t_proj.weight in decay_params | train.py | ✅ AUTO-FIXED |
| 7.3 | HIGH | Missing post-embedding LayerNorm | model.py | ✅ FIXED |
| 7.4 | HIGH | GradScaler no-op + DDP divergence risk | train.py | ✅ FIXED |
| 7.5 | MEDIUM | epoch not saved/restored on resume | train.py | ✅ FIXED |
| 7.6 | MEDIUM | final_norm is AdaLN not AdaLNZero | model.py | ✅ FIXED |
| 7.7 | MEDIUM | Paper metric (token acc 76%) never logged | train.py | ✅ FIXED |
| 7.8 | MEDIUM | Val comment "5000 samples" wrong by 2× | train.py | ✅ FIXED |
| 7.9 | LOW | Non-deterministic val masking | train.py | ✅ FIXED |

**Test suite: 88/88 passed** (`test_training_correctness.py` + `test_v2_audit2.py`)
