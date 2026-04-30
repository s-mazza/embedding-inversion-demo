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
