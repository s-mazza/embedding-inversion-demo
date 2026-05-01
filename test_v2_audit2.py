"""
Proof tests for audit round 2: Issues 1-9 identified in the plan.
Each test asserts the fix is applied against the live code/architecture.

Run with: pytest test_v2_audit2.py -v
"""
import copy
import inspect
import pytest
import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from model import ConditionalMDLM, AdaLNZero, AdaLNZeroSplit, TransformerBlock
import train


def _v2_config(hidden=64, layers=2):
    return {
        "model": {
            "vocab_size": 100,
            "hidden_dim": hidden,
            "max_seq_len": 8,
            "mask_token_id": 99,
            "num_heads": 4,
            "num_layers": layers,
            "ff_dim": 128,
            "embedding_cond_dim": 32,
            "tie_weights": False,
        }
    }


# ---------------------------------------------------------------------------
# Issue 1 — Per-layer t and c conditioning (paper Eq. 6-9)
# ---------------------------------------------------------------------------

class TestIssue1PerLayerConditioning:

    def test_adaln_zero_split_class_exists(self):
        """AdaLNZeroSplit must exist with independent c_proj and t_proj."""
        m = AdaLNZeroSplit(64)
        assert hasattr(m, 'c_proj'), "c_proj missing from AdaLNZeroSplit"
        assert hasattr(m, 't_proj'), "t_proj missing from AdaLNZeroSplit"
        assert m.c_proj.out_features == 3 * 64
        assert m.t_proj.out_features == 3 * 64

    def test_adaln_zero_split_zero_init(self):
        """Both c_proj and t_proj must be zero-initialized (identity at init)."""
        m = AdaLNZeroSplit(64)
        assert (m.c_proj.weight == 0).all()
        assert (m.c_proj.bias == 0).all()
        assert (m.t_proj.weight == 0).all()
        assert (m.t_proj.bias == 0).all()

    def test_transformer_block_uses_adaln_zero_split(self):
        """TransformerBlock.adaln1 and adaln2 must be AdaLNZeroSplit instances."""
        block = TransformerBlock(64, 4, 128)
        assert isinstance(block.adaln1, AdaLNZeroSplit), (
            f"adaln1 is {type(block.adaln1).__name__}, expected AdaLNZeroSplit"
        )
        assert isinstance(block.adaln2, AdaLNZeroSplit), (
            f"adaln2 is {type(block.adaln2).__name__}, expected AdaLNZeroSplit"
        )

    def test_model_has_t_embed_not_shared_t_proj(self):
        """Model must have t_embed (shared scalar→hidden), not old shared t_proj."""
        model = ConditionalMDLM(_v2_config())
        assert hasattr(model, 't_embed'), "t_embed must exist for shared t→hidden mapping"
        assert not hasattr(model, 't_proj'), "Old shared t_proj must be removed"
        assert isinstance(model.t_embed, nn.Linear)
        assert model.t_embed.in_features == 1
        assert model.t_embed.out_features == 64

    def test_per_layer_weights_are_independent(self):
        """Each block must have its own t_proj and c_proj (not shared tensors)."""
        model = ConditionalMDLM(_v2_config(layers=2))
        # Compare data pointers — must differ for independent parameters
        b0_t = model.blocks[0].adaln1.t_proj.weight.data_ptr()
        b1_t = model.blocks[1].adaln1.t_proj.weight.data_ptr()
        b0_c = model.blocks[0].adaln1.c_proj.weight.data_ptr()
        b1_c = model.blocks[1].adaln1.c_proj.weight.data_ptr()
        assert b0_t != b1_t, "Per-layer t_proj weights must be independent tensors"
        assert b0_c != b1_c, "Per-layer c_proj weights must be independent tensors"

    def test_t_conditioning_changes_output_when_weights_nonzero(self):
        """With non-zero t_proj weights, changing mask fraction changes output."""
        model = ConditionalMDLM(_v2_config())
        model.eval()
        # Simulate trained state: set t_proj and c_proj to non-zero
        with torch.no_grad():
            for block in model.blocks:
                block.adaln1.t_proj.weight.fill_(0.05)
                block.adaln2.t_proj.weight.fill_(0.05)

        B, L = 2, 8
        ids_low  = torch.zeros(B, L, dtype=torch.long)           # no masking → low t
        ids_high = torch.zeros(B, L, dtype=torch.long)
        ids_high[:, 4:] = 99                                       # 50% masked → higher t
        emb = torch.zeros(B, 32)

        with torch.no_grad():
            out_low  = model(ids_low, emb)
            out_high = model(ids_high, emb)

        diff = (out_low[:, :4] - out_high[:, :4]).abs().max().item()
        assert diff > 0, "t conditioning must affect output at non-masked positions"


# ---------------------------------------------------------------------------
# Issue 2 — t_proj weight decay grouping (auto-fixed by Issue 1)
# ---------------------------------------------------------------------------

class TestIssue2TprojWeightDecay:

    def test_adaln_per_layer_t_proj_in_no_decay(self):
        """Per-layer t_proj weights (inside adaln1/adaln2) must be in no_decay via 'adaln' match."""
        model = ConditionalMDLM(_v2_config())
        no_decay_names = [
            name for name, param in model.named_parameters()
            if param.requires_grad and ("bias" in name or "norm" in name or "adaln" in name)
        ]
        for name, _ in model.named_parameters():
            if 'adaln' in name and 't_proj.weight' in name:
                assert name in no_decay_names, (
                    f"{name} should be in no_decay group (matched by 'adaln' substring)"
                )

    def test_t_embed_in_decay(self):
        """t_embed.weight is a projection weight (not zero-init like adaln) and should be in decay."""
        model = ConditionalMDLM(_v2_config())
        decay_names = [
            name for name, param in model.named_parameters()
            if param.requires_grad and not ("bias" in name or "norm" in name or "adaln" in name)
        ]
        assert "t_embed.weight" in decay_names, (
            "t_embed.weight should be in decay_params (it's a trainable projection)"
        )


# ---------------------------------------------------------------------------
# Issue 3 — Post-embedding LayerNorm
# ---------------------------------------------------------------------------

class TestIssue3EmbedNorm:

    def test_embed_norm_exists(self):
        """Model must have self.embed_norm as a LayerNorm."""
        model = ConditionalMDLM(_v2_config())
        assert hasattr(model, 'embed_norm'), "embed_norm is missing"
        assert isinstance(model.embed_norm, nn.LayerNorm), (
            f"embed_norm is {type(model.embed_norm).__name__}, expected LayerNorm"
        )

    def test_embed_norm_applied_normalizes_input_to_blocks(self):
        """Input to first block must be LayerNorm-normalized (mean≈0 across features)."""
        model = ConditionalMDLM(_v2_config())
        model.eval()

        captured = {}
        def hook(module, inp, out):
            captured['x'] = inp[0].detach().clone()
        model.blocks[0].register_forward_hook(hook)

        ids = torch.randint(0, 98, (4, 8))
        emb = torch.randn(4, 32)
        with torch.no_grad():
            model(ids, emb)

        x = captured['x']  # [B, L, hidden]
        # LayerNorm is applied per-token across features → mean across feature dim ≈ 0
        mean_per_token = x.mean(dim=-1)
        assert mean_per_token.abs().max().item() < 0.1, (
            f"Input to first block not normalized (max mean={mean_per_token.abs().max():.3f}). "
            "embed_norm may not be applied."
        )


# ---------------------------------------------------------------------------
# Issue 4 — GradScaler disabled for BF16
# ---------------------------------------------------------------------------

class TestIssue4GradScalerDisabled:

    def test_scaler_always_disabled(self):
        """GradScaler must be created with enabled=False (BF16 needs no loss scaling)."""
        src = inspect.getsource(train.train)
        assert "GradScaler('cuda', enabled=False)" in src, (
            "GradScaler must use enabled=False for BF16 training. "
            "BF16 has fp32 dynamic range — loss scaling provides no benefit and "
            "introduces DDP asymmetric skip risk."
        )

    def test_scaler_not_enabled_by_use_amp(self):
        """GradScaler must not use enabled=use_amp (always disabled, not config-dependent)."""
        src = inspect.getsource(train.train)
        assert "GradScaler('cuda', enabled=use_amp)" not in src, (
            "GradScaler must not be conditionally enabled via use_amp for BF16."
        )


# ---------------------------------------------------------------------------
# Issue 5 — Epoch saved/restored in checkpoint
# ---------------------------------------------------------------------------

class TestIssue5EpochInCheckpoint:

    def test_save_checkpoint_has_epoch_param(self):
        """save_checkpoint must accept an epoch parameter."""
        sig = inspect.signature(train.save_checkpoint)
        assert 'epoch' in sig.parameters, (
            "save_checkpoint has no epoch parameter. "
            "epoch is needed to restore DistributedSampler shuffle seed on resume."
        )

    def test_save_checkpoint_writes_epoch_key(self):
        """save_checkpoint must include 'epoch' in the saved dict."""
        src = inspect.getsource(train.save_checkpoint)
        assert '"epoch"' in src or "'epoch'" in src, (
            "save_checkpoint does not write epoch to the checkpoint dict."
        )

    def test_resume_reads_epoch_from_checkpoint(self):
        """train() must restore epoch from checkpoint on resume."""
        src = inspect.getsource(train.train)
        assert 'ckpt.get("epoch"' in src or "ckpt.get('epoch'" in src, (
            "train() does not read epoch from checkpoint on resume. "
            "DistributedSampler.set_epoch() will always restart from 0."
        )


# ---------------------------------------------------------------------------
# Issue 6 — final_norm is AdaLNZero (not AdaLN)
# ---------------------------------------------------------------------------

class TestIssue6FinalNormIsAdaLNZero:

    def test_final_norm_is_adaln_zero(self):
        """final_norm must be AdaLNZero (scale + shift + alpha gate)."""
        model = ConditionalMDLM(_v2_config())
        assert isinstance(model.final_norm, AdaLNZero), (
            f"final_norm is {type(model.final_norm).__name__}, expected AdaLNZero. "
            "AdaLN (no gate) breaks identity-at-init in the output path."
        )

    def test_final_norm_proj_outputs_three_params(self):
        """AdaLNZero.proj must output 3*hidden (scale + shift + alpha)."""
        model = ConditionalMDLM(_v2_config())
        assert model.final_norm.proj.out_features == 3 * model.hidden_dim, (
            f"final_norm.proj outputs {model.final_norm.proj.out_features}, "
            f"expected {3 * model.hidden_dim} (3 * hidden_dim for scale/shift/alpha)."
        )

    def test_final_norm_proj_zero_at_init(self):
        """final_norm.proj must be zero-initialized (identity at init)."""
        model = ConditionalMDLM(_v2_config())
        assert (model.final_norm.proj.weight == 0).all()
        assert (model.final_norm.proj.bias == 0).all()


# ---------------------------------------------------------------------------
# Issue 7 — Paper metric (token accuracy at 100% masking) logged during training
# ---------------------------------------------------------------------------

class TestIssue7TokenAccuracyLogging:

    def test_token_acc_computed_in_val_loop(self):
        """train() must compute EMA token accuracy at 100% masking."""
        src = inspect.getsource(train.train)
        assert 'token_acc' in src, (
            "token_acc is not computed in train(). "
            "Paper Table 1 reports 76.0% token accuracy; this metric must be logged."
        )

    def test_full_mask_pattern_used(self):
        """Token accuracy must use 100% masking (vmask = ~vpad2)."""
        src = inspect.getsource(train.train)
        assert 'full_mask' in src or '~vpad' in src, (
            "100%-mask token accuracy must explicitly mask all content tokens."
        )

    def test_paper_target_documented_in_log(self):
        """Log message must reference the paper target (0.760 / 76%)."""
        src = inspect.getsource(train.train)
        assert '0.760' in src or '76' in src, (
            "Paper target (76.0% / 0.760) should be referenced in the log output."
        )


# ---------------------------------------------------------------------------
# Issue 8 — Val loop comment accuracy
# ---------------------------------------------------------------------------

class TestIssue8ValCommentAccurate:

    def test_val_comment_says_10000_not_5000(self):
        """Comment for val batches must reflect 50 × 200 = 10000 samples."""
        src = inspect.getsource(train.train)
        assert "10000" in src, (
            "Val loop comment must say 10000 samples (50 batches × batch_size=200)."
        )
        assert "5000 samples" not in src, (
            "Stale '5000 samples' comment must be removed (was wrong by 2×)."
        )


# ---------------------------------------------------------------------------
# Issue 9 — Deterministic val masking for stable checkpoint selection
# ---------------------------------------------------------------------------

class TestIssue9ValDeterministic:

    def test_val_uses_fork_rng(self):
        """Val masking must use fork_rng to not contaminate training RNG state."""
        src = inspect.getsource(train.train)
        assert 'fork_rng' in src, (
            "Val masking must use torch.random.fork_rng() so the training RNG "
            "is not affected by val evaluation."
        )

    def test_val_uses_fixed_seed(self):
        """Val masking must use a fixed manual_seed for reproducible checkpoint selection."""
        src = inspect.getsource(train.train)
        assert 'manual_seed' in src, (
            "Val masking must set a fixed manual_seed so that the same model "
            "evaluated twice gives the same val_loss."
        )
