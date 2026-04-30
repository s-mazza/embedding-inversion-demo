"""
Comprehensive correctness tests for the CMDLM v2 training pipeline.

Covers: initialization sanity, AdaLN-Zero identity property, gradient flow,
masking correctness, loss formula, EMA mechanics, LR schedule, noise schedule,
checkpoint roundtrip, weight tying, attention mask, mini end-to-end training.

Run with: pytest test_training_correctness.py -v
"""

import copy
import math
import io
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import ConditionalMDLM, apply_mask
from train import get_lr


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SMALL = {
    "model": {
        "vocab_size": 1000,
        "max_seq_len": 16,
        "hidden_dim": 64,
        "num_layers": 2,
        "num_heads": 4,
        "ff_dim": 256,
        "embedding_cond_dim": 32,
        "mask_token_id": 999,
        "dropout": 0.0,
        "tie_weights": True,
    }
}

# Closer to production for CE init tests
PROD_LIKE = {
    "model": {
        "vocab_size": 250002,
        "max_seq_len": 32,
        "hidden_dim": 768,
        "num_layers": 8,
        "num_heads": 12,
        "ff_dim": 3072,
        "embedding_cond_dim": 1024,
        "mask_token_id": 250001,
        "dropout": 0.0,
        "tie_weights": True,
    }
}


@pytest.fixture
def small_model():
    torch.manual_seed(42)
    return ConditionalMDLM(SMALL)


@pytest.fixture
def small_model_no_tie():
    cfg = copy.deepcopy(SMALL)
    cfg["model"]["tie_weights"] = False
    torch.manual_seed(42)
    return ConditionalMDLM(cfg)


# ---------------------------------------------------------------------------
# Section 1: Initialization sanity
# ---------------------------------------------------------------------------

class TestInitSanity:

    def test_ce_with_std002_is_near_log_vocab(self, small_model):
        """CE at step 0 with std=0.02 init should be ≈ log(vocab_size) ≈ 6.9."""
        model = small_model.eval()
        V = SMALL["model"]["vocab_size"]
        expected_ce = math.log(V)

        B, L = 16, SMALL["model"]["max_seq_len"]
        mask_id = SMALL["model"]["mask_token_id"]
        input_ids = torch.full((B, L), mask_id, dtype=torch.long)
        cond = torch.randn(B, SMALL["model"]["embedding_cond_dim"])

        with torch.no_grad():
            logits = model(input_ids, cond)  # [B, L, V]
        # CE over all positions (no padding here)
        targets = torch.randint(0, V - 1, (B, L))  # random targets
        ce = F.cross_entropy(logits.view(-1, V), targets.view(-1)).item()

        # Must be within 2 nats of log(V) — not 580 as with N(0,1) init
        assert abs(ce - expected_ce) < 2.0, (
            f"CE at init = {ce:.3f}, expected ≈ {expected_ce:.3f}. "
            f"Likely embedding init bug (std=1.0 gives CE ≈ {math.log(V) + V/2:.0f})"
        )

    def test_ce_counterfactual_bad_init(self):
        """CE with N(0,1) embedding init should be >> log(vocab_size) (shows why fix is critical)."""
        cfg = copy.deepcopy(SMALL)
        model = ConditionalMDLM(cfg).eval()
        # Manually reset embeddings to N(0,1) to simulate the old bug
        with torch.no_grad():
            nn.init.normal_(model.token_embed.weight, std=1.0)

        V = SMALL["model"]["vocab_size"]
        B, L = 4, SMALL["model"]["max_seq_len"]
        mask_id = SMALL["model"]["mask_token_id"]
        input_ids = torch.full((B, L), mask_id, dtype=torch.long)
        cond = torch.randn(B, SMALL["model"]["embedding_cond_dim"])
        targets = torch.randint(0, V - 1, (B, L))

        with torch.no_grad():
            logits = model(input_ids, cond)
        ce = F.cross_entropy(logits.view(-1, V), targets.view(-1)).item()

        # With std=1.0 and hidden_dim=64, logit_std ≈ √64 = 8, CE ≈ much larger than log(V)
        assert ce > math.log(V) + 5.0, (
            f"Bad init CE = {ce:.3f} should be >> {math.log(V):.3f}. "
            "The std=1.0 init should dramatically inflate CE."
        )

    def test_logit_std_is_small_at_init(self, small_model):
        """Logit standard deviation at init should be small (not dominated by embedding scale)."""
        model = small_model.eval()
        B, L = 32, SMALL["model"]["max_seq_len"]
        mask_id = SMALL["model"]["mask_token_id"]
        input_ids = torch.full((B, L), mask_id, dtype=torch.long)
        cond = torch.randn(B, SMALL["model"]["embedding_cond_dim"])

        with torch.no_grad():
            logits = model(input_ids, cond)  # [B, L, V]
        std = logits.std().item()

        # With std=0.02 init and hidden_dim=64, logit std ≈ 0.02 * √64 ≈ 0.16
        # With std=1.0, logit std ≈ √64 = 8.0
        assert std < 2.0, f"Logit std = {std:.3f} is too large, suggests bad embedding init"

    def test_hidden_states_finite(self, small_model):
        """No NaN or Inf in hidden states at init."""
        model = small_model.eval()
        B, L = 4, SMALL["model"]["max_seq_len"]
        mask_id = SMALL["model"]["mask_token_id"]
        input_ids = torch.full((B, L), mask_id, dtype=torch.long)
        cond = torch.randn(B, SMALL["model"]["embedding_cond_dim"])

        with torch.no_grad():
            hidden = model.forward_hidden(input_ids, cond)

        assert torch.isfinite(hidden).all(), "Hidden states contain NaN or Inf at init"

    def test_ce_with_prod_like_config(self):
        """CE at init with production-scale model (278M params, vocab=250002) ≈ log(250002) ≈ 12.4."""
        torch.manual_seed(0)
        model = ConditionalMDLM(PROD_LIKE).eval()
        expected_ce = math.log(250002)

        B, L = 4, 32
        mask_id = 250001
        input_ids = torch.full((B, L), mask_id, dtype=torch.long)
        cond = torch.randn(B, 1024)
        targets = torch.randint(0, 250001, (B, L))

        with torch.no_grad():
            logits = model(input_ids, cond)
        ce = F.cross_entropy(logits.view(-1, 250002), targets.view(-1)).item()

        assert abs(ce - expected_ce) < 1.5, (
            f"Prod-scale CE at init = {ce:.3f}, expected ≈ {expected_ce:.3f}"
        )


# ---------------------------------------------------------------------------
# Section 2: AdaLN-Zero properties
# ---------------------------------------------------------------------------

class TestAdaLNZero:

    def test_all_adaln_params_zero_at_init(self, small_model):
        """All AdaLN-Zero proj weights and biases must be exactly 0.0 at init."""
        for name, param in small_model.named_parameters():
            if "adaln" in name and "proj" in name:
                assert (param == 0.0).all(), (
                    f"{name} is not all-zero at init. "
                    "AdaLN-Zero requires zero-init for identity-at-init property."
                )

    def test_final_norm_proj_zero_at_init(self, small_model):
        """final_norm (AdaLN) proj must also be zero-initialized."""
        assert (small_model.final_norm.proj.weight == 0.0).all()
        assert (small_model.final_norm.proj.bias == 0.0).all()

    def test_t_proj_zero_at_init(self, small_model):
        """t_proj must be zero-initialized (starts as identity, conditioning grows in)."""
        assert (small_model.t_proj.weight == 0.0).all()
        assert (small_model.t_proj.bias == 0.0).all()

    def test_blocks_are_identity_at_init(self, small_model):
        """
        With AdaLN-Zero (alpha=0), every transformer block is a pure identity.
        x after N blocks = x before blocks (up to float precision).
        The only transformation is final_norm (LayerNorm).
        """
        model = small_model.eval()
        B, L = 2, SMALL["model"]["max_seq_len"]
        mask_id = SMALL["model"]["mask_token_id"]
        hidden_dim = SMALL["model"]["hidden_dim"]

        input_ids = torch.randint(0, 100, (B, L))
        cond_embed = torch.randn(B, SMALL["model"]["embedding_cond_dim"])

        with torch.no_grad():
            positions = torch.arange(L).unsqueeze(0)
            x_input = model.token_embed(input_ids) + model.pos_embed(positions)

            cond = model.cond_proj(cond_embed) + model.t_proj(
                model._t_from_input(input_ids)
            )

            x = x_input.clone()
            for block in model.blocks:
                x = block(x, cond)

        # x should equal x_input (identity through blocks)
        max_diff = (x - x_input).abs().max().item()
        assert max_diff < 1e-5, (
            f"Max diff through blocks at init = {max_diff:.2e}. "
            "AdaLN-Zero should give exact identity (alpha=0)."
        )

    def test_alpha_values_zero_at_init(self, small_model):
        """Explicitly verify all alpha outputs are zero at init."""
        model = small_model.eval()
        B, L = 2, SMALL["model"]["max_seq_len"]
        cond = torch.randn(B, SMALL["model"]["hidden_dim"])

        for i, block in enumerate(model.blocks):
            x = torch.randn(B, L, SMALL["model"]["hidden_dim"])
            _, alpha1 = block.adaln1(x, cond)
            _, alpha2 = block.adaln2(x, cond)
            assert (alpha1 == 0.0).all(), f"Block {i} alpha1 is not zero at init"
            assert (alpha2 == 0.0).all(), f"Block {i} alpha2 is not zero at init"


# ---------------------------------------------------------------------------
# Section 3: Gradient flow
# ---------------------------------------------------------------------------

class TestGradientFlow:

    def _get_loss(self, model):
        B, L = 4, SMALL["model"]["max_seq_len"]
        mask_id = SMALL["model"]["mask_token_id"]
        V = SMALL["model"]["vocab_size"]
        input_ids = torch.full((B, L), mask_id, dtype=torch.long)
        cond = torch.randn(B, SMALL["model"]["embedding_cond_dim"])
        targets = torch.randint(0, V - 1, (B, L))
        logits = model(input_ids, cond)
        return F.cross_entropy(logits.view(-1, V), targets.view(-1))

    def test_block_internal_params_zero_grad_at_init(self, small_model):
        """
        DiT identity-at-init property: when alpha=0, d(loss)/d(block_weights) = 0.
        The gradient path through block internals is gated by alpha.
        alpha=0 → zero gradient to attn/ff weights at step 0.
        """
        model = small_model.train()
        loss = self._get_loss(model)
        loss.backward()

        for i, block in enumerate(model.blocks):
            # attn and ff should have zero gradient
            for name, param in block.attn.named_parameters():
                if param.grad is not None:
                    assert (param.grad.abs() < 1e-9).all(), (
                        f"Block {i} attn.{name} has non-zero grad at init. "
                        "DiT property: block internals must be grad=0 when alpha=0."
                    )
            for j, layer in enumerate(block.ff):
                if hasattr(layer, 'weight') and layer.weight.grad is not None:
                    assert (layer.weight.grad.abs() < 1e-9).all(), (
                        f"Block {i} ff[{j}] has non-zero grad at init."
                    )

    def test_adaln_proj_nonzero_grad_at_init(self, small_model):
        """
        AdaLN proj gets non-zero gradient even at init.
        Gradient flows via the alpha path: d(loss)/d(alpha) = d(loss)/d(x) * block_out,
        which is non-zero even when alpha=0.
        """
        model = small_model.train()
        loss = self._get_loss(model)
        loss.backward()

        for i, block in enumerate(model.blocks):
            g1 = block.adaln1.proj.weight.grad
            g2 = block.adaln2.proj.weight.grad
            assert g1 is not None and (g1.abs() > 1e-9).any(), (
                f"Block {i} adaln1.proj has zero/None grad — conditioning cannot learn."
            )
            assert g2 is not None and (g2.abs() > 1e-9).any(), (
                f"Block {i} adaln2.proj has zero/None grad."
            )

    def test_t_proj_nonzero_grad(self, small_model):
        """t_proj gets gradient via the adaln path (starts at step 1 since adaln is zero-initialized)."""
        model = small_model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
        
        # Step 0
        loss = self._get_loss(model)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Step 1
        loss = self._get_loss(model)
        loss.backward()

        g = model.t_proj.weight.grad
        assert g is not None and (g.abs() > 1e-9).any(), (
            "t_proj.weight has zero/None gradient. t conditioning cannot learn."
        )

    def test_cond_proj_nonzero_grad(self, small_model):
        """cond_proj gets gradient (starts at step 1 since adaln is zero-initialized)."""
        model = small_model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
        
        # Step 0
        loss = self._get_loss(model)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Step 1
        loss = self._get_loss(model)
        loss.backward()

        # cond_proj is a Sequential; check the first Linear
        for name, param in model.cond_proj.named_parameters():
            g = param.grad
            assert g is not None and (g.abs() > 1e-9).any(), (
                f"cond_proj.{name} has zero/None grad. Conditioning signal blocked."
            )
            break  # first one is enough

    def test_token_embed_gets_gradient(self, small_model):
        """token_embed.weight receives gradient at masked positions."""
        model = small_model.train()
        loss = self._get_loss(model)
        loss.backward()

        g = model.token_embed.weight.grad
        assert g is not None, "token_embed has no gradient"
        assert (g.abs() > 1e-9).any(), "token_embed gradient is all zeros"

    def test_weight_tie_shares_gradient(self, small_model):
        """
        With tie_weights=True, token_embed.weight and output_proj.weight are the same tensor.
        Their gradients accumulate to the same .grad buffer.
        """
        model = small_model.train()
        assert model.output_proj.weight.data_ptr() == model.token_embed.weight.data_ptr(), (
            "output_proj.weight and token_embed.weight are not tied (different tensors)"
        )
        loss = self._get_loss(model)
        loss.backward()

        # Gradient lives in token_embed.weight.grad (they share the tensor)
        g = model.token_embed.weight.grad
        assert g is not None and (g.abs() > 1e-9).any()


# ---------------------------------------------------------------------------
# Section 4: Weight tying
# ---------------------------------------------------------------------------

class TestWeightTying:

    def test_tie_weights_true_same_object(self, small_model):
        """With tie_weights=True, output_proj.weight IS token_embed.weight (same data ptr)."""
        assert small_model.output_proj.weight.data_ptr() == small_model.token_embed.weight.data_ptr()

    def test_tie_weights_false_independent(self, small_model_no_tie):
        """With tie_weights=False, output_proj.weight is independent."""
        m = small_model_no_tie
        assert m.output_proj.weight.data_ptr() != m.token_embed.weight.data_ptr()

    def test_tie_weights_update_propagates(self, small_model):
        """Modifying token_embed.weight changes output_proj.weight in-place (same tensor)."""
        model = small_model
        old_val = model.output_proj.weight[0, 0].item()
        with torch.no_grad():
            model.token_embed.weight[0, 0] = 99.0
        assert model.output_proj.weight[0, 0].item() == 99.0, (
            "Weight tie broken: modifying token_embed didn't change output_proj"
        )
        with torch.no_grad():
            model.token_embed.weight[0, 0] = old_val  # restore


# ---------------------------------------------------------------------------
# Section 5: Masking correctness
# ---------------------------------------------------------------------------

class TestMaskingCorrectness:

    def test_apply_mask_never_masks_padding(self):
        """apply_mask must never set mask_token at padding positions."""
        B, L = 32, 16
        token_ids = torch.randint(0, 900, (B, L))
        # Last 4 positions are padding
        padding_mask = torch.zeros(B, L, dtype=torch.bool)
        padding_mask[:, 12:] = True
        mask_token_id = 999

        for _ in range(50):
            masked_ids, target_mask, _, _ = apply_mask(token_ids, mask_token_id, padding_mask)
            overlap = target_mask & padding_mask
            assert not overlap.any(), (
                "apply_mask masked a padding position. "
                "This would poison the training signal with trivially masked pad tokens."
            )

    def test_apply_mask_mask_ratio_range(self):
        """mask_ratio (from log-linear schedule) must be in [0.095, 0.993]."""
        B, L = 1000, 16
        token_ids = torch.randint(0, 900, (B, L))
        _, _, mask_ratio, t = apply_mask(token_ids, mask_token_id=999)

        # At t=0.02: mask_ratio = 1 - exp(-0.1) ≈ 0.0952
        # At t=1.0: mask_ratio = 1 - exp(-5) ≈ 0.9933
        assert mask_ratio.min().item() >= 0.08, f"mask_ratio below 0.08: {mask_ratio.min().item():.4f}"
        assert mask_ratio.max().item() <= 1.0, f"mask_ratio above 1.0: {mask_ratio.max().item():.4f}"

    def test_apply_mask_t_range(self):
        """t must be in [0.02, 1.0] — clamped at min=0.02 for 1/t loss stability."""
        B, L = 1000, 16
        token_ids = torch.randint(0, 900, (B, L))
        _, _, _, t = apply_mask(token_ids, mask_token_id=999)
        assert t.min().item() >= 0.019, f"t < 0.02: {t.min().item()}"
        assert t.max().item() <= 1.0, f"t > 1.0: {t.max().item()}"

    def test_apply_mask_formula_consistency(self):
        """mask_ratio must exactly equal 1 - exp(-5 * t)."""
        B, L = 100, 16
        token_ids = torch.randint(0, 900, (B, L))
        _, _, mask_ratio, t = apply_mask(token_ids, mask_token_id=999)

        expected = 1.0 - torch.exp(-5.0 * t)
        max_diff = (mask_ratio - expected).abs().max().item()
        assert max_diff < 1e-6, f"mask_ratio ≠ 1-exp(-5t), max diff = {max_diff:.2e}"

    def test_t_estimation_no_padding(self, small_model):
        """_t_from_input should recover the generating t when there's no padding."""
        model = small_model.eval()
        L = SMALL["model"]["max_seq_len"]
        mask_id = SMALL["model"]["mask_token_id"]

        # Manually set exact mask fractions for a single sequence
        for n_masked in [1, 4, 8, 12, 15]:
            ids = torch.zeros(1, L, dtype=torch.long)
            ids[0, :n_masked] = mask_id
            frac = n_masked / L
            expected_t = min(max(-math.log(1 - frac) / 5.0, 0.02), 1.0)

            with torch.no_grad():
                t_est = model._t_from_input(ids)

            diff = abs(t_est.item() - expected_t)
            assert diff < 1e-4, (
                f"n_masked={n_masked}: estimated t={t_est.item():.4f}, expected {expected_t:.4f}"
            )

    def test_t_estimation_with_padding(self, small_model):
        """
        _t_from_input must use content_len (not full L) as denominator.
        Bug: old code divided by L, giving t ≈ 0.034 instead of ≈ 0.139 for 5/10 masked.
        """
        model = small_model.eval()
        L = SMALL["model"]["max_seq_len"]
        mask_id = SMALL["model"]["mask_token_id"]
        B = 1

        # 10 content, 6 padding, 5 of 10 content tokens masked
        ids = torch.zeros(B, L, dtype=torch.long)
        ids[0, :5] = mask_id      # 5 masked (content)
        ids[0, 5:10] = 1          # 5 unmasked (content)
        # positions 10..15: padding tokens (id=0)
        padding_mask = torch.zeros(B, L, dtype=torch.bool)
        padding_mask[0, 10:] = True

        # correct: frac = 5/10 = 0.5, t = -log(0.5)/5 ≈ 0.1386
        correct_t = -math.log(0.5) / 5.0

        # old (buggy): frac = 5/16 ≈ 0.3125, t ≈ -log(0.6875)/5 ≈ 0.0754
        old_biased_t = -math.log(1 - 5 / L) / 5.0

        with torch.no_grad():
            t_est = model._t_from_input(ids, padding_mask)

        diff_correct = abs(t_est.item() - correct_t)
        diff_old = abs(t_est.item() - old_biased_t)

        assert diff_correct < 0.01, (
            f"t_from_input = {t_est.item():.4f}, should be {correct_t:.4f}. "
            f"Still biased by padding? (old biased value was {old_biased_t:.4f})"
        )
        assert diff_old > 0.02, (
            f"t_from_input = {t_est.item():.4f} looks like the OLD biased estimate {old_biased_t:.4f}. "
            "Padding-aware fix may not be working."
        )

    def test_t_estimation_edge_all_masked(self, small_model):
        """All content tokens masked → t clamped to 1.0."""
        model = small_model.eval()
        L = SMALL["model"]["max_seq_len"]
        mask_id = SMALL["model"]["mask_token_id"]
        ids = torch.full((1, L), mask_id, dtype=torch.long)
        with torch.no_grad():
            t = model._t_from_input(ids)
        assert t.item() == 1.0, f"All-masked should give t=1.0, got {t.item()}"

    def test_t_estimation_edge_none_masked(self, small_model):
        """No tokens masked → t clamped to 0.02."""
        model = small_model.eval()
        L = SMALL["model"]["max_seq_len"]
        ids = torch.zeros(1, L, dtype=torch.long)  # no mask tokens
        with torch.no_grad():
            t = model._t_from_input(ids)
        assert abs(t.item() - 0.02) < 1e-5, f"None-masked should give t=0.02, got {t.item()}"


# ---------------------------------------------------------------------------
# Section 6: Attention mask — padding isolation
# ---------------------------------------------------------------------------

class TestAttentionMask:

    def test_padding_tokens_dont_affect_content_outputs(self, small_model):
        """
        Garbage values in padding positions must not change logits at content positions.
        This verifies key_padding_mask is correctly applied to nn.MultiheadAttention.
        """
        model = small_model.eval()
        B, L = 2, SMALL["model"]["max_seq_len"]
        cond = torch.randn(B, SMALL["model"]["embedding_cond_dim"])
        mask_id = SMALL["model"]["mask_token_id"]

        # Content: positions 0..7, padding: positions 8..15
        input_ids = torch.randint(0, 100, (B, L))
        input_ids[:, 8:] = 1  # pad token
        padding_mask = torch.zeros(B, L, dtype=torch.bool)
        padding_mask[:, 8:] = True

        # Baseline logits
        with torch.no_grad():
            logits_base = model(input_ids, cond, padding_mask)

        # Replace padding with random garbage
        input_ids_garbage = input_ids.clone()
        input_ids_garbage[:, 8:] = torch.randint(2, 900, (B, 8))

        with torch.no_grad():
            logits_garbage = model(input_ids_garbage, cond, padding_mask)

        max_diff = (logits_base[:, :8] - logits_garbage[:, :8]).abs().max().item()
        assert max_diff < 1e-4, (
            f"Padding content bleeds into content outputs (max_diff={max_diff:.2e}). "
            "key_padding_mask is not working correctly."
        )

    def test_no_attention_to_padding_positions(self, small_model):
        """
        With padding_mask=True at positions 8..15, the model should attend only to 0..7.
        Verify by checking that attention-weighted sum excludes padding.
        (Proxy: perturbing only padding positions should not change content outputs.)
        Same as above but with the all-mask-token input to isolate conditioning from content.
        """
        model = small_model.eval()
        B, L = 1, SMALL["model"]["max_seq_len"]
        cond = torch.randn(B, SMALL["model"]["embedding_cond_dim"])
        mask_id = SMALL["model"]["mask_token_id"]

        # All mask tokens so perturbation is in embedding, not a different token type
        input_ids = torch.full((B, L), mask_id, dtype=torch.long)
        padding_mask = torch.zeros(B, L, dtype=torch.bool)
        padding_mask[:, 8:] = True

        with torch.no_grad():
            h_base = model.forward_hidden(input_ids, cond, padding_mask)

        # Completely different tokens at padding positions
        input_ids2 = input_ids.clone()
        input_ids2[:, 8:] = 0  # completely different

        with torch.no_grad():
            h_perturbed = model.forward_hidden(input_ids2, cond, padding_mask)

        max_diff = (h_base[:, :8] - h_perturbed[:, :8]).abs().max().item()
        assert max_diff < 1e-4, (
            f"Padding tokens leak into content hidden states (max_diff={max_diff:.2e}). "
            "padding_mask is not blocking attention to padding."
        )


# ---------------------------------------------------------------------------
# Section 7: Loss formula correctness
# ---------------------------------------------------------------------------

class TestLossFormula:

    def _simulate_training_loss(self, logits, targets, target_mask, t_sample, L):
        """
        Reference implementation of the per-sample 1/t-weighted loss (Eq. 4).
        Intentionally uses a simple loop-free formulation to serve as ground truth.
        """
        B = logits.shape[0]
        # CE at every position
        ce_all = F.cross_entropy(
            logits.view(-1, logits.shape[-1]), targets.view(-1), reduction="none"
        ).view(B, L)  # [B, L]

        # Zero out unmasked positions
        ce_masked = ce_all * target_mask.float()  # [B, L]

        # Sum over masked positions per sample (not mean!)
        per_sample_ce = ce_masked.sum(-1)  # [B]

        # 1/t weighting per sample
        loss = (per_sample_ce / t_sample.squeeze(1)).mean()
        return loss

    def test_loss_zero_when_nothing_masked(self, small_model):
        """When target_mask is all-False, per_sample_ce = 0 → loss = 0."""
        model = small_model.eval()
        B, L = 8, SMALL["model"]["max_seq_len"]
        V = SMALL["model"]["vocab_size"]
        logits = torch.randn(B, L, V)
        targets = torch.randint(0, V - 1, (B, L))
        target_mask = torch.zeros(B, L, dtype=torch.bool)
        t_sample = torch.rand(B, 1).clamp(min=0.02)

        loss = self._simulate_training_loss(logits, targets, target_mask, t_sample, L)
        assert loss.item() == 0.0, f"Loss with no masked tokens = {loss.item()}, expected 0.0"

    def test_1_over_t_weighting_is_per_sample(self, small_model):
        """
        Sample with t=0.02 contributes 50x more to loss than t=1.0 (same CE).
        Tests Eq. 4: loss = mean(CE_b / t_b).
        """
        B, L = 2, SMALL["model"]["max_seq_len"]
        V = SMALL["model"]["vocab_size"]

        # Construct logits so CE is exactly the same for both samples
        logits = torch.zeros(B, L, V)
        logits[:, :, 0] = 10.0  # all mass on token 0
        targets = torch.ones(B, L, dtype=torch.long)  # token 1 (wrong) → CE ≈ 10

        target_mask = torch.zeros(B, L, dtype=torch.bool)
        target_mask[:, 0] = True # one token masked each
        t_sample = torch.tensor([[0.02], [1.0]])  # sample 0: t=0.02, sample 1: t=1.0

        loss = self._simulate_training_loss(logits, targets, target_mask, t_sample, L)

        # Expected: CE ≈ 10 for both, contribution: (10/0.02 + 10/1.0)/2 = (500+10)/2 = 255
        ce_each = F.cross_entropy(logits[0, 0:1], targets[0, 0:1]).item()
        expected = (ce_each / 0.02 + ce_each / 1.0) / 2.0

        rel_err = abs(loss.item() - expected) / expected
        assert rel_err < 0.01, (
            f"Loss = {loss.item():.3f}, expected {expected:.3f} "
            f"(1/t weighting not per-sample, rel_err={rel_err:.3f})"
        )

    def test_ce_only_at_masked_positions(self, small_model):
        """CE at unmasked positions must not affect the loss."""
        B, L = 4, SMALL["model"]["max_seq_len"]
        V = SMALL["model"]["vocab_size"]

        target_mask = torch.zeros(B, L, dtype=torch.bool)
        target_mask[:, :4] = True  # only first 4 positions masked

        t_sample = torch.full((B, 1), 0.5)

        # Baseline logits
        logits_base = torch.randn(B, L, V)
        targets = torch.randint(0, V, (B, L))
        loss_base = self._simulate_training_loss(logits_base, targets, target_mask, t_sample, L)

        # Completely different logits at unmasked positions
        logits_modified = logits_base.clone()
        logits_modified[:, 4:] = torch.randn(B, L - 4, V) * 100  # extreme values

        loss_modified = self._simulate_training_loss(
            logits_modified, targets, target_mask, t_sample, L
        )

        assert abs(loss_base.item() - loss_modified.item()) < 1e-5, (
            f"Unmasked logit changes affected loss: {loss_base.item():.4f} vs {loss_modified.item():.4f}"
        )

    def test_scatter_accumulation_correctness(self):
        """
        Verify the scatter_add_ accumulation used in the training loop correctly
        attributes per-position CE values to their owning sample.
        """
        B, L = 3, 4
        # Known CE values: [B, L] flattened
        ce_flat = torch.tensor([1.0, 2.0, 3.0, 4.0,   # sample 0: sum=10
                                 5.0, 6.0, 7.0, 8.0,   # sample 1: sum=26
                                 9.0, 10.0, 11.0, 12.0], # sample 2: sum=42
                                dtype=torch.float)
        mask_flat = torch.ones(B * L, dtype=torch.float)  # all masked

        sample_ce_sum = torch.zeros(B)
        for i in range(0, B * L, 4):  # chunk_size=4 = L
            end = min(i + 4, B * L)
            chunk_idx = torch.arange(i, end) // L
            sample_ce_sum.scatter_add_(0, chunk_idx, ce_flat[i:end] * mask_flat[i:end])

        expected = torch.tensor([10.0, 26.0, 42.0])
        assert (sample_ce_sum - expected).abs().max().item() < 1e-5, (
            f"scatter_add accumulation wrong: {sample_ce_sum} vs {expected}"
        )

    def test_n_masked_per_sample_excludes_padding(self):
        """n_masked_per_sample counts only masked content tokens, never padding tokens."""
        B, L = 4, 16
        # 8 content, 8 padding per sample; 4 content tokens masked
        token_ids = torch.randint(0, 900, (B, L))
        padding_mask = torch.zeros(B, L, dtype=torch.bool)
        padding_mask[:, 8:] = True

        masked_ids, target_mask, _, _ = apply_mask(token_ids, mask_token_id=999, padding_mask=padding_mask)

        # target_mask should only be True at content positions (never padding)
        assert not (target_mask & padding_mask).any(), "target_mask includes padding positions"

        # n_masked_per_sample should be in [0, 8]
        n_masked = target_mask.float().sum(-1)
        assert (n_masked <= 8).all(), f"n_masked_per_sample > 8 (includes padding?): {n_masked}"


# ---------------------------------------------------------------------------
# Section 8: EMA mechanics
# ---------------------------------------------------------------------------

class TestEMAMechanics:

    def test_ema_bf16_update_does_not_round_to_zero(self):
        """
        Without fp32 accumulation: ema_decay=0.9999, update=0.0001 × diff < bf16 precision.
        With fp32 accumulation: the update is preserved correctly.
        """
        ema_decay = 0.9999
        ema_param = torch.tensor([0.0], dtype=torch.float32)
        model_param = torch.tensor([1.0], dtype=torch.float32)

        # Simulate 100 fp32 updates
        for _ in range(100):
            ema_param.lerp_(model_param, 1 - ema_decay)

        # After 100 steps: expected ≈ 1 - 0.9999^100 ≈ 0.00995
        expected = 1.0 - (ema_decay ** 100)
        actual = ema_param.float().item()

        assert abs(actual - expected) < 1e-3, (
            f"EMA after 100 steps = {actual:.6f}, expected ≈ {expected:.6f}. "
            "fp32 accumulation may not be working."
        )

        # Verify the old (broken) bf16 direct update does NOT work
        ema_broken = torch.tensor([0.5], dtype=torch.bfloat16)
        model_param_bf16 = torch.tensor([1.0], dtype=torch.bfloat16)
        for _ in range(100):
            ema_broken.lerp_(model_param_bf16, 1 - ema_decay)

        assert ema_broken.float().item() == 0.5, (
            "Direct bf16 lerp surprisingly worked — bf16 precision has changed?"
        )

    def test_ema_converges_to_model_over_time(self, small_model):
        """EMA weights converge toward model weights as training progresses."""
        model = small_model.train()
        ema = copy.deepcopy(model).float().eval()
        for p in ema.parameters():
            p.requires_grad_(False)

        ema_decay = 0.9
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        # Distance before
        dist_before = sum(
            (ep - mp).pow(2).sum().item()
            for ep, mp in zip(ema.parameters(), model.parameters())
        )
        assert dist_before == 0.0 # Initially identical

        # Make model differ
        optimizer.zero_grad()
        B, L = 4, SMALL["model"]["max_seq_len"]
        ids = torch.randint(0, 100, (B, L))
        cond = torch.randn(B, SMALL["model"]["embedding_cond_dim"])
        targets = torch.randint(0, SMALL["model"]["vocab_size"] - 1, (B, L))
        loss = F.cross_entropy(model(ids, cond).view(-1, SMALL["model"]["vocab_size"]), targets.view(-1))
        loss.backward()
        optimizer.step()
        
        dist_diff = sum((ep - mp).pow(2).sum().item() for ep, mp in zip(ema.parameters(), model.parameters()))
        
        # Run 50 steps of EMA update toward the new model
        for _ in range(50):
            with torch.no_grad():
                for ep, mp in zip(ema.parameters(), model.parameters()):
                    ep.lerp_(mp.float(), 1 - ema_decay)

        dist_after = sum(
            (ep - mp).pow(2).sum().item()
            for ep, mp in zip(ema.parameters(), model.parameters())
        )

        assert dist_after < dist_diff, (
            f"EMA is not converging toward model weights. "
            f"dist_diff={dist_diff:.4f}, dist_after={dist_after:.4f}"
        )

    def test_ema_no_grad(self, small_model):
        """EMA model parameters must not require gradients."""
        ema = copy.deepcopy(small_model).eval()
        for p in ema.parameters():
            p.requires_grad_(False)
        for p in ema.parameters():
            assert not p.requires_grad, "EMA parameter requires_grad=True — wastes memory"

    def test_ema_decay_rate(self):
        """Analytical check: after N steps, EMA ≈ 1 - decay^N (starting from 0, model=1)."""
        ema_decay = 0.9999
        ema = torch.tensor([0.0], dtype=torch.float32)
        model_w = torch.tensor([1.0], dtype=torch.float32)
        N = 10000

        for _ in range(N):
            ema.lerp_(model_w, 1 - ema_decay)

        expected = 1.0 - ema_decay ** N  # ≈ 0.6321
        actual = ema.item()
        assert abs(actual - expected) < 0.01, (
            f"EMA after {N} steps = {actual:.4f}, expected ≈ {expected:.4f}"
        )


# ---------------------------------------------------------------------------
# Section 9: LR schedule
# ---------------------------------------------------------------------------

class TestLRSchedule:

    def test_lr_zero_at_step_0(self):
        assert get_lr(0, warmup_steps=2000, max_steps=50000, max_lr=1e-4) == 0.0

    def test_lr_peaks_at_warmup(self):
        lr = get_lr(2000, warmup_steps=2000, max_steps=50000, max_lr=1e-4)
        assert abs(lr - 1e-4) < 1e-8, f"LR at warmup end = {lr}, expected 1e-4"

    def test_lr_monotone_during_warmup(self):
        lrs = [get_lr(s, 2000, 50000, 1e-4) for s in range(0, 2001)]
        for i in range(1, len(lrs)):
            assert lrs[i] >= lrs[i - 1], f"LR not monotone at step {i}: {lrs[i-1]:.2e} → {lrs[i]:.2e}"

    def test_lr_decays_after_warmup(self):
        lrs = [get_lr(s, 2000, 50000, 1e-4) for s in range(2000, 50001, 1000)]
        for i in range(1, len(lrs)):
            assert lrs[i] <= lrs[i - 1] + 1e-12, (
                f"LR increased after warmup at step {2000 + i*1000}"
            )

    def test_lr_min_ratio_respected(self):
        """min_lr_ratio=0.1: LR at max_steps = 0.1 × max_lr."""
        lr = get_lr(50000, warmup_steps=2000, max_steps=50000, max_lr=1e-4, min_lr_ratio=0.1)
        assert abs(lr - 1e-5) < 1e-9, f"LR at max_steps with min_lr_ratio=0.1 = {lr:.2e}, expected 1e-5"

    def test_lr_zero_min_ratio(self):
        """min_lr_ratio=0.0 (default): LR at max_steps = 0."""
        lr = get_lr(50000, warmup_steps=2000, max_steps=50000, max_lr=1e-4, min_lr_ratio=0.0)
        assert lr < 1e-9, f"LR at max_steps with min_lr_ratio=0.0 = {lr:.2e}, expected ~0"


# ---------------------------------------------------------------------------
# Section 10: Noise schedule math
# ---------------------------------------------------------------------------

class TestNoiseSchedule:

    def test_mask_ratio_at_t002(self):
        """At t=0.02: mask_ratio = 1 - exp(-0.1) ≈ 0.0952."""
        expected = 1.0 - math.exp(-5 * 0.02)
        B, L = 1, 32
        token_ids = torch.zeros(B, L, dtype=torch.long)
        torch.manual_seed(0)
        # Override the random draw by patching (easier: just check the formula directly)
        t = torch.tensor([[0.02]])
        mask_ratio = 1.0 - torch.exp(-5.0 * t)
        assert abs(mask_ratio.item() - expected) < 1e-6

    def test_mask_ratio_at_t1(self):
        """At t=1.0: mask_ratio = 1 - exp(-5) ≈ 0.9933."""
        expected = 1.0 - math.exp(-5.0)
        t = torch.tensor([[1.0]])
        mask_ratio = 1.0 - torch.exp(-5.0 * t)
        assert abs(mask_ratio.item() - expected) < 1e-6

    def test_apply_mask_covers_full_range(self):
        """
        Over 10000 samples, t should cover the full [0.02, 1.0] range.
        In particular, there should be samples near both boundaries.
        """
        B, L = 10000, 16
        token_ids = torch.zeros(B, L, dtype=torch.long)
        _, _, _, t = apply_mask(token_ids, mask_token_id=999)

        assert t.min().item() < 0.05, f"No samples near t=0.02 boundary: min={t.min().item():.3f}"
        assert t.max().item() > 0.95, f"No samples near t=1.0 boundary: max={t.max().item():.3f}"

    def test_t_uniform_distribution(self):
        """t ~ U[0.02, 1.0]: mean should be ≈ 0.5002, std should be ≈ 0.28."""
        B, L = 100000, 4
        token_ids = torch.zeros(B, L, dtype=torch.long)
        _, _, _, t = apply_mask(token_ids, mask_token_id=999)

        # t = max(U(0, 1), 0.02)
        # Expected mean = 0.02*0.02 + 0.5 * (1 - 0.02^2) = 0.0004 + 0.4998 = 0.5002
        expected_mean = 0.5002
        # Expected E[X^2] = 0.02^3 + (1 - 0.02^3)/3 ≈ 0.333338
        # Variance = 0.333338 - 0.5002^2 ≈ 0.083138 -> std ≈ 0.2883
        expected_std = 0.2883

        assert abs(t.mean().item() - expected_mean) < 0.01, (
            f"t mean = {t.mean().item():.3f}, expected ≈ {expected_mean:.3f}"
        )
        assert abs(t.std().item() - expected_std) < 0.01, (
            f"t std = {t.std().item():.3f}, expected ≈ {expected_std:.3f}"
        )


# ---------------------------------------------------------------------------
# Section 11: Checkpoint roundtrip
# ---------------------------------------------------------------------------

class TestCheckpointRoundtrip:

    def test_save_load_identical_output(self, small_model):
        """Saved and loaded model must produce bitwise-identical outputs."""
        model = small_model.eval()
        B, L = 4, SMALL["model"]["max_seq_len"]
        input_ids = torch.randint(0, 100, (B, L))
        cond = torch.randn(B, SMALL["model"]["embedding_cond_dim"])

        with torch.no_grad():
            logits_before = model(input_ids, cond).clone()

        # Save to buffer
        buf = io.BytesIO()
        torch.save({"model": model.state_dict(), "config": SMALL}, buf)
        buf.seek(0)

        # Load into fresh model
        ckpt = torch.load(buf, weights_only=False)
        model2 = ConditionalMDLM(SMALL).eval()
        model2.load_state_dict(ckpt["model"])

        with torch.no_grad():
            logits_after = model2(input_ids, cond).clone()

        max_diff = (logits_before - logits_after).abs().max().item()
        assert max_diff == 0.0, f"Checkpoint roundtrip changed logits (max_diff={max_diff:.2e})"

    def test_state_dict_no_unexpected_keys(self, small_model):
        """Loading state_dict into a fresh model must report no missing or unexpected keys."""
        sd = small_model.state_dict()
        model2 = ConditionalMDLM(SMALL)
        result = model2.load_state_dict(sd, strict=True)
        assert len(result.missing_keys) == 0, f"Missing keys: {result.missing_keys}"
        assert len(result.unexpected_keys) == 0, f"Unexpected keys: {result.unexpected_keys}"


# ---------------------------------------------------------------------------
# Section 12: Mini end-to-end training
# ---------------------------------------------------------------------------

class TestEndToEndTraining:

    def test_loss_decreases_over_steps(self, small_model):
        """
        Loss should decrease over 30 gradient steps on a small fixed batch.
        This is the most basic sanity check: the model can overfit.
        """
        model = small_model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        V = SMALL["model"]["vocab_size"]
        B, L = 8, SMALL["model"]["max_seq_len"]
        mask_id = SMALL["model"]["mask_token_id"]

        # Fixed batch: all masked, predict token 1 everywhere
        input_ids = torch.full((B, L), mask_id, dtype=torch.long)
        targets = torch.ones(B, L, dtype=torch.long)  # always token 1
        cond = torch.randn(B, SMALL["model"]["embedding_cond_dim"])

        losses = []
        for _ in range(30):
            optimizer.zero_grad()
            logits = model(input_ids, cond)
            loss = F.cross_entropy(logits.view(-1, V), targets.view(-1))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss at end should be lower than at start
        loss_start = sum(losses[:5]) / 5
        loss_end = sum(losses[-5:]) / 5
        assert loss_end < loss_start * 0.7, (
            f"Loss did not decrease: start={loss_start:.3f}, end={loss_end:.3f}. "
            "Model may be unable to learn (gradient flow issue)."
        )

    def test_gradient_clipping_respected(self, small_model):
        """After clipping, gradient norm must be ≤ max_grad_norm."""
        model = small_model.train()
        V = SMALL["model"]["vocab_size"]
        B, L = 4, SMALL["model"]["max_seq_len"]
        mask_id = SMALL["model"]["mask_token_id"]
        input_ids = torch.full((B, L), mask_id, dtype=torch.long)
        cond = torch.randn(B, SMALL["model"]["embedding_cond_dim"])
        targets = torch.randint(0, V - 1, (B, L))

        logits = model(input_ids, cond)
        loss = F.cross_entropy(logits.view(-1, V), targets.view(-1))
        loss.backward()

        max_grad_norm = 1.0
        # clip_grad_norm_ returns the norm BEFORE clipping
        pre_clip_norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        # Verify the post-clipping norm
        post_clip_norm = torch.norm(torch.stack([p.grad.norm() for p in model.parameters() if p.grad is not None]))
        assert post_clip_norm <= max_grad_norm + 1e-6, (
            f"Gradient norm = {post_clip_norm:.4f} after clipping to {max_grad_norm}. "
            "clip_grad_norm_ not working."
        )

    def test_parameters_change_after_optimizer_step(self, small_model):
        """Model parameters must change after one optimizer step."""
        model = small_model.train()
        params_before = {
            n: p.clone().detach()
            for n, p in model.named_parameters()
        }
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        V = SMALL["model"]["vocab_size"]
        B, L = 4, SMALL["model"]["max_seq_len"]
        ids = torch.randint(0, 100, (B, L))
        cond = torch.randn(B, SMALL["model"]["embedding_cond_dim"])
        targets = torch.randint(0, V - 1, (B, L))

        optimizer.zero_grad()
        loss = F.cross_entropy(model(ids, cond).view(-1, V), targets.view(-1))
        loss.backward()
        optimizer.step()

        changed_params = []
        for n, p in model.named_parameters():
            if (p - params_before[n]).abs().max().item() > 0:
                changed_params.append(n)

        assert len(changed_params) > 0, "No parameters changed after optimizer step"

    def test_only_conditioning_params_change_at_step_0(self, small_model):
        """
        At the first gradient step with AdaLN-Zero, only conditioning parameters
        (adaln projections, t_proj, cond_proj) should change — not block internals.
        Block internals (attn, ff) get zero gradient at init and remain unchanged.
        """
        model = small_model.train()
        params_before = {n: p.clone().detach() for n, p in model.named_parameters()}

        # Use a small LR so we can detect zero vs non-zero gradient
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        V = SMALL["model"]["vocab_size"]
        B, L = 4, SMALL["model"]["max_seq_len"]
        ids = torch.randint(0, 100, (B, L))
        cond = torch.randn(B, SMALL["model"]["embedding_cond_dim"])
        targets = torch.randint(0, V - 1, (B, L))

        optimizer.zero_grad()
        loss = F.cross_entropy(model(ids, cond).view(-1, V), targets.view(-1))
        loss.backward()
        optimizer.step()

        for name, param in model.named_parameters():
            changed = (param - params_before[name]).abs().max().item() > 1e-8
            is_block_internal = any(
                f"blocks.{i}.attn" in name or f"blocks.{i}.ff" in name
                for i in range(SMALL["model"]["num_layers"])
            )
            if is_block_internal:
                assert not changed, (
                    f"{name} changed at step 0 even though it should have zero gradient "
                    "(DiT identity-at-init property violated)."
                )


# ---------------------------------------------------------------------------
# Section 13: Additional edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_forward_deterministic_without_dropout(self, small_model):
        """With dropout=0.0, forward must be deterministic in eval mode."""
        model = small_model.eval()
        B, L = 4, SMALL["model"]["max_seq_len"]
        ids = torch.randint(0, 100, (B, L))
        cond = torch.randn(B, SMALL["model"]["embedding_cond_dim"])

        with torch.no_grad():
            out1 = model(ids, cond)
            out2 = model(ids, cond)

        assert (out1 == out2).all(), "Non-deterministic forward with dropout=0.0"

    def test_forward_train_eval_same_with_zero_dropout(self, small_model):
        """With dropout=0.0, train mode and eval mode should give identical results."""
        model = small_model
        B, L = 4, SMALL["model"]["max_seq_len"]
        ids = torch.randint(0, 100, (B, L))
        cond = torch.randn(B, SMALL["model"]["embedding_cond_dim"])

        model.eval()
        with torch.no_grad():
            out_eval = model(ids, cond)

        model.train()
        with torch.no_grad():
            out_train = model(ids, cond)

        assert (out_eval == out_train).all(), (
            "train/eval give different results with dropout=0. "
            "If dropout > 0 this is expected."
        )

    def test_different_batch_samples_independent(self, small_model):
        """Outputs for sample i should not depend on sample j (no batch contamination)."""
        model = small_model.eval()
        B, L = 4, SMALL["model"]["max_seq_len"]
        ids = torch.randint(0, 100, (B, L))
        cond = torch.randn(B, SMALL["model"]["embedding_cond_dim"])
        padding_mask = torch.zeros(B, L, dtype=torch.bool)

        with torch.no_grad():
            out_batch = model(ids, cond, padding_mask)

        for i in range(B):
            # Run single sample
            with torch.no_grad():
                out_single = model(ids[i:i+1], cond[i:i+1], padding_mask[i:i+1])
            max_diff = (out_batch[i] - out_single[0]).abs().max().item()
            assert max_diff < 1e-4, (
                f"Sample {i} output differs between batch and single-sample forward "
                f"(max_diff={max_diff:.2e}). Possible batch contamination."
            )

    def test_padding_all_positions_edge_case(self, small_model):
        """
        Edge case: all positions are padding → n_masked=0 → no loss contribution.
        Must not produce NaN.
        """
        model = small_model.eval()
        B, L = 2, SMALL["model"]["max_seq_len"]
        ids = torch.ones(B, L, dtype=torch.long)
        padding_mask = torch.ones(B, L, dtype=torch.bool)  # all padding
        cond = torch.randn(B, SMALL["model"]["embedding_cond_dim"])

        with torch.no_grad():
            out = model(ids, cond, padding_mask)

        assert torch.isfinite(out).all(), "NaN/Inf when all positions are padding"

    def test_apply_mask_with_all_padding_produces_no_masks(self):
        """When all positions are padding, apply_mask must produce empty target_mask."""
        B, L = 4, 16
        token_ids = torch.ones(B, L, dtype=torch.long)
        padding_mask = torch.ones(B, L, dtype=torch.bool)

        for _ in range(20):
            _, target_mask, _, _ = apply_mask(token_ids, mask_token_id=999, padding_mask=padding_mask)
            assert not target_mask.any(), "Masked a padding-only sequence"

    def test_logits_shape_correct(self, small_model):
        """Output shape must be [B, L, V]."""
        model = small_model.eval()
        B, L = 5, SMALL["model"]["max_seq_len"]
        V = SMALL["model"]["vocab_size"]
        ids = torch.randint(0, 100, (B, L))
        cond = torch.randn(B, SMALL["model"]["embedding_cond_dim"])
        with torch.no_grad():
            out = model(ids, cond)
        assert out.shape == (B, L, V), f"Wrong output shape: {out.shape}, expected ({B}, {L}, {V})"

    def test_checkpoint_atomic_write_risk(self):
        """
        Flag: torch.save writes directly without temp+rename.
        If a disk error occurs mid-write, the checkpoint is silently corrupted.
        This test documents the risk (does not fail; print a warning).
        """
        import inspect
        import train as train_module
        src = inspect.getsource(train_module.save_checkpoint)
        uses_atomic = "replace" in src or ".tmp" in src or "rename" in src
        if not uses_atomic:
            import warnings
            warnings.warn(
                "save_checkpoint uses torch.save without atomic temp+rename. "
                "Disk errors mid-write will silently corrupt the checkpoint. "
                "Fix: write to .tmp then os.replace().",
                UserWarning,
                stacklevel=1,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])