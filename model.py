"""
Conditional Masked Diffusion Language Model (CMDLM) for embedding inversion.

Two architectures supported via config:
  - From-scratch (v2): 8-layer custom transformer, no pretrained backbone.
    Activated when 'pretrained_token_embeddings' is absent from config.
  - mmBERT (v3): pretrained 22-layer ModernBERT backbone with AdaLN-Zero.
    Activated when 'pretrained_token_embeddings' is set in config.

Input: masked token sequence [B, L] + embedding condition [B, cond_dim]
Output: logits [B, L, V] predicting original tokens at all positions
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint


# ---------------------------------------------------------------------------
# Shared conditioning modules
# ---------------------------------------------------------------------------

class AdaLN(nn.Module):
    """
    AdaLN for final norm in v2 (scale + shift, no gate, zero-initialized).
    Zero-init: at init scale=shift=0 → behaves as plain LayerNorm.
    """

    def __init__(self, hidden_dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.proj = nn.Linear(cond_dim, 2 * hidden_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, cond):
        params = self.proj(cond).unsqueeze(1)  # [B, 1, 2*hidden]
        scale, shift = params.chunk(2, dim=-1)
        return self.norm(x) * (1 + scale) + shift


class AdaLNZero(nn.Module):
    """
    AdaLN-Zero (DiT-style, paper §3.3): scale + shift + gate α, all zero-initialized.
    At init: scale=shift=α=0 → block output is fully gated away → identity mapping.
    Used in both v2 transformer blocks and v3 mmBERT layers.
    """

    def __init__(self, hidden_dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.proj = nn.Linear(cond_dim, 3 * hidden_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, cond):
        """
        Returns:
            normalized: norm(x) * (1 + scale) + shift   [B, L, hidden]
            alpha:      gate for gated residual           [B, 1, hidden]
        """
        params = self.proj(cond).unsqueeze(1)  # [B, 1, 3*hidden]
        scale, shift, alpha = params.chunk(3, dim=-1)
        normalized = self.norm(x) * (1 + scale) + shift
        return normalized, alpha


class AdaLNZeroSplit(nn.Module):
    """
    Per-layer t and c conditioning (paper Eq. 6-9, per-layer superscript ℓ).
    Each block has independent c_proj and t_proj, combined additively.
    All projections zero-initialized → identity at init.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.c_proj = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.t_proj = nn.Linear(hidden_dim, 3 * hidden_dim)
        nn.init.zeros_(self.c_proj.weight)
        nn.init.zeros_(self.c_proj.bias)
        nn.init.zeros_(self.t_proj.weight)
        nn.init.zeros_(self.t_proj.bias)

    def forward(self, x, cond_c, cond_t):
        c_params = self.c_proj(cond_c).unsqueeze(1)  # [B, 1, 3*hidden]
        t_params = self.t_proj(cond_t).unsqueeze(1)  # [B, 1, 3*hidden]
        scale_c, shift_c, alpha_c = c_params.chunk(3, dim=-1)
        scale_t, shift_t, alpha_t = t_params.chunk(3, dim=-1)
        scale = scale_c + scale_t
        shift = shift_c + shift_t
        alpha = alpha_c + alpha_t
        return self.norm(x) * (1 + scale) + shift, alpha


# ---------------------------------------------------------------------------
# From-scratch architecture (v2)
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """
    Transformer block with per-layer AdaLN-Zero conditioning (paper §3.3, Eq. 6-9).

    Each block has independent c_proj and t_proj (via AdaLNZeroSplit) so layers can
    learn different t vs. c sensitivities. Gated residuals ensure identity at init.
    padding_mask: [B, L] True at padding positions; prevents attention to padding tokens.
    """

    def __init__(self, hidden_dim, num_heads, ff_dim, dropout=0.0):
        super().__init__()
        self.adaln1 = AdaLNZeroSplit(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.adaln2 = AdaLNZeroSplit(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
        )

    def forward(self, x, cond_c, cond_t, padding_mask=None):
        if padding_mask is not None:
            # Prevent NaN in MultiheadAttention if an entire sequence is padding
            all_padded = padding_mask.all(dim=-1)
            if all_padded.any():
                padding_mask = padding_mask.clone()
                padding_mask[all_padded, 0] = False

        normed, alpha1 = self.adaln1(x, cond_c, cond_t)
        attn_out, _ = self.attn(normed, normed, normed,
                                key_padding_mask=padding_mask, need_weights=False)
        x = x + alpha1 * attn_out
        normed, alpha2 = self.adaln2(x, cond_c, cond_t)
        x = x + alpha2 * self.ff(normed)
        return x


# ---------------------------------------------------------------------------
# mmBERT architecture (v3) — pretrained backbone with AdaLN-Zero
# ---------------------------------------------------------------------------

class ModernBertLayerWithAdaLN(nn.Module):
    """
    Wraps a pretrained ModernBERT layer with AdaLN-Zero conditioning.
    
    Architecture:
        - Replace layer's internal norms with AdaLN-Zero
        - Add gated residuals with alpha initialized to 0
    """

    def __init__(self, pretrained_layer, hidden_dim, cond_dim):
        super().__init__()
        self.pretrained_layer = pretrained_layer
        
        # AdaLN-Zero conditioning  parameters (scale, shift, gate)
        # We project cond to get 6*hidden: (scale_attn, shift_attn, alpha_attn, scale_mlp, shift_mlp, alpha_mlp)
        self.cond_proj = nn.Linear(cond_dim, 6 * hidden_dim)
        nn.init.zeros_(self.cond_proj.weight)
        nn.init.zeros_(self.cond_proj.bias)
        
        # Store reference to original norms (we'll replace their functionality)
        self.attn_norm = pretrained_layer.attn_norm  
        self.mlp_norm = pretrained_layer.mlp_norm

    def forward(self, hidden_states, cond, position_embeddings):
        """
        Args:
            hidden_states: [B, L, hidden]
            cond: [B, cond_dim] conditioning vector
            position_embeddings: (cos, sin) tuple for RoPE
        
        Returns:
            hidden_states: [B, L, hidden]
        """
        # Get all AdaLN-Zero parameters at once
        adaln_params = self.cond_proj(cond).unsqueeze(1)  # [B, 1, 6*hidden]
        scale_attn, shift_attn, alpha_attn, scale_mlp, shift_mlp, alpha_mlp = \
            adaln_params.chunk(6, dim=-1)  # each [B, 1, hidden]
        
        # Attention block
        # Apply AdaLN-Zero norm (replace attn_norm)
        if isinstance(self.attn_norm, nn.Identity):
            # Layer 0 has no norm
            normed_attn = hidden_states * (1 + scale_attn) + shift_attn
        else:
            # Use pretrained norm but apply AdaLN conditioning
            normed_attn = self.attn_norm(hidden_states) * (1 + scale_attn) + shift_attn

        B, L, _ = hidden_states.shape
        position_ids = torch.arange(L, device=hidden_states.device).unsqueeze(0).expand(B, -1)

        # Attention (returns tuple (output, attention_weights) but we only need output)
        attn_out = self.pretrained_layer.attn(
            normed_attn,
            attention_mask=None,
            sliding_window_mask=None,
            position_ids=position_ids,
            position_embeddings=None,
        )
        # attn might return a tuple, take first element
        attn_output = attn_out[0] if isinstance(attn_out, tuple) else attn_out
        
        # Gated residual: alpha_attn is [B, 1, hidden], broadcasts to [B, L, hidden]
        hidden_states = hidden_states + (alpha_attn * attn_output)
        
        # MLP block
        # Apply AdaLN-Zero norm (replace mlp_norm)
        normed_mlp = self.mlp_norm(hidden_states) * (1 + scale_mlp) + shift_mlp
        
        # MLP
        mlp_output = self.pretrained_layer.mlp(normed_mlp)
        
        # Gated residual: alpha_mlp is [B, 1, hidden], broadcasts to [B, L, hidden]
        hidden_states = hidden_states + (alpha_mlp * mlp_output)
        
        return hidden_states


class ConditionalMDLM(nn.Module):
    """
    Conditional Masked Diffusion Language Model for embedding inversion.

    Dispatches to from-scratch (v2) or mmBERT (v3) architecture based on
    whether 'pretrained_token_embeddings' is present in the model config.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        mc = config["model"]

        self.vocab_size = mc["vocab_size"]
        self.hidden_dim = mc["hidden_dim"]
        self.max_seq_len = mc["max_seq_len"]
        self.mask_token_id = mc["mask_token_id"]

        pretrained_model = mc.get("pretrained_token_embeddings")
        self._from_scratch = not bool(pretrained_model)

        if self._from_scratch:
            self._init_scratch(mc)
        else:
            self._init_mmbert(mc, pretrained_model)

        total, trainable = self.count_params()
        print(f"Total params: {total:,} | Trainable: {trainable:,} ({100*trainable/total:.1f}%)")

    def _init_scratch(self, mc):
        """8-layer from-scratch architecture — state-dict compatible with demo_server.py."""
        self.token_embed = nn.Embedding(self.vocab_size, self.hidden_dim)
        nn.init.normal_(self.token_embed.weight, std=0.02)
        self.pos_embed = nn.Embedding(self.max_seq_len, self.hidden_dim)
        nn.init.normal_(self.pos_embed.weight, std=0.02)
        self.embed_norm = nn.LayerNorm(self.hidden_dim)
        cond_dim = mc["embedding_cond_dim"]
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        # Shared scalar-t → hidden embedding; per-layer c/t projections live in AdaLNZeroSplit.
        self.t_embed = nn.Linear(1, self.hidden_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(self.hidden_dim, mc["num_heads"], mc["ff_dim"], mc.get("dropout", 0.0))
            for _ in range(mc["num_layers"])
        ])
        self.final_norm = AdaLNZeroSplit(self.hidden_dim)
        self.output_proj = nn.Linear(self.hidden_dim, self.vocab_size, bias=False)
        if mc.get("tie_weights", False):
            self.output_proj.weight = self.token_embed.weight
            print("Weight tying: output_proj shares token_embed (saves 192M params)")
        print(f"From-scratch: {mc['num_layers']} layers, hidden={self.hidden_dim}, vocab={self.vocab_size}")

    def _init_mmbert(self, mc, pretrained_model):
        """22-layer mmBERT backbone with AdaLN-Zero conditioning."""
        # Load pretrained mmBERT model
        
        print(f"Loading pretrained mmBERT from {pretrained_model}...")
        from transformers import AutoModel  # noqa: PLC0415
        mmbert = AutoModel.from_pretrained(pretrained_model, trust_remote_code=True)
        
        # Extract components
        pretrained_embeddings = mmbert.embeddings.tok_embeddings
        self.embed_norm = mmbert.embeddings.norm
        self.embed_drop = mmbert.embeddings.drop
        # rotary_emb handled internally by ModernBert
        
        # Extend token embeddings if vocab_size is larger (e.g., for mask token)
        pretrained_vocab = pretrained_embeddings.weight.shape[0]
        if self.vocab_size > pretrained_vocab:
            print(f"Extending embeddings from {pretrained_vocab} to {self.vocab_size} (adding mask token)")
            self.token_embed = nn.Embedding(self.vocab_size, self.hidden_dim)
            # Copy pretrained weights
            self.token_embed.weight.data[:pretrained_vocab].copy_(pretrained_embeddings.weight.data)
            # Initialize extra tokens (mask token) with small random values
            nn.init.normal_(self.token_embed.weight.data[pretrained_vocab:], std=0.02)
        else:
            self.token_embed = pretrained_embeddings
        
        # Freeze token embeddings if specified
        self.freeze_embeddings = mc.get("freeze_token_embeddings", False)
        if self.freeze_embeddings:
            self.token_embed.weight.requires_grad_(False)
            print(f"Frozen token embeddings ({self.vocab_size} x {self.hidden_dim})")
        
        # Project conditioning embedding to internal dim
        cond_dim = mc["embedding_cond_dim"]
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        
        # Wrap pretrained transformer layers with AdaLN-Zero
        print(f"Wrapping {len(mmbert.layers)} mmBERT layers with AdaLN-Zero conditioning...")
        self.layers = nn.ModuleList([
            ModernBertLayerWithAdaLN(layer, self.hidden_dim, self.hidden_dim)
            for layer in mmbert.layers
        ])
        
        # Final layer norm with AdaLN-Zero
        self.final_adaln = AdaLNZero(self.hidden_dim, self.hidden_dim)
        
        # Output projection
        self.tie_weights = mc.get("tie_weights", True)
        self.output_proj = nn.Linear(self.hidden_dim, self.vocab_size, bias=False)
        
        if self.tie_weights:
            # Tie with token embeddings
            self.output_proj.weight = self.token_embed.weight
            if self.freeze_embeddings:
                # If embeddings are frozen and tied, output_proj is also frozen
                print("Weight tying: output_proj shares weights with token_embed (frozen)")
            else:
                print("Weight tying: output_proj shares weights with token_embed")
        else:
            # Initialize from token embeddings (already extended to vocab_size)
            self.output_proj.weight.data.copy_(self.token_embed.weight.data)
            print("output_proj initialized from token embeddings (independent, trainable)")
        
        # RoPE position embeddings handled internally by ModernBert
        # Clean up
        del mmbert
        
        # Gradient checkpointing (disabled by default)
        self.use_checkpoint = False

    def forward(self, input_ids, cond_embedding, padding_mask=None, t=None):
        """
        Args:
            input_ids: [B, L] token ids (with some masked by mask_token_id)
            cond_embedding: [B, cond_dim] target embedding vector
            padding_mask: [B, L] True where padding
            t: [B, 1] explicit timestep. If None, recovered from realised mask
                fraction via _t_from_input (inference fallback).
        Returns:
            logits: [B, L, V]
        """
        if self._from_scratch:
            return self._forward_scratch(input_ids, cond_embedding, padding_mask, t)
        return self._forward_mmbert(input_ids, cond_embedding, padding_mask, t)

    def _t_from_input(self, input_ids, padding_mask=None):
        """
        Estimate t from mask fraction, inverting the log-linear schedule (λ=5).
        Uses only content positions (not padding) to avoid systematic underestimation.
        padding_mask: [B, L] True at padding positions.
        """
        is_mask = (input_ids == self.mask_token_id).float()  # [B, L]
        if padding_mask is not None:
            content = (~padding_mask).float()  # [B, L], 1 at content positions
            content_len = content.sum(dim=-1, keepdim=True).clamp(min=1)
            frac = (is_mask * content).sum(dim=-1, keepdim=True) / content_len
        else:
            frac = is_mask.mean(dim=-1, keepdim=True)
        t = (-torch.log((1 - frac).clamp(min=1e-4)) / 5.0).clamp(0.02, 1.0)
        return t  # [B, 1]

    def _forward_scratch(self, input_ids, cond_embedding, padding_mask=None, t=None):
        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0)
        x = self.embed_norm(self.token_embed(input_ids) + self.pos_embed(positions))
        cond_c = self.cond_proj(cond_embedding)
        if t is None:
            t = self._t_from_input(input_ids, padding_mask)
        cond_t = self.t_embed(t)
        for block in self.blocks:
            x = block(x, cond_c, cond_t, padding_mask)
        x_normed, _ = self.final_norm(x, cond_c, cond_t)
        return self.output_proj(x_normed)

    def _forward_mmbert(self, input_ids, cond_embedding, padding_mask=None, t=None):
        hidden_states = self.token_embed(input_ids)
        hidden_states = self.embed_norm(hidden_states)
        hidden_states = self.embed_drop(hidden_states)
        cond = self.cond_proj(cond_embedding)
        position_embeddings = None
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                hidden_states = torch_checkpoint(
                    layer, hidden_states, cond, position_embeddings,
                    use_reentrant=False
                )
            else:
                hidden_states = layer(hidden_states, cond, position_embeddings)
        hidden_states, _ = self.final_adaln(hidden_states, cond)
        return self.output_proj(hidden_states)

    def forward_hidden(self, input_ids, cond_embedding, padding_mask=None, t=None):
        """Returns hidden states before output_proj.

        t: [B, 1] explicit timestep. If None, recovered from realised mask
        fraction via _t_from_input (inference fallback).
        """
        if self._from_scratch:
            B, L = input_ids.shape
            positions = torch.arange(L, device=input_ids.device).unsqueeze(0)
            x = self.embed_norm(self.token_embed(input_ids) + self.pos_embed(positions))
            cond_c = self.cond_proj(cond_embedding)
            if t is None:
                t = self._t_from_input(input_ids, padding_mask)
            cond_t = self.t_embed(t)
            for block in self.blocks:
                x = block(x, cond_c, cond_t, padding_mask)
            x_normed, _ = self.final_norm(x, cond_c, cond_t)
            return x_normed
        hidden_states = self.token_embed(input_ids)
        hidden_states = self.embed_norm(hidden_states)
        hidden_states = self.embed_drop(hidden_states)
        cond = self.cond_proj(cond_embedding)
        for layer in self.layers:
            hidden_states = layer(hidden_states, cond, None)
        hidden_states, _ = self.final_adaln(hidden_states, cond)
        return hidden_states

    def count_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


def apply_mask(token_ids, mask_token_id, padding_mask=None):
    """
    Apply random masking for MDLM training.

    Sample t ~ Uniform[0, 1] per the paper, then derive mask_ratio from the
    log-linear noise schedule. The training loop applies the 1/t weight floor
    (clamp at min=0.02) on the *weight*, not on t itself, so the model still
    sees the full t distribution in conditioning.

    Returns:
        masked_ids: [B, L] with some tokens replaced by mask_token_id
        target_mask: [B, L] boolean, True at positions that were masked (loss targets)
        mask_ratio: [B, 1] the mask ratio used for each sample
        t: [B, 1] sampled timestep
    """
    B, L = token_ids.shape
    device = token_ids.device

    # Log-linear noise schedule: α_t = e^{-λt}, λ=5.0 (MDLM paper §3.2)
    # t ~ Uniform[0, 1] (paper §4); mask_ratio = 1 - e^{-5t}
    t = torch.rand(B, 1, device=device)
    mask_ratio = 1 - torch.exp(-5.0 * t)

    # Random scores for each position
    rand_scores = torch.rand(B, L, device=device)

    # Don't mask padding positions
    if padding_mask is not None:
        rand_scores[padding_mask] = 2.0  # never mask padding

    # Mask positions where rand < ratio
    target_mask = rand_scores < mask_ratio  # [B, L]

    # Apply mask
    masked_ids = token_ids.clone()
    masked_ids[target_mask] = mask_token_id

    return masked_ids, target_mask, mask_ratio, t  # t returned for 1/t loss weighting (Eq. 4)
