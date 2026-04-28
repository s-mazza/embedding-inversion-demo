"""
Conditional Masked Diffusion Language Model (CMDLM) for embedding inversion.

Two architectures supported via config:
  - From-scratch (v2): 8-layer custom transformer, no pretrained backbone.
    Activated when 'pretrained_token_embeddings' is absent from config.
    State-dict compatible with demo_server.py.
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
# From-scratch architecture (v2) — demo_server.py compatible
# ---------------------------------------------------------------------------

class AdaLN(nn.Module):
    """Adaptive LayerNorm for from-scratch architecture (scale + shift, no gate)."""

    def __init__(self, hidden_dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.proj = nn.Linear(cond_dim, 2 * hidden_dim)

    def forward(self, x, cond):
        params = self.proj(cond).unsqueeze(1)
        scale, shift = params.chunk(2, dim=-1)
        return self.norm(x) * (1 + scale) + shift


class TransformerBlock(nn.Module):
    """Standard transformer block with AdaLN conditioning (v2 from-scratch)."""

    def __init__(self, hidden_dim, num_heads, ff_dim, dropout=0.0):
        super().__init__()
        self.adaln1 = AdaLN(hidden_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.adaln2 = AdaLN(hidden_dim, hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
        )

    def forward(self, x, cond):
        normed = self.adaln1(x, cond)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        x = x + attn_out
        normed = self.adaln2(x, cond)
        x = x + self.ff(normed)
        return x


# ---------------------------------------------------------------------------
# mmBERT architecture (v3) — pretrained backbone with AdaLN-Zero
# ---------------------------------------------------------------------------

class AdaLNZero(nn.Module):
    """
    Adaptive Layer Normalization with Zero initialization (DiT-style).
    
    Output: norm(x) * (1 + scale) + shift
    Gate: x = x + alpha * block_output (alpha initialized to 0)
    
    All conditioning projections initialized to zero so the model starts
    as identity (vanilla mmBERT) and gradually learns to use conditioning.
    """

    def __init__(self, hidden_dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        # Project condition to (scale, shift, alpha)
        self.proj = nn.Linear(cond_dim, 3 * hidden_dim)
        # Zero initialization - model starts as identity
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, cond):
        """
        Args:
            x: [B, L, hidden] input
            cond: [B, cond_dim] conditioning vector
        
        Returns:
            normalized: norm(x) * (1 + scale) + shift
            alpha: [B, 1, hidden] gate for residual (to be used as x = x + alpha * block_output)
        """
        # cond: [B, cond_dim] -> [B, 3*hidden]
        params = self.proj(cond).unsqueeze(1)  # [B, 1, 3*hidden]
        scale, shift, alpha = params.chunk(3, dim=-1)  # each [B, 1, hidden]
        normalized = self.norm(x) * (1 + scale) + shift
        return normalized, alpha


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
        self.pos_embed = nn.Embedding(self.max_seq_len, self.hidden_dim)
        cond_dim = mc["embedding_cond_dim"]
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.blocks = nn.ModuleList([
            TransformerBlock(self.hidden_dim, mc["num_heads"], mc["ff_dim"], mc.get("dropout", 0.0))
            for _ in range(mc["num_layers"])
        ])
        self.final_norm = AdaLN(self.hidden_dim, self.hidden_dim)
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

    def forward(self, input_ids, cond_embedding, padding_mask=None):
        """
        Args:
            input_ids: [B, L] token ids (with some masked by mask_token_id)
            cond_embedding: [B, cond_dim] target embedding vector
            padding_mask: [B, L] True where padding (mmBERT path only)
        Returns:
            logits: [B, L, V]
        """
        if self._from_scratch:
            return self._forward_scratch(input_ids, cond_embedding)
        return self._forward_mmbert(input_ids, cond_embedding, padding_mask)

    def _forward_scratch(self, input_ids, cond_embedding):
        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0)
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        cond = self.cond_proj(cond_embedding)
        for block in self.blocks:
            x = block(x, cond)
        x = self.final_norm(x, cond)
        return self.output_proj(x)

    def _forward_mmbert(self, input_ids, cond_embedding, padding_mask=None):
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

    def forward_hidden(self, input_ids, cond_embedding, padding_mask=None):
        """Returns hidden states before output_proj."""
        if self._from_scratch:
            B, L = input_ids.shape
            positions = torch.arange(L, device=input_ids.device).unsqueeze(0)
            x = self.token_embed(input_ids) + self.pos_embed(positions)
            cond = self.cond_proj(cond_embedding)
            for block in self.blocks:
                x = block(x, cond)
            return self.final_norm(x, cond)
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

    For each sample in batch, randomly choose a mask ratio in [0.1, 1.0],
    then mask that fraction of non-padding tokens.

    Returns:
        masked_ids: [B, L] with some tokens replaced by mask_token_id
        target_mask: [B, L] boolean, True at positions that were masked (loss targets)
        mask_ratio: [B, 1] the mask ratio used for each sample
    """
    B, L = token_ids.shape
    device = token_ids.device

    # Random mask ratio per sample
    # Log-linear noise schedule (MDLM paper: better than uniform)
    u = torch.rand(B, 1, device=device)
    eps = 1e-3
    mask_ratio = 1 - (1 - eps) ** u  # log-linear, range ~[eps, 1.0]
    mask_ratio = mask_ratio.clamp(min=0.1, max=1.0)  # keep min 0.1 for stability

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

    return masked_ids, target_mask, mask_ratio  # mask_ratio: [B, 1]
