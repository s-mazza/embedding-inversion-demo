
import torch
import yaml
from model import ConditionalMDLM, apply_mask
import os

# Create a dummy config for v3 (mmBERT)
config = {
    "model": {
        "vocab_size": 256001,
        "max_seq_len": 32,
        "hidden_dim": 768,
        "num_layers": 2,  # Small for testing
        "num_heads": 12,
        "ff_dim": 1152,
        "embedding_cond_dim": 1024,
        "mask_token_id": 256000,
        "pretrained_token_embeddings": "jhu-clsp/mmBERT-base",
        "freeze_token_embeddings": True,
        "tie_weights": False
    }
}

print("--- Testing mmBERT Path (v3) ---")
# Note: This might download mmBERT weights, which is fine in this environment
try:
    model = ConditionalMDLM(config).cuda()
except Exception as e:
    print(f"Failed to init model: {e}")
    # Fallback to CPU if no GPU
    model = ConditionalMDLM(config).cpu()

device = next(model.parameters()).device

# 1. Check if 't' affects the output in v3 path
# In v3, t is calculated from the input_ids (mask fraction) in apply_mask,
# but it's never used in _forward_mmbert.
print("\n1. Testing if mask fraction (t) affects logits in v3...")
input_ids_1 = torch.full((1, 32), 256000, device=device).long() # All masked
input_ids_1[0, 0] = 100 # One unmasked
input_ids_2 = torch.full((1, 32), 256000, device=device).long() # All masked
input_ids_2[0, 0:10] = 100 # 10 unmasked

cond = torch.randn(1, 1024, device=device)

with torch.no_grad():
    logits_1 = model(input_ids_1, cond)
    logits_2 = model(input_ids_2, cond)

# Compare logits at a common masked position
diff = (logits_1[0, 31] - logits_2[0, 31]).abs().max().item()
print(f"Max logit diff between different mask fractions: {diff}")
if diff < 1e-6:
    print("BUG CONFIRMED: Logits do NOT depend on mask fraction (t) in v3 path.")
else:
    print("Logits DO depend on mask fraction (t) in v3 path (indirectly via attention).")
    print("Wait, if they depend on it, is it because of the tokens themselves or the conditioning?")
    # Test with same tokens but different 't' (if it was possible to pass t)
    # But t is derived from input_ids in _forward_scratch, and NOT EVEN DERIVED in _forward_mmbert.

# 2. Check if padding_mask is used
print("\n2. Testing if padding_mask affects logits...")
padding_mask_1 = torch.zeros((1, 32), device=device, dtype=torch.bool)
padding_mask_2 = torch.zeros((1, 32), device=device, dtype=torch.bool)
padding_mask_2[0, 20:] = True # Pad last 12 tokens

with torch.no_grad():
    logits_p1 = model(input_ids_1, cond, padding_mask=padding_mask_1)
    logits_p2 = model(input_ids_1, cond, padding_mask=padding_mask_2)

diff_p = (logits_p1 - logits_p2).abs().max().item()
print(f"Max logit diff between different padding masks: {diff_p}")
if diff_p < 1e-6:
    print("BUG CONFIRMED: padding_mask is IGNORED in v3 path.")
else:
    print("padding_mask IS used in v3 path.")

# 3. Check parameter counts
total, trainable = model.count_params()
print(f"\n3. Parameter counts for 2-layer v3: Total={total:,}, Trainable={trainable:,}")
# Estimate for 22 layers
# Trainable components: cond_proj (768*768 + 768*768), output_proj (256k*768), 
# layers.cond_proj (1024*6*768 per layer)
# Wait, layers.cond_proj takes 'cond' which is project of cond_embedding (1024 -> 768)
# So it's 768 -> 6*768 per layer.
print(f"Output proj size: {256001 * 768:,}")
print(f"Cond proj per layer size: {768 * 6 * 768:,}")

# 4. Check RoPE
print("\n4. Checking RoPE implementation in ModernBertLayerWithAdaLN...")
# We saw 'position_embeddings=None' in model.py. Let's see if it crashes or works.
try:
    out = model(input_ids_1, cond)
    print("Forward pass successful with position_embeddings=None.")
except Exception as e:
    print(f"Forward pass FAILED: {e}")
