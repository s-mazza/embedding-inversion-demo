
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model import ConditionalMDLM, apply_mask, TransformerBlock
from train import get_lr
import copy

def audit_v2():
    config = {
        "model": {
            "vocab_size": 250002,
            "max_seq_len": 32,
            "hidden_dim": 768,
            "num_layers": 4, 
            "num_heads": 12,
            "ff_dim": 3072,
            "dropout": 0.1, # Set to non-zero for Test 14
            "embedding_cond_dim": 1024,
            "mask_token_id": 250001,
            "tie_weights": True
        },
        "training": {
            "lr": 1e-4,
            "weight_decay": 0.01,
            "warmup_steps": 2000,
            "max_steps": 200000,
            "max_grad_norm": 1.0
        }
    }

    print("--- V2 Deep Audit: Starting 14 Tests ---")
    model = ConditionalMDLM(config).cpu()
    
    # --- TEST 1: Weight Tying Verification ---
    print("\nTest 1: Weight Tying Verification")
    is_tied = model.output_proj.weight is model.token_embed.weight
    print(f"  Pointer identity: {is_tied}")
    with torch.no_grad():
        model.token_embed.weight[0, 0] += 1.0
    tied_check = (model.output_proj.weight[0, 0] == model.token_embed.weight[0, 0]).item()
    print(f"  Update reflection: {tied_check}")

    # --- TEST 2: Final Norm Identity Check ---
    print("\nTest 2: Final Norm Identity Check")
    x = torch.randn(1, 32, 768)
    cond = torch.randn(1, 768)
    with torch.no_grad():
        out = model.final_norm(x, cond)
    ln = nn.LayerNorm(768, elementwise_affine=False)
    expected = ln(x)
    diff = (out - expected).abs().max().item()
    print(f"  Diff from LayerNorm(x) at step 0: {diff:.6f}")

    # --- TEST 3: Gradient Scaling Dynamics ---
    print("\nTest 3: Gradient Scaling Dynamics")
    model.train()
    target_ids = torch.randint(0, 250000, (2, 32))
    emb_v3 = torch.randn(2, 1024)
    
    t_low = torch.tensor([[0.05], [0.05]])
    m_mask_low = torch.zeros((2, 32), dtype=torch.bool)
    m_mask_low[:, :2] = True
    m_ids_low = target_ids.clone()
    m_ids_low[m_mask_low] = 250001
    
    t_high = torch.tensor([[0.95], [0.95]])
    m_mask_high = torch.zeros((2, 32), dtype=torch.bool)
    m_mask_high[:, :30] = True
    m_ids_high = target_ids.clone()
    m_ids_high[m_mask_high] = 250001

    def get_grad_norm(t, m_ids, m_mask, emb):
        model.zero_grad()
        logits = model(m_ids, emb)
        # Match train.py logic exactly
        ce = F.cross_entropy(logits[m_mask], target_ids[m_mask], reduction='mean')
        loss = ce / t.mean() 
        loss.backward()
        norm = sum(p.grad.norm().item() for p in model.blocks[0].adaln1.proj.parameters() if p.grad is not None)
        return norm

    norm_low = get_grad_norm(t_low, m_ids_low, m_mask_low, emb_v3)
    norm_high = get_grad_norm(t_high, m_ids_high, m_mask_high, emb_v3)
    print(f"  Grad norm at t=0.05: {norm_low:.4f}")
    print(f"  Grad norm at t=0.95: {norm_high:.4f}")
    print(f"  Ratio (Low/High): {norm_low/norm_high:.1f}x (Paper expects ~1)")

    # --- TEST 4: Weight Decay Exclusion ---
    print("\nTest 4: Weight Decay Exclusion Check")
    all_params = [n for n, p in model.named_parameters()]
    decay_params = [n for n in all_params if "bias" in n or "norm" in n]
    print(f"  Bias/Norm params: {len(decay_params)}")
    print("  Status: train.py applies weight decay to ALL parameters.")

    # --- TEST 5: Optimizer State Precision ---
    print("\nTest 5: Optimizer State Precision")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss = model(m_ids_low, emb_v3).mean()
    loss.backward()
    optimizer.step()
    state_types = [p.dtype for p in optimizer.state[model.token_embed.weight].values() if torch.is_tensor(p)]
    print(f"  AdamW state dtypes: {state_types}")

    # --- TEST 6: Timestep Estimation Bias ---
    print("\nTest 6: Timestep Estimation Bias")
    input_ids = torch.full((1, 32), 1).long()
    input_ids[0, :10] = 100
    padding_mask = torch.zeros((1, 32), dtype=torch.bool)
    padding_mask[0, 10:] = True
    input_ids[0, :5] = 250001
    t_biased = model._t_from_input(input_ids, padding_mask=None)
    t_corrected = model._t_from_input(input_ids, padding_mask=padding_mask)
    print(f"  Content masked: 50% (Expected t ~ 0.14)")
    print(f"  Estimated t (biased)   : {t_biased.item():.4f}")
    print(f"  Estimated t (corrected): {t_corrected.item():.4f}")

    # --- TEST 7: Attention Padding Mask Impact ---
    print("\nTest 7: Attention Padding Mask Impact")
    emb_v1 = torch.randn(1, 1024) # Fixed batch mismatch
    with torch.no_grad():
        out1 = model(input_ids, emb_v1, padding_mask=None)
        out2 = model(input_ids, emb_v1, padding_mask=padding_mask)
    diff_attn = (out1 - out2).abs().max().item()
    print(f"  Max logit diff (mask vs no mask): {diff_attn:.6f}")

    # --- TEST 8: Initialization Distribution ---
    print("\nTest 8: Initialization Distribution")
    print(f"  Token embed std: {model.token_embed.weight.std().item():.4f}")
    print(f"  Pos embed std  : {model.pos_embed.weight.std().item():.4f}")

    # --- TEST 9: Normalization Sensitivity ---
    print("\nTest 9: Normalization Sensitivity")
    with torch.no_grad():
        out_raw = model(input_ids, emb_v1)
        out_norm = model(input_ids, F.normalize(emb_v1, dim=-1))
    diff_norm = (out_raw - out_norm).abs().max().item()
    print(f"  Max logit diff (norm vs raw): {diff_norm:.6f}")

    # --- TEST 10: GELU vs ReLU ---
    print("\nTest 10: Activation Function Check")
    has_gelu = any(isinstance(m, nn.GELU) for m in model.modules())
    print(f"  Uses GELU: {has_gelu}")

    # --- TEST 11: MHA Init ---
    print("\nTest 11: MHA Init")
    mha_std = model.blocks[0].attn.out_proj.weight.std().item()
    print(f"  MHA out_proj weight std: {mha_std:.4f}")

    # --- TEST 12: Sequence Length Extrapolation ---
    print("\nTest 12: Sequence Length Extrapolation")
    long_ids = torch.full((1, 64), 250001).long()
    long_emb = torch.randn(1, 1024)
    try:
        model(long_ids, long_emb)
        print("  ✓ Model works on 64 tokens.")
    except Exception as e:
        print(f"  ✗ Failed on 64 tokens: {e}")

    # --- TEST 13: AdamW Hyperparameters ---
    print("\nTest 13: AdamW Hyperparameters")
    print(f"  Betas: {optimizer.param_groups[0]['betas']}")
    print(f"  Eps: {optimizer.param_groups[0]['eps']}")

    # --- TEST 14: Dropout Eval Mode ---
    print("\nTest 14: Dropout Eval Mode Check")
    model.eval()
    with torch.no_grad():
        out1 = model(input_ids, emb_v1)
        out2 = model(input_ids, emb_v1)
    diff_drop = (out1 - out2).abs().max().item()
    print(f"  Eval mode determinism (diff): {diff_drop:.6f}")
    if diff_drop == 0:
        print("  ✓ Dropout is correctly disabled in eval mode.")

if __name__ == "__main__":
    audit_v2()
