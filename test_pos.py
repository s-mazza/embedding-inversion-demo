
import torch
import yaml
from model import ConditionalMDLM
import torch.nn.functional as F

def test_position_sensitivity():
    config = {
        "model": {
            "vocab_size": 256001,
            "max_seq_len": 32,
            "hidden_dim": 768,
            "num_layers": 2,
            "num_heads": 12,
            "ff_dim": 1152,
            "embedding_cond_dim": 1024,
            "mask_token_id": 256000,
            "pretrained_token_embeddings": "jhu-clsp/mmBERT-base",
            "freeze_token_embeddings": True,
            "tie_weights": False
        }
    }

    print("--- Testing Position Sensitivity in v3 (mmBERT) ---")
    try:
        model = ConditionalMDLM(config).cuda().eval()
    except:
        model = ConditionalMDLM(config).cpu().eval()
    
    device = next(model.parameters()).device
    
    # Input: all mask tokens
    input_ids = torch.full((1, 32), 256000, device=device).long()
    cond = torch.randn(1, 1024, device=device)
    
    with torch.no_grad():
        logits = model(input_ids, cond) # [1, 32, V]
    
    # Check if logits at different positions are the same
    # If position-insensitive, all positions should have same logits
    diff = (logits[0, 0] - logits[0, 1]).abs().max().item()
    print(f"Max logit diff between position 0 and 1 (all masked): {diff}")
    
    if diff < 1e-6:
        print("BUG CONFIRMED: Model is POSITION-INSENSITIVE in v3 path.")
        print("This means RoPE is NOT working.")
    else:
        print("Model is position-sensitive (diff > 0).")

if __name__ == "__main__":
    test_position_sensitivity()
