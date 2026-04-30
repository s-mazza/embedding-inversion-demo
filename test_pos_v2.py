
import torch
from model import ConditionalMDLM

def test_pos_v2():
    config = {
        "model": {
            "vocab_size": 250002,
            "max_seq_len": 32,
            "hidden_dim": 768,
            "num_layers": 8,
            "num_heads": 12,
            "ff_dim": 3072,
            "embedding_cond_dim": 1024,
            "mask_token_id": 250001
        }
    }
    model = ConditionalMDLM(config).eval()
    input_ids = torch.full((1, 32), 250001).long()
    cond = torch.randn(1, 1024)
    with torch.no_grad():
        logits = model(input_ids, cond)
    diff = (logits[0, 0] - logits[0, 1]).abs().max().item()
    print(f"v2 Max logit diff between pos 0 and 1: {diff}")

if __name__ == "__main__":
    test_pos_v2()
