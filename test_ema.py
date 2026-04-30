
import torch

def test_ema_bf16():
    ema_decay = 0.9999
    
    # bf16 version
    ema_bf16 = torch.ones(10, dtype=torch.bfloat16)
    model_weights = torch.ones(10, dtype=torch.bfloat16) + 0.1 # Some difference
    
    print(f"Initial EMA (bf16): {ema_bf16[0].item()}")
    
    for _ in range(100):
        with torch.no_grad():
            ep_fp32 = ema_bf16.float()
            ep_fp32.lerp_(model_weights.float(), 1 - ema_decay)
            ema_bf16.copy_(ep_fp32)
            
    print(f"EMA (bf16) after 100 steps: {ema_bf16[0].item()}")
    
    # fp32 version
    ema_fp32 = torch.ones(10, dtype=torch.float32)
    model_weights_fp32 = torch.ones(10, dtype=torch.float32) + 0.1
    
    for _ in range(100):
        with torch.no_grad():
            ema_fp32.lerp_(model_weights_fp32, 1 - ema_decay)
            
    print(f"EMA (fp32) after 100 steps: {ema_fp32[0].item()}")

if __name__ == "__main__":
    test_ema_bf16()
