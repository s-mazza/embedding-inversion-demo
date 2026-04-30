
import torch

def test_ema_logic_v2():
    ema_decay = 0.9999
    
    # Current implementation in train.py:
    # ema_model is bfloat16
    ema_model = torch.ones(10, dtype=torch.bfloat16)
    # raw_model (mp) is bfloat16 during accumulation/lerp
    mp = torch.ones(10, dtype=torch.bfloat16) + 0.1
    
    print(f"Initial EMA (bf16): {ema_model[0].item()}")
    
    for i in range(100):
        with torch.no_grad():
            # This is the logic in train.py:
            ep_fp32 = ema_model.float()
            ep_fp32.lerp_(mp.float(), 1 - ema_decay)
            ema_model.copy_(ep_fp32)
            
    print(f"EMA (bf16) after 100 steps: {ema_model[0].item()}")
    
    # Correct implementation: keep EMA in fp32
    ema_fp32 = torch.ones(10, dtype=torch.float32)
    mp_fp32 = torch.ones(10, dtype=torch.float32) + 0.1
    
    for i in range(100):
        with torch.no_grad():
            ema_fp32.lerp_(mp_fp32, 1 - ema_decay)
            
    print(f"EMA (fp32) after 100 steps: {ema_fp32[0].item()}")

if __name__ == "__main__":
    test_ema_logic_v2()
