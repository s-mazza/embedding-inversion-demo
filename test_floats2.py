import torch
ema_broken = torch.tensor([0.5], dtype=torch.bfloat16)
model_param_bf16 = torch.tensor([1.0], dtype=torch.bfloat16)
for _ in range(100):
    ema_broken.lerp_(model_param_bf16, 1 - 0.9999)
print("ema_broken from 0.5 after bf16 lerp:", ema_broken.float().item())
