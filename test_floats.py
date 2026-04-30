import torch
import math

# t_proj grad mathematically 0
x = torch.tensor([0.02], dtype=torch.float32)
print("float32 0.02 >= 0.02:", x.item() >= 0.02)

# ema broken
ema_broken = torch.tensor([0.0], dtype=torch.bfloat16)
model_param_bf16 = torch.tensor([1.0], dtype=torch.bfloat16)
for _ in range(100):
    ema_broken.lerp_(model_param_bf16, 1 - 0.9999)
print("ema_broken after bf16 lerp:", ema_broken.float().item())
