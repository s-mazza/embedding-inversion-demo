import re

with open("train.py", "r") as f:
    content = f.read()

# 1. Atomic saves
old_saves = '''def save_checkpoint(path, step, best_val_loss, best_step, model, ema_model, optimizer, scaler, config):
    """Save full checkpoint for resuming training (.pt, includes optimizer state)."""
    torch.save({
        "step": step,
        "best_val_loss": best_val_loss,
        "best_step": best_step,
        "model": model.state_dict(),
        "ema_model": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "config": config,
    }, path)


def save_ema(path, step, best_val_loss, ema_model, config):
    """Save inference-only EMA weights as safetensors with metadata."""
    st_path = path.replace(".pt", ".safetensors")
    safetensors_save_model(ema_model, st_path, metadata=_meta(step, best_val_loss, config))'''

new_saves = '''def save_checkpoint(path, step, best_val_loss, best_step, model, ema_model, optimizer, scaler, config):
    """Save full checkpoint for resuming training (.pt, includes optimizer state)."""
    tmp_path = path + ".tmp"
    torch.save({
        "step": step,
        "best_val_loss": best_val_loss,
        "best_step": best_step,
        "model": model.state_dict(),
        "ema_model": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "config": config,
    }, tmp_path)
    os.replace(tmp_path, path)


def save_ema(path, step, best_val_loss, ema_model, config):
    """Save inference-only EMA weights as safetensors with metadata."""
    st_path = path.replace(".pt", ".safetensors")
    tmp_path = st_path + ".tmp"
    safetensors_save_model(ema_model, tmp_path, metadata=_meta(step, best_val_loss, config))
    os.replace(tmp_path, st_path)'''

content = content.replace(old_saves, new_saves)

# 2. EMA model to fp32
old_ema = '''    # EMA model in bf16 to save ~1.2 GB VRAM per GPU (sufficient for inference)
    ema_decay = tc.get("ema_decay", 0.9999)
    ema_model = copy.deepcopy(model).bfloat16()
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)
    if is_main:
        print(f"EMA decay: {ema_decay} (bf16)", flush=True)'''

new_ema = '''    # EMA model must be kept in fp32 to prevent small updates from rounding to zero
    ema_decay = tc.get("ema_decay", 0.9999)
    ema_model = copy.deepcopy(model).float()
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)
    if is_main:
        print(f"EMA decay: {ema_decay} (fp32)", flush=True)'''

content = content.replace(old_ema, new_ema)

# 3. Optimizer weight decay
old_opt = '''    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"]
    )'''

new_opt = '''    # Optimizer: exclude biases and LayerNorm/AdaLN weights from weight decay
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "norm" in name or "adaln" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
            
    optimizer = torch.optim.AdamW([
        {'params': decay_params, 'weight_decay': tc["weight_decay"]},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ], lr=tc["lr"])'''

content = content.replace(old_opt, new_opt)

# 4. Resume EMA
old_resume = '''            if "ema_model" in ckpt:
                ema_sd = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in ckpt["ema_model"].items()}
                # cast to bf16 to match ema_model dtype (checkpoint may be fp32)
                ema_sd = {k: v.bfloat16() for k, v in ema_sd.items()}
                ema_model.load_state_dict(ema_sd)
                if is_main:
                    print("Loaded EMA weights (bf16)", flush=True)'''

new_resume = '''            if "ema_model" in ckpt:
                ema_sd = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in ckpt["ema_model"].items()}
                # keep in fp32
                ema_sd = {k: v.float() for k, v in ema_sd.items()}
                ema_model.load_state_dict(ema_sd)
                if is_main:
                    print("Loaded EMA weights (fp32)", flush=True)'''

content = content.replace(old_resume, new_resume)

# 5. Loss weighting
old_loss = '''            # Mean CE per masked position per sample, then weight by 1/t (Eq. 4)
            n_masked_per_sample = target_mask.float().sum(-1).clamp(min=1)  # [B]
            per_sample_ce = sample_ce_sum / n_masked_per_sample             # [B]
            loss = (per_sample_ce / t_sample.squeeze(1)).mean()'''

new_loss = '''            # MDLM Eq 4: 1/t weights the sum of CE across masked positions for each sample.
            # (Not the mean, which would incorrectly attenuate gradients when t ≈ 1)
            loss = (sample_ce_sum / t_sample.squeeze(1)).mean()'''

content = content.replace(old_loss, new_loss)

# 6. EMA update
old_ema_update = '''        # EMA update: accumulate in fp32 to avoid bf16 precision underflow
        # (bf16 step size ~0.008 >> ema update ~0.0001, rounds to zero otherwise)
        with torch.no_grad():
            for ep, mp in zip(ema_model.parameters(), raw_model.parameters()):
                ep_fp32 = ep.float()
                ep_fp32.lerp_(mp.float(), 1 - ema_decay)
                ep.copy_(ep_fp32)'''

new_ema_update = '''        # EMA update (ema_model is in fp32)
        with torch.no_grad():
            for ep, mp in zip(ema_model.parameters(), raw_model.parameters()):
                ep.lerp_(mp.float(), 1 - ema_decay)'''

content = content.replace(old_ema_update, new_ema_update)

with open("train.py", "w") as f:
    f.write(content)

print("train.py patched!")
