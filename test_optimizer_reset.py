#!/usr/bin/env python3
"""
Verifica se l'Adam state nel checkpoint è "stale" rispetto ai gradienti DDP.

Tests:
  1. Cosine similarity tra Adam momentum (m_t) e gradienti correnti
     → se bassa (<0.5): momentum punta direzione sbagliata → reset utile
     → se alta (>0.7): momentum allineato → reset inutile

  2. Stima della loss spike con reset dell'ottimizzatore
     → esegue 100 step con Adam fresco e registra la loss
     → confronta con 100 step con Adam originale

Usage (inside Docker):
    python test_optimizer_reset.py
    python test_optimizer_reset.py --checkpoint checkpoints_v3_mmbert_jinav3/best.pt
"""

import argparse
import json
import copy
import torch
import torch.nn.functional as F
import numpy as np
from torch.amp import autocast, GradScaler

from model import ConditionalMDLM, apply_mask
from dataset import EmbeddingInversionDataset


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = ConditionalMDLM(config).to(device)
    state = {k.replace("module.", "").replace("_orig_mod.", ""): v
             for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"]
    )
    optimizer.load_state_dict(ckpt["optimizer"])

    return model, optimizer, config, ckpt["step"], ckpt["best_val_loss"]


def get_batch(config, data_dir, device, n=100):
    mc = config["model"]
    ds = EmbeddingInversionDataset(
        data_dir, mc["max_seq_len"],
        val=False, val_split=0.01,
        pad_token_id=1, bos_token_id=None
    )
    indices = torch.randperm(len(ds))[:n]
    token_ids = torch.stack([ds[i]["token_ids"] for i in indices]).to(device)
    embeddings = torch.stack([ds[i]["embedding"] for i in indices]).to(device)
    padding_masks = torch.stack([ds[i]["padding_mask"] for i in indices]).to(device)
    return token_ids, embeddings, padding_masks


def compute_loss(model, token_ids, embeddings, padding_masks, config):
    mc = config["model"]
    mask_id = mc["mask_token_id"]
    masked_ids, target_mask, mask_ratio, _ = apply_mask(token_ids, mask_id, padding_masks)

    with autocast('cuda', dtype=torch.bfloat16):
        hidden = model.forward_hidden(masked_ids, embeddings, padding_masks)
        w = model.output_proj.weight
        total_positions = hidden.shape[0] * hidden.shape[1]
        hidden_flat = hidden.view(-1, hidden.shape[-1])
        targets_flat = token_ids.view(-1)
        mask_flat = target_mask.view(-1).float()
        total_masked = mask_flat.sum()

        total_loss = torch.tensor(0.0, device=hidden.device)
        for i in range(0, total_positions, 256):
            end = min(i + 256, total_positions)
            lc = F.linear(hidden_flat[i:end], w)
            lc_loss = F.cross_entropy(lc, targets_flat[i:end], reduction="none")
            total_loss = total_loss + (lc_loss * mask_flat[i:end]).sum()

        loss = total_loss / total_masked.clamp(min=1)
        loss_weight = (1.0 / mask_ratio.squeeze(1)).mean()
        loss = loss * loss_weight

    return loss


# ── Test 1: Cosine similarity m_t vs current gradient ─────────────────────────

def test_momentum_alignment(model, optimizer, config, data_dir, device):
    print("\n" + "─"*60)
    print("TEST 1: Adam momentum alignment with current gradients")
    print("─"*60)
    print("  Misura: cosine similarity tra Adam m_t e gradiente corrente")
    print("  Alta (>0.7): momentum allineato, reset potrebbe disturbare")
    print("  Bassa (<0.5): momentum stale, reset probabilmente utile\n")

    token_ids, embeddings, padding_masks = get_batch(config, data_dir, device)

    # Compute current gradient
    optimizer.zero_grad()
    loss = compute_loss(model, token_ids, embeddings, padding_masks, config)
    loss.backward()

    # Compare gradient with Adam's first moment (m_t)
    cos_sims = []
    grad_norms = []
    mom_norms = []

    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.grad is None:
                continue
            state = optimizer.state[p]
            if "exp_avg" not in state:
                continue

            g = p.grad.float().view(-1)
            m = state["exp_avg"].float().view(-1)

            if g.norm() < 1e-10 or m.norm() < 1e-10:
                continue

            cos = F.cosine_similarity(g.unsqueeze(0), m.unsqueeze(0)).item()
            cos_sims.append(cos)
            grad_norms.append(g.norm().item())
            mom_norms.append(m.norm().item())

    avg_cos = float(np.mean(cos_sims))
    median_cos = float(np.median(cos_sims))
    frac_positive = float(np.mean([c > 0 for c in cos_sims]))
    frac_high = float(np.mean([c > 0.5 for c in cos_sims]))

    print(f"  Params analizzati   : {len(cos_sims)}")
    print(f"  Cosine sim mean     : {avg_cos:.4f}")
    print(f"  Cosine sim median   : {median_cos:.4f}")
    print(f"  Frac positiva (>0)  : {frac_positive:.2%}")
    print(f"  Frac alta (>0.5)    : {frac_high:.2%}")
    print(f"  Grad norm mean      : {float(np.mean(grad_norms)):.6f}")
    print(f"  Momentum norm mean  : {float(np.mean(mom_norms)):.6f}")
    print(f"  Ratio mom/grad      : {float(np.mean(mom_norms))/float(np.mean(grad_norms)):.3f}")

    if avg_cos < 0.3:
        verdict = "⚠  MOMENTUM STALE: reset fortemente consigliato"
    elif avg_cos < 0.5:
        verdict = "~  MOMENTUM PARZIALMENTE STALE: reset probabilmente utile"
    elif avg_cos < 0.7:
        verdict = "~  MOMENTUM ACCETTABILE: reset ha effetto moderato"
    else:
        verdict = "✓  MOMENTUM ALLINEATO: reset potrebbe disturbare"

    print(f"\n  Verdetto: {verdict}")
    optimizer.zero_grad(set_to_none=True)
    return avg_cos


# ── Test 2: Loss spike con reset ──────────────────────────────────────────────

def test_reset_spike(model, optimizer, config, data_dir, device, n_steps=50):
    print("\n" + "─"*60)
    print("TEST 2: Loss spike con reset vs senza reset (50 step ciascuno)")
    print("─"*60)

    def run_steps(mdl, opt, n):
        scaler = GradScaler('cuda')
        losses = []
        for _ in range(n):
            token_ids, embeddings, padding_masks = get_batch(
                config, data_dir, device, n=config["training"]["batch_size"])
            opt.zero_grad(set_to_none=True)
            loss = compute_loss(mdl, token_ids, embeddings, padding_masks, config)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(mdl.parameters(), config["training"]["max_grad_norm"])
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
        return losses

    # Original optimizer (deep copy to not modify the real one)
    model_orig = copy.deepcopy(model)
    opt_orig = torch.optim.AdamW(
        model_orig.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"]
    )
    opt_orig.load_state_dict(copy.deepcopy(optimizer.state_dict()))

    print(f"  Running {n_steps} steps con Adam ORIGINALE...")
    losses_orig = run_steps(model_orig, opt_orig, n_steps)
    del model_orig, opt_orig

    # Fresh optimizer
    model_fresh = copy.deepcopy(model)
    opt_fresh = torch.optim.AdamW(
        model_fresh.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"]
    )

    print(f"  Running {n_steps} steps con Adam FRESCO...")
    losses_fresh = run_steps(model_fresh, opt_fresh, n_steps)
    del model_fresh, opt_fresh

    # Compare
    def stats(lst, label):
        first10 = lst[:10]
        last10 = lst[-10:]
        return (f"  {label}: "
                f"first10={np.mean(first10):.3f}  "
                f"last10={np.mean(last10):.3f}  "
                f"delta={np.mean(last10)-np.mean(first10):+.3f}")

    print(stats(losses_orig,  "Original"))
    print(stats(losses_fresh, "Fresh   "))

    orig_trend  = np.mean(losses_orig[-10:])  - np.mean(losses_orig[:10])
    fresh_trend = np.mean(losses_fresh[-10:]) - np.mean(losses_fresh[:10])
    spike = np.mean(losses_fresh[:5]) - np.mean(losses_orig[:5])

    print(f"\n  Spike iniziale (fresh - orig, primi 5 step): {spike:+.4f}")
    print(f"  Trend orig  (last10 - first10): {orig_trend:+.4f}")
    print(f"  Trend fresh (last10 - first10): {fresh_trend:+.4f}")

    if spike > 1.0:
        print("  ⚠  Spike significativo — fresh Adam parte peggio")
    elif spike > 0.3:
        print("  ~  Spike moderato — accettabile se si recupera")
    else:
        print("  ✓  Spike trascurabile — reset sicuro")

    if fresh_trend < orig_trend - 0.3:
        print("  ✓  Fresh Adam converge più velocemente di Original")
    elif abs(fresh_trend - orig_trend) < 0.3:
        print("  ~  Trend simili — reset neutro sul breve termine")
    else:
        print("  ⚠  Original converge meglio — reset dannoso")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",
                        default="checkpoints_v3_mmbert_jinav3/best.pt")
    parser.add_argument("--data-dir", default="data_mmbert_jinav3")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"\n{'═'*60}")
    print(f"  Test: dovremmo resettare l'ottimizzatore Adam?")
    print(f"{'═'*60}")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Data dir   : {args.data_dir}")

    print("\nLoading checkpoint...")
    model, optimizer, config, step, best_vl = load_checkpoint(args.checkpoint, device)
    print(f"  step={step}  best_val_loss={best_vl:.4f}")

    cos = test_momentum_alignment(model, optimizer, config, args.data_dir, device)
    test_reset_spike(model, optimizer, config, args.data_dir, device, n_steps=50)

    print(f"\n{'═'*60}")
    print(f"  CONCLUSIONE")
    print(f"{'═'*60}")
    if cos < 0.4:
        print("  Reset CONSIGLIATO: momentum stale, spike gestibile")
    elif cos < 0.6:
        print("  Reset NEUTRO: effetto moderato, decidere in base allo spike")
    else:
        print("  Reset NON CONSIGLIATO: momentum già allineato")
    print()


if __name__ == "__main__":
    main()
