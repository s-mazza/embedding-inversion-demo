#!/usr/bin/env python3
"""
Diagnostic evaluation for embedding inversion model.

Tests:
  1. Checkpoint integrity   — model loads and produces valid logits
  2. In-distribution        — accuracy on training dataset samples (should be high)
  3. Logit confidence       — entropy of predictions (low = confident)
  4. Decoding step ablation — quality vs number of decoding steps
  5. Val-loss recomputation — verify the reported 2.6825 is real

Usage (inside Docker on faretra):
    python eval_diagnostic.py --checkpoint checkpoints_v3_mmbert_jinav3/best.pt
    python eval_diagnostic.py --checkpoint checkpoints_v3_mmbert_jinav3/best.pt --n-train 50
"""

import argparse
import json
import math
import os
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModel, AutoTokenizer

from model import ConditionalMDLM, apply_mask
from dataset import EmbeddingInversionDataset


# ─── Loading ──────────────────────────────────────────────────────────────────

def load_model(checkpoint_path, device):
    if checkpoint_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        from safetensors import safe_open
        tensors = load_file(checkpoint_path, device=str(device))
        with safe_open(checkpoint_path, framework="pt") as f:
            meta = f.metadata()
        config = json.loads(meta["config_json"])
        model = ConditionalMDLM(config).to(device).eval()
        model.load_state_dict({k: v.float() for k, v in tensors.items()})
        step = meta["step"]; vl = meta["best_val_loss"]
    else:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = ckpt["config"]
        model = ConditionalMDLM(config).to(device).eval()
        state = ckpt.get("ema_model") or ckpt["model"]
        state = {k.replace("module.", "").replace("_orig_mod.", ""): v.float()
                 for k, v in state.items()}
        model.load_state_dict(state)
        step = ckpt["step"]; vl = f"{ckpt['best_val_loss']:.4f}"
    print(f"  checkpoint: step={step}  saved_val_loss={vl}")
    return model, config


# ─── Test 1: Checkpoint integrity ─────────────────────────────────────────────

def test_checkpoint_integrity(model, config, device):
    print("\n" + "─"*60)
    print("TEST 1: Checkpoint integrity")
    print("─"*60)
    mc = config["model"]
    B, L = 2, mc["max_seq_len"]
    mask_id = mc["mask_token_id"]

    ids = torch.randint(0, mc["vocab_size"]-1, (B, L), device=device)
    emb = torch.randn(B, mc["embedding_cond_dim"], device=device)
    emb = F.normalize(emb, dim=-1)

    with torch.no_grad():
        logits = model(ids, emb)

    assert logits.shape == (B, L, mc["vocab_size"]), \
        f"Wrong output shape: {logits.shape}"

    # Check logits are finite and not all identical (collapsed)
    finite = logits.isfinite().all().item()
    collapsed = (logits.std(dim=-1) < 1e-6).any().item()
    mean_entropy = -(F.softmax(logits, dim=-1) *
                     F.log_softmax(logits, dim=-1)).sum(-1).mean().item()
    max_logit = logits.max().item()
    min_logit = logits.min().item()

    print(f"  Output shape   : {tuple(logits.shape)}  ✓")
    print(f"  All finite     : {finite}")
    print(f"  Collapsed      : {collapsed}  (True = bad)")
    print(f"  Mean entropy   : {mean_entropy:.3f}  (log({mc['vocab_size']})={math.log(mc['vocab_size']):.2f} = max)")
    print(f"  Logit range    : [{min_logit:.2f}, {max_logit:.2f}]")

    if mean_entropy > math.log(mc["vocab_size"]) * 0.95:
        print("  ⚠ WARN: entropy near max — model is nearly uniform (not converged or loading issue)")
    else:
        print("  ✓ Model produces peaked predictions")
    return {"finite": finite, "collapsed": collapsed, "entropy": mean_entropy}


# ─── Test 2: In-distribution accuracy ─────────────────────────────────────────

def test_in_distribution(model, config, device, data_dir, n_samples=50):
    print("\n" + "─"*60)
    print("TEST 2: In-distribution accuracy (training data)")
    print("─"*60)
    mc = config["model"]
    mask_id = mc["mask_token_id"]

    # Load dataset
    try:
        ds = EmbeddingInversionDataset(
            data_dir, mc["max_seq_len"],
            val=False, val_split=0.01,
            pad_token_id=1, bos_token_id=None
        )
    except Exception as e:
        print(f"  ✗ Could not load dataset from {data_dir}: {e}")
        return None

    indices = torch.randperm(len(ds))[:n_samples]
    accs_full = []    # accuracy with full masking (hardest)
    accs_half = []    # accuracy with 50% masking
    val_losses = []

    with torch.no_grad():
        for idx in indices.tolist():
            item = ds[idx]
            token_ids = item["token_ids"].unsqueeze(0).to(device)
            embedding  = item["embedding"].unsqueeze(0).to(device)
            padding_mask = item["padding_mask"].unsqueeze(0).to(device)

            # Full masking (100%) — hardest case
            masked_ids_full = token_ids.clone()
            non_pad = ~padding_mask[0]
            masked_ids_full[0, non_pad] = mask_id
            target_mask_full = non_pad.unsqueeze(0)

            logits_full = model(masked_ids_full, embedding)
            preds_full = logits_full.argmax(-1)
            m = target_mask_full[0]
            if m.sum() > 0:
                acc = (preds_full[0][m] == token_ids[0][m]).float().mean().item()
                accs_full.append(acc)

                # val_loss (raw CE on masked positions, no 1/t weighting)
                ce = F.cross_entropy(logits_full[0][m], token_ids[0][m]).item()
                val_losses.append(ce)

            # 50% masking — easier
            masked_ids_half, target_mask_half, _, _ = apply_mask(
                token_ids, mask_id, padding_mask)
            logits_half = model(masked_ids_half, embedding)
            preds_half = logits_half.argmax(-1)
            m2 = target_mask_half[0]
            if m2.sum() > 0:
                acc2 = (preds_half[0][m2] == token_ids[0][m2]).float().mean().item()
                accs_half.append(acc2)

    avg_full = float(np.mean(accs_full)) if accs_full else 0
    avg_half = float(np.mean(accs_half)) if accs_half else 0
    avg_vl   = float(np.mean(val_losses)) if val_losses else 0

    print(f"  Samples tested         : {len(accs_full)}")
    print(f"  Token acc (100% mask)  : {avg_full:.4f}  (paper mmBERT: 0.10-0.30)")
    print(f"  Token acc (50% mask)   : {avg_half:.4f}  (easier — more context)")
    print(f"  Val-loss (recomputed)  : {avg_vl:.4f}  (checkpoint says: 2.6825)")

    if abs(avg_vl - 2.6825) > 0.5:
        print(f"  ⚠ WARN: recomputed val_loss differs significantly from saved value")
    else:
        print(f"  ✓ Val-loss matches saved checkpoint value")

    return {"acc_full": avg_full, "acc_half": avg_half, "val_loss": avg_vl}


# ─── Test 3: Logit confidence analysis ────────────────────────────────────────

def test_logit_confidence(model, config, device, data_dir):
    print("\n" + "─"*60)
    print("TEST 3: Logit confidence (entropy analysis)")
    print("─"*60)
    mc = config["model"]
    mask_id = mc["mask_token_id"]
    max_entropy = math.log(mc["vocab_size"])

    try:
        ds = EmbeddingInversionDataset(
            data_dir, mc["max_seq_len"],
            val=False, val_split=0.01,
            pad_token_id=1, bos_token_id=None
        )
    except Exception as e:
        print(f"  ✗ Could not load dataset: {e}")
        return

    entropies, top1_probs, top5_probs = [], [], []

    with torch.no_grad():
        for idx in torch.randperm(len(ds))[:20].tolist():
            item = ds[idx]
            token_ids   = item["token_ids"].unsqueeze(0).to(device)
            embedding   = item["embedding"].unsqueeze(0).to(device)
            padding_mask = item["padding_mask"].unsqueeze(0).to(device)

            # Full masking
            masked = token_ids.clone()
            non_pad = ~padding_mask[0]
            masked[0, non_pad] = mask_id

            logits = model(masked, embedding)  # [1, L, V]
            probs  = F.softmax(logits[0], dim=-1)  # [L, V]

            # Only non-padding positions
            p = probs[non_pad]
            ent = -(p * p.log().clamp(min=-100)).sum(-1)
            entropies.extend(ent.cpu().tolist())
            top1_probs.extend(p.max(-1).values.cpu().tolist())
            top5_probs.extend(p.topk(5, dim=-1).values.sum(-1).cpu().tolist())

    avg_ent  = float(np.mean(entropies))
    avg_top1 = float(np.mean(top1_probs))
    avg_top5 = float(np.mean(top5_probs))

    print(f"  Mean entropy           : {avg_ent:.3f}  / {max_entropy:.3f} ({100*avg_ent/max_entropy:.1f}% of max)")
    print(f"  Mean top-1 probability : {avg_top1:.4f}  (1.0 = perfectly confident)")
    print(f"  Mean top-5 probability : {avg_top5:.4f}")

    if avg_top1 < 0.05:
        print("  ⚠ WARN: model very uncertain — predictions barely above chance")
    elif avg_top1 < 0.20:
        print("  ~ Model moderately uncertain — partially converged")
    else:
        print("  ✓ Model shows clear preferences in predictions")


# ─── Test 4: Decoding step ablation ───────────────────────────────────────────

def test_decoding_steps(model, config, device):
    print("\n" + "─"*60)
    print("TEST 4: Decoding quality vs number of steps")
    print("─"*60)

    mc = config["model"]
    mask_id = mc["mask_token_id"]
    jina_ok = False

    try:
        jina_tok = AutoTokenizer.from_pretrained(
            "jinaai/jina-embeddings-v3", trust_remote_code=True)
        jina_model = AutoModel.from_pretrained(
            "jinaai/jina-embeddings-v3", trust_remote_code=True).to(device).eval()
        jina_ok = True
    except Exception as e:
        print(f"  ✗ Could not load jina: {e}")
        return

    dec_tok = AutoTokenizer.from_pretrained(
        config["model"].get("decoder_tokenizer", "jhu-clsp/mmBERT-base"))
    special_ids = {dec_tok.pad_token_id or 1,
                   dec_tok.bos_token_id or 0, mask_id}

    sentence = "The European Union established new trade regulations this month."

    # Encode
    inputs = jina_tok([sentence], return_tensors="pt", padding=True,
                      truncation=True, max_length=512).to(device)
    with torch.no_grad():
        out = jina_model(**inputs)
    mask = inputs["attention_mask"].unsqueeze(-1).float()
    emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    emb = F.normalize(emb, dim=-1)

    print(f"  Input: {sentence}\n")
    print(f"  {'Steps':>6}  {'Cosine':>8}  Reconstruction")
    print(f"  {'─'*55}")

    for steps in [1, 2, 4, 8, 16, 32]:
        L = mc["max_seq_len"]
        per_step = max(1, L // steps)
        ids = torch.full((1, L), mask_id, dtype=torch.long, device=device)
        unmasked = torch.zeros(L, dtype=torch.bool, device=device)

        with torch.no_grad():
            for _ in range(steps):
                if unmasked.all():
                    break
                logits = model(ids, emb)
                probs = F.softmax(logits[0], dim=-1)
                conf, preds = probs.max(dim=-1)
                conf[unmasked] = -1.0
                k = min(per_step, (~unmasked).sum().item())
                _, topk = conf.topk(k)
                ids[0, topk] = preds[topk]
                unmasked[topk] = True

        clean = [t for t in ids[0].cpu().tolist() if t not in special_ids]
        recon = dec_tok.decode(clean, skip_special_tokens=True).strip()

        # Cosine sim
        inputs2 = jina_tok([recon or "."], return_tensors="pt", padding=True,
                            truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out2 = jina_model(**inputs2)
        mask2 = inputs2["attention_mask"].unsqueeze(-1).float()
        emb2 = (out2.last_hidden_state * mask2).sum(1) / mask2.sum(1).clamp(min=1e-9)
        emb2 = F.normalize(emb2, dim=-1)
        cos = F.cosine_similarity(emb, emb2).item()

        print(f"  {steps:>6}  {cos:>8.4f}  {recon[:60]}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",
                        default="checkpoints_v3_mmbert_jinav3/best.pt")
    parser.add_argument("--data-dir",
                        default="data_mmbert_jinav3",
                        help="Training data directory (for in-distribution tests)")
    parser.add_argument("--n-train", type=int, default=50,
                        help="Samples for in-distribution test")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"\n{'═'*60}")
    print(f"  Embedding Inversion — Diagnostic Evaluation")
    print(f"{'═'*60}")
    print(f"  checkpoint : {args.checkpoint}")
    print(f"  data_dir   : {args.data_dir}")
    print(f"  device     : {device}")

    model, config = load_model(args.checkpoint, device)

    r1 = test_checkpoint_integrity(model, config, device)
    r2 = test_in_distribution(model, config, device, args.data_dir, args.n_train)
    test_logit_confidence(model, config, device, args.data_dir)
    test_decoding_steps(model, config, device)

    print(f"\n{'═'*60}")
    print("  SUMMARY")
    print(f"{'═'*60}")
    if r2:
        print(f"  In-distrib acc (100% mask) : {r2['acc_full']:.4f}")
        print(f"  In-distrib acc (50% mask)  : {r2['acc_half']:.4f}")
        print(f"  Val-loss (recomputed)      : {r2['val_loss']:.4f}")
    print(f"  Logit entropy              : {r1['entropy']:.3f}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
