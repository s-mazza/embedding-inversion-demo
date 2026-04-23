#!/usr/bin/env python3
"""
Evaluate embedding inversion quality and compare against paper baselines.

Metrics:
  - Cosine similarity  (primary paper metric, Table 2)
  - Token accuracy     (fraction of tokens correctly reconstructed)
  - Exact match rate   (full sentence reconstruction)
  - BLEU               (sacrebleu, optional)

Paper baseline (jina-embeddings-v3, sequential greedy decoding):
  Cosine similarity : 0.715
  Token accuracy    : 0.760  (training metric on cached embeddings)

Usage:
    python eval_quality.py
    python eval_quality.py --checkpoint checkpoints_v3_mmbert_jinav3/best.pt
    python eval_quality.py --n 50 --steps 16
"""

import argparse
import json
import sys
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModel, AutoTokenizer
from model import ConditionalMDLM


# ─── Paper baselines (Table 2, jina-v3, sequential greedy) ───────────────────
PAPER_COSINE_SIM  = 0.715
PAPER_TOKEN_ACC   = 0.760  # training metric on cached embeddings

# ─── Test sentences (diverse English, not from training data) ─────────────────
TEST_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning has transformed natural language processing.",
    "The president signed the new climate bill into law yesterday.",
    "She carefully placed the fragile vase on the wooden shelf.",
    "Quantum computers could revolutionize cryptography within a decade.",
    "The stock market fell sharply after the unexpected announcement.",
    "Children learn language naturally through exposure and imitation.",
    "The ancient ruins were discovered beneath the city streets.",
    "Scientists have identified a new species of deep-sea fish.",
    "The restaurant received three Michelin stars for its innovative cuisine.",
    "Football is the most popular sport in the world by far.",
    "He refused to sign the contract without reading it carefully first.",
    "The vaccine showed ninety-five percent efficacy in clinical trials.",
    "Paris is known as the city of light and romance.",
    "The software update fixed several critical security vulnerabilities.",
    "She won the Nobel Prize in Physics for her groundbreaking research.",
    "The train was delayed by two hours due to a signal failure.",
    "Artificial intelligence cannot yet fully match human creativity.",
    "The company announced record profits despite the economic downturn.",
    "Sleep deprivation significantly impairs cognitive performance and memory.",
    "The new bridge will reduce commute times by thirty minutes.",
    "He spent three years writing his debut novel about the war.",
    "Global temperatures have risen by more than one degree since 1880.",
    "The jury found the defendant guilty on all three counts.",
    "She practiced piano for six hours every day before the competition.",
]


# ─── Model loading ────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device):
    if checkpoint_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        from safetensors import safe_open
        tensors = load_file(checkpoint_path, device=str(device))
        with safe_open(checkpoint_path, framework="pt") as f:
            meta = f.metadata()
        config = json.loads(meta["config_json"])
        model = ConditionalMDLM(config).to(device).eval()
        model.load_state_dict({k: v.float() for k, v in tensors.items()})
        print(f"  Loaded safetensors  step={meta['step']}  val_loss={meta['best_val_loss']}")
    else:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = ckpt["config"]
        model = ConditionalMDLM(config).to(device).eval()
        state = ckpt.get("ema_model") or ckpt["model"]
        state = {k.replace("module.", "").replace("_orig_mod.", ""): v.float()
                 for k, v in state.items()}
        model.load_state_dict(state)
        print(f"  Loaded checkpoint   step={ckpt['step']}  val_loss={ckpt['best_val_loss']:.4f}")
    return model, config


# ─── Encoding ────────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_jina(texts: list, jina_model, jina_tok, device: torch.device):
    inputs = jina_tok(texts, return_tensors="pt", padding=True,
                      truncation=True, max_length=512).to(device)
    out = jina_model(**inputs)
    mask = inputs["attention_mask"].unsqueeze(-1).float()
    emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    return F.normalize(emb, dim=-1)


# ─── Decoding ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def greedy_decode(embedding: torch.Tensor, model, config: dict, steps: int = 8):
    """
    Confidence-based sequential decoding (equivalent to paper's sequential greedy).
    At each step reveals the top-k most confident masked positions.
    """
    device = embedding.device
    L       = config["model"]["max_seq_len"]
    mask_id = config["model"]["mask_token_id"]
    per_step = max(1, L // steps)

    ids = torch.full((1, L), mask_id, dtype=torch.long, device=device)
    unmasked = torch.zeros(L, dtype=torch.bool, device=device)

    for _ in range(steps):
        if unmasked.all():
            break
        logits = model(ids, embedding)              # [1, L, V]
        probs = F.softmax(logits[0], dim=-1)       # [L, V]
        confidence, preds = probs.max(dim=-1)      # [L]
        confidence[unmasked] = -1.0
        k = min(per_step, (~unmasked).sum().item())
        _, topk = confidence.topk(k)
        ids[0, topk] = preds[topk]
        unmasked[topk] = True

    return ids[0]  # [L]


# ─── Metrics ──────────────────────────────────────────────────────────────────

def token_accuracy(pred_ids: torch.Tensor, orig_ids: torch.Tensor,
                   special_ids: set) -> float:
    special = torch.tensor(list(special_ids), device=orig_ids.device)
    valid = ~torch.isin(orig_ids, special)
    if valid.sum() == 0:
        return 0.0
    return (pred_ids[valid] == orig_ids[valid]).float().mean().item()


def corpus_bleu(predictions: list, references: list):
    try:
        import sacrebleu as sb
        return sb.corpus_bleu(predictions, [references]).score / 100.0
    except ImportError:
        return None


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints_v3_mmbert_jinav3/best.pt",
                        help="Path to .pt checkpoint or .safetensors file")
    parser.add_argument("--n", type=int, default=25,
                        help="Number of test sentences (max 25)")
    parser.add_argument("--steps", type=int, default=8,
                        help="Decoding steps (paper uses 8 for seq len 32)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device    = torch.device(args.device)
    sentences = TEST_SENTENCES[:min(args.n, len(TEST_SENTENCES))]

    print(f"\n{'═'*62}")
    print(f"  Embedding Inversion — Quality Evaluation")
    print(f"{'═'*62}")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Sentences  : {len(sentences)}")
    print(f"  Dec. steps : {args.steps}")
    print(f"  Device     : {device}")
    print(f"{'═'*62}\n")

    # Load inversion model
    print("Loading inversion model...")
    model, config = load_model(args.checkpoint, device)

    # Load jina-embeddings-v3
    print("Loading jina-embeddings-v3...")
    jina_tok = AutoTokenizer.from_pretrained(
        "jinaai/jina-embeddings-v3", trust_remote_code=True)
    jina_model = AutoModel.from_pretrained(
        "jinaai/jina-embeddings-v3", trust_remote_code=True).to(device).eval()

    # Load mmBERT tokenizer for decoding
    decoder_tok_name = config["model"].get("decoder_tokenizer", "jhu-clsp/mmBERT-base")
    print(f"Loading decoder tokenizer ({decoder_tok_name})...")
    dec_tok  = AutoTokenizer.from_pretrained(decoder_tok_name)
    pad_id   = dec_tok.pad_token_id or 1
    bos_id   = dec_tok.bos_token_id or 0
    mask_id  = config["model"]["mask_token_id"]
    special_ids = {pad_id, bos_id, mask_id}
    max_len  = config["model"]["max_seq_len"]

    # ── Evaluate ─────────────────────────────────────────────────────────────
    records = []
    print(f"\n{'─'*62}")
    print("  QUALITATIVE EXAMPLES (first 5)")
    print(f"{'─'*62}")

    for i, sentence in enumerate(sentences):
        # Encode original with jina
        emb = encode_jina([sentence], jina_model, jina_tok, device)  # [1, 1024]

        # Tokenize original for token accuracy
        orig_enc = dec_tok(sentence, return_tensors="pt", padding="max_length",
                           max_length=max_len, truncation=True).to(device)
        orig_ids = orig_enc["input_ids"][0]

        # Invert
        pred_ids = greedy_decode(emb, model, config, steps=args.steps)

        # Decode predicted IDs to text
        clean = [t for t in pred_ids.cpu().tolist() if t not in special_ids]
        recon = dec_tok.decode(clean, skip_special_tokens=True).strip()

        # Re-encode reconstructed text for cosine similarity
        emb_recon = encode_jina([recon], jina_model, jina_tok, device)
        cos = F.cosine_similarity(emb, emb_recon).item()

        tok_acc = token_accuracy(pred_ids, orig_ids, special_ids)
        exact   = sentence.strip().lower() == recon.lower()

        records.append({"orig": sentence, "recon": recon,
                        "cos": cos, "tok_acc": tok_acc, "exact": exact})

        if i < 5:
            print(f"\n  [{i+1}] Original : {sentence}")
            print(f"       Recon    : {recon}")
            print(f"       cos={cos:.4f}  tok_acc={tok_acc:.3f}  exact={exact}")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    cos_list   = [r["cos"]     for r in records]
    tok_list   = [r["tok_acc"] for r in records]
    exact_list = [r["exact"]   for r in records]
    originals  = [r["orig"]    for r in records]
    recons     = [r["recon"]   for r in records]

    avg_cos   = float(np.mean(cos_list))
    avg_tok   = float(np.mean(tok_list))
    exact_rate = float(np.mean(exact_list))
    bleu      = corpus_bleu(recons, originals)

    cos_gap = avg_cos - PAPER_COSINE_SIM
    tok_gap = avg_tok - PAPER_TOKEN_ACC

    print(f"\n{'═'*62}")
    print(f"  RESULTS vs PAPER BASELINE")
    print(f"{'═'*62}")
    print(f"  {'Metric':<28} {'Ours':>8}  {'Paper':>8}  {'Gap':>8}")
    print(f"  {'─'*56}")
    print(f"  {'Cosine Similarity':<28} {avg_cos:>8.4f}  {PAPER_COSINE_SIM:>8.4f}  {cos_gap:>+8.4f}")
    print(f"  {'Token Accuracy':<28} {avg_tok:>8.4f}  {PAPER_TOKEN_ACC:>8.4f}  {tok_gap:>+8.4f}")
    print(f"  {'Exact Match Rate':<28} {exact_rate:>8.4f}  {'  N/A':>8}")
    if bleu is not None:
        print(f"  {'BLEU':<28} {bleu:>8.4f}  {'  N/A':>8}")
    print(f"  {'─'*56}")
    print(f"  {'Cosine sim distribution':}")
    print(f"    min={min(cos_list):.4f}  median={float(np.median(cos_list)):.4f}"
          f"  max={max(cos_list):.4f}  std={float(np.std(cos_list)):.4f}")
    print(f"{'═'*62}")
    print(f"\n  Note: Paper = jina-v3 sequential greedy on multilingual mC4.")
    print(f"        Our model trained on English C4 only (smaller distribution).")
    if bleu is None:
        print(f"        Install sacrebleu for BLEU: pip install sacrebleu")
    print()


if __name__ == "__main__":
    main()
