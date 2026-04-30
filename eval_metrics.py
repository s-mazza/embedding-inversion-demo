import os
import json
import time
import argparse
import requests
import evaluate
from data_utils import load_pairs
from inference_utils import invert_text

# ─────────────────────────────────────────────
# DEFAULTS
# ─────────────────────────────────────────────
DATASET  = "this_is_not"
N_SAMPLES = 100
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_STS_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2"

# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────

def get_semantic_similarity(source: str, target: str) -> float:
    """Cosine similarity via HuggingFace STS API."""
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    payload = {"inputs": {"source_sentence": source, "sentences": [target]}}

    for _ in range(3):
        resp = requests.post(HF_STS_URL, headers=headers, json=payload)
        if resp.status_code == 200:
            return resp.json()[0]
        if "estimated_time" in resp.text:
            wait = resp.json().get("estimated_time", 20)
            print(f"  HF model loading, waiting {wait:.0f}s...")
            time.sleep(wait)
        else:
            print(f"  HF API error ({resp.status_code}): {resp.text[:80]}")
            return 0.0
    return 0.0

# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def run_evaluation(dataset: str, n_samples: int, use_distractors: bool):
    pairs = load_pairs(dataset, n_samples, use_distractors=use_distractors)
    n = len(pairs)
    if n == 0:
        print("No pairs found matching criteria.")
        return

    p_texts = [pair["p"] for pair in pairs]
    n_texts = [pair["n"] for pair in pairs]

    xp_texts, xn_texts = [], []
    print(f"\nRunning inversion on {n} P+N pairs...")
    for i, pair in enumerate(pairs):
        print(f"\n--- Pair {i+1}/{n} ---")
        print(f"  P:   {pair['p']}")
        xp = invert_text(pair["p"])
        print(f"  x_p: {xp}")
        xp_texts.append(xp)

        print(f"  N:   {pair['n']}")
        xn = invert_text(pair["n"])
        print(f"  x_n: {xn}")
        xn_texts.append(xn)

    rouge = evaluate.load("rouge")
    bleu  = evaluate.load("sacrebleu")

    rouge_p = rouge.compute(predictions=xp_texts, references=p_texts)
    bleu_p  = bleu.compute(predictions=xp_texts,  references=p_texts)
    rouge_n = rouge.compute(predictions=xn_texts, references=n_texts)
    bleu_n  = bleu.compute(predictions=xn_texts,  references=n_texts)

    print("\nComputing STS similarities via HF API...")
    sim_p = sum(get_semantic_similarity(p_texts[i], xp_texts[i]) for i in range(n)) / n
    sim_n = sum(get_semantic_similarity(n_texts[i], xn_texts[i]) for i in range(n)) / n

    rl_p = rouge_p["rougeL"]
    rl_n = rouge_n["rougeL"]
    bl_p = bleu_p["score"] / 100.0
    bl_n = bleu_n["score"] / 100.0

    dist_label = "yes" if use_distractors else "no"
    print(f"\n{'='*55}")
    print(f"RESULTS | dataset={dataset!r} | n={n} | distractors={dist_label}")
    print(f"{'='*55}")
    print(f"{'Metric':<25} {'P':>8} {'N':>8} {'Diff(P-N)':>12}")
    print(f"{'-'*55}")
    print(f"{'ROUGE-L':<25} {rl_p:>8.4f} {rl_n:>8.4f} {rl_p - rl_n:>12.4f}")
    print(f"{'BLEU':<25} {bl_p:>8.4f} {bl_n:>8.4f} {bl_p - bl_n:>12.4f}")
    print(f"{'STS Similarity (HF)':<25} {sim_p:>8.4f} {sim_n:>8.4f} {sim_p - sim_n:>12.4f}")
    print(f"{'='*55}")
    print("\nNOTE: Higher Diff = model struggles more with negations.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate embedding inversion on P vs N sentences (Overlap/STS).")
    parser.add_argument("--dataset", choices=["this_is_not", "jina"], default=DATASET)
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES, metavar="N")
    parser.add_argument("--distractors", action="store_true", default=False)
    args = parser.parse_args()
    run_evaluation(dataset=args.dataset, n_samples=args.n_samples, use_distractors=args.distractors)
