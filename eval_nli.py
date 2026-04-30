import os
import argparse
import time
import requests
from data_utils import load_pairs
from inference_utils import invert_text

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
NLI_MODEL = "facebook/bart-large-mnli"
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_API_URL = f"https://router.huggingface.co/hf-inference/models/{NLI_MODEL}"

# ─────────────────────────────────────────────
# NLI LOGIC
# ─────────────────────────────────────────────

def zsc(text: str, labels: list) -> dict:
    """
    Zero-shot classification with candidate sentences as direct hypotheses
    (hypothesis_template="{}"). The model runs NLI(text, label) for each label
    and returns softmax-normalised entailment scores.

    Passing [P, N] as labels gives a relative score: which sentence does the
    reconstructed text resemble more? This is exactly what we need for flip detection.
    """
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    payload = {
        "inputs": text,
        "parameters": {"candidate_labels": labels, "hypothesis_template": "{}"},
    }
    for _ in range(4):
        resp = requests.post(HF_API_URL, headers=headers, json=payload)
        if resp.status_code == 200:
            result = resp.json()
            return {item["label"]: item["score"] for item in result}
        if resp.status_code == 503 and "estimated_time" in resp.text:
            wait = resp.json().get("estimated_time", 20)
            print(f"  Model loading, waiting {wait:.0f}s...")
            time.sleep(wait)
        else:
            print(f"  NLI API error ({resp.status_code}): {resp.text[:120]}")
            return {label: 0.0 for label in labels}
    return {label: 0.0 for label in labels}

# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def run_nli_evaluation(dataset: str, n_samples: int, use_distractors: bool):
    pairs = load_pairs(dataset, n_samples, use_distractors=use_distractors)
    n = len(pairs)

    if n == 0:
        print("No pairs found.")
        return

    print(f"\nInverting and evaluating {n} pairs via NLI (HF zero-shot, no local model)...")

    results = []

    for i, pair in enumerate(pairs):
        print(f"\n--- Pair {i+1}/{n} ---")
        p_orig = pair["p"]
        n_orig = pair["n"]

        print(f"  P:   {p_orig}")
        p_recon = invert_text(p_orig)
        print(f"  x_p: {p_recon}")

        print(f"  N:   {n_orig}")
        n_recon = invert_text(n_orig)
        print(f"  x_n: {n_recon}")

        # zsc(x_p, [P, N]): high score on P = faithful reconstruction
        scores_p = zsc(p_recon, [p_orig, n_orig])
        # zsc(x_n, [P, N]): high score on N = negation preserved; high on P = flip
        scores_n = zsc(n_recon, [p_orig, n_orig])

        print(f"  zsc(x_p): P={scores_p[p_orig]:.3f}  N={scores_p[n_orig]:.3f}")
        print(f"  zsc(x_n): P={scores_n[p_orig]:.3f}  N={scores_n[n_orig]:.3f}")

        results.append({
            "p_fidelity": scores_p[p_orig],   # x_p closer to P than N?
            "n_fidelity": scores_n[n_orig],   # x_n closer to N than P?
            "flip_score": scores_n[p_orig],   # how much x_n resembles P (= 1 - n_fidelity)
            "flipped": scores_n[p_orig] > 0.5,
        })

    avg_p_fidelity = sum(r["p_fidelity"] for r in results) / n
    avg_n_fidelity = sum(r["n_fidelity"] for r in results) / n
    avg_flip_score = sum(r["flip_score"] for r in results) / n
    flip_rate      = sum(r["flipped"]    for r in results) / n

    print(f"\n{'='*65}")
    print(f"NLI EVALUATION | dataset={dataset!r} | n={n}")
    print(f"{'='*65}")
    print(f"{'Metric':<45} {'Score':>10}")
    print(f"{'-'*65}")
    print(f"{'P-Fidelity  score(x_p, P) vs N':<45} {avg_p_fidelity:>10.4f}")
    print(f"{'N-Fidelity  score(x_n, N) vs P':<45} {avg_n_fidelity:>10.4f}")
    print(f"{'Flip Score  score(x_n, P) vs N  [CRITICAL]':<45} {avg_flip_score:>10.4f}")
    print(f"{'Flip Rate   score(x_n,P)>0.5  (%)':<45} {flip_rate*100:>9.1f}%")
    print(f"{'='*65}")
    print("\nEXPLANATION:")
    print(" - P-Fidelity:  Quanto x_p assomiglia a P rispetto a N. Atteso: alto.")
    print(" - N-Fidelity:  Quanto x_n assomiglia a N rispetto a P. Atteso: alto.")
    print(" - Flip Score:  CRITICO. Alto = il modello ha trasformato 'not X' in 'X'.")
    print(" - Flip Rate:   % di coppie in cui la negazione è andata completamente persa.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NLI-based evaluation for Embedding Inversion.")
    parser.add_argument("--dataset", choices=["this_is_not", "jina"], default="this_is_not")
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--distractors", action="store_true")
    args = parser.parse_args()

    if not HF_TOKEN:
        print("Warning: HF  _TOKEN not set. Requests to HF API may fail or be rate-limited.")

    run_nli_evaluation(args.dataset, args.n_samples, args.distractors)
