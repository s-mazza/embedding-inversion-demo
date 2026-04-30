"""
Benchmark script to probe the limits of zero-shot NLI (BART-large-mnli via HF API)
for the embedding inversion evaluation use case.

Tests:
  1. Baseline          — short, clear P/N pair (expected: high confidence)
  2. Minimal negation  — only negation word differs (expected: near 50/50 risk)
  3. Long labels       — verbose sentences approaching token limits
  4. Paraphrase drift  — x_n is semantically correct but surface-form distant from N
  5. Double negation   — "not un-X" style, semantically positive but surface-negative
"""

import os
import time
import requests

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_API_URL = "https://router.huggingface.co/hf-inference/models/facebook/bart-large-mnli"


def zsc(text: str, p: str, n: str) -> dict:
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    payload = {
        "inputs": text,
        "parameters": {"candidate_labels": [p, n], "hypothesis_template": "{}"},
    }
    for _ in range(4):
        resp = requests.post(HF_API_URL, headers=headers, json=payload)
        if resp.status_code == 200:
            result = resp.json()
            return {item["label"]: item["score"] for item in result}
        if resp.status_code == 503 and "estimated_time" in resp.text:
            wait = resp.json().get("estimated_time", 20)
            print(f"  [API] Model loading, waiting {wait:.0f}s...")
            time.sleep(wait)
        else:
            print(f"  [API] Error {resp.status_code}: {resp.text[:120]}")
            return {p: 0.0, n: 0.0}
    return {p: 0.0, n: 0.0}


def run(label: str, text: str, p: str, n: str, expect_p: bool):
    scores = zsc(text, p, n)
    score_p = scores[p]
    score_n = scores[n]
    predicted_p = score_p > 0.5
    correct = predicted_p == expect_p
    margin = abs(score_p - score_n)

    status = "OK" if correct else "FLIP"
    confidence = "high" if margin > 0.4 else ("medium" if margin > 0.15 else "LOW")

    print(f"\n[{label}]")
    print(f"  text : {text[:90]}")
    print(f"  P    : {p[:90]}")
    print(f"  N    : {n[:90]}")
    print(f"  scores  → P: {score_p:.3f}  N: {score_n:.3f}  (margin: {margin:.3f})")
    print(f"  result  → {status} | confidence: {confidence} | expected P={expect_p}, got P={predicted_p}")
    return {"label": label, "correct": correct, "margin": margin, "confidence": confidence}


# ─────────────────────────────────────────────────────────────────────────────

results = []

# 1. Baseline — clear, short, unambiguous
results.append(run(
    label="1. Baseline (clear)",
    text="The sun rises in the east every morning.",
    p="The sun rises in the east.",
    n="The sun does not rise in the east.",
    expect_p=True,
))

# 2. Minimal negation — only "not" differs; risk of near 50/50
results.append(run(
    label="2. Minimal negation (only 'not' differs)",
    text="Water boils at 100 degrees Celsius.",
    p="Water boils at 100 degrees Celsius.",
    n="Water does not boil at 100 degrees Celsius.",
    expect_p=True,
))

# Same but text matches the negated version
results.append(run(
    label="2b. Minimal negation (text matches N)",
    text="Water does not boil at 100 degrees Celsius.",
    p="Water boils at 100 degrees Celsius.",
    n="Water does not boil at 100 degrees Celsius.",
    expect_p=False,
))

# 3. Long labels — verbose P/N close to model token limits
long_p = ("Ice cream is a frozen dairy dessert that is commonly made from milk, cream, "
          "sugar, and various flavorings such as vanilla, chocolate, or fruit, and it "
          "typically melts when exposed to heat or sunlight for an extended period of time.")
long_n = ("Ice cream is a frozen dairy dessert that is commonly made from milk, cream, "
          "sugar, and various flavorings such as vanilla, chocolate, or fruit, and it "
          "does not typically melt when exposed to heat or sunlight for an extended period of time.")

results.append(run(
    label="3. Long labels (verbose sentences)",
    text="Ice cream melts in the sun because it is made of dairy and sugar.",
    p=long_p,
    n=long_n,
    expect_p=True,
))

# 4. Paraphrase drift — x_n is semantically correct but worded very differently from N
# x_n is a paraphrase of N but doesn't share surface form
results.append(run(
    label="4. Paraphrase drift (x_n ≈ N but different words)",
    text="The sky lacks any blue coloration and appears colorless to the observer.",  # paraphrase of "sky is not blue"
    p="The sky is blue.",
    n="The sky is not blue.",
    expect_p=False,
))

# 5. Double negation — surface looks negative but is semantically positive
results.append(run(
    label="5. Double negation ('not un-X' = positive)",
    text="The result was not unsuccessful.",   # = the result was successful
    p="The result was successful.",
    n="The result was not successful.",
    expect_p=True,
))

# 6. Subtle context flip — embedding inversion realistic failure case
# Simulates x_n that the inversion model produced with "not" still present but meaning flipped
results.append(run(
    label="6. Realistic inversion failure (negation word present but meaning flipped)",
    text="Ice cream does not resist melting; it melts under the sun.",  # says it melts despite "not"
    p="Ice cream melts under the sun.",
    n="Ice cream does not melt under the sun.",
    expect_p=True,
))

# ─────────────────────────────────────────────────────────────────────────────
# Summary

print(f"\n{'='*65}")
print(f"SUMMARY")
print(f"{'='*65}")
print(f"{'Test':<45} {'Correct':>8} {'Margin':>8} {'Confidence':>12}")
print(f"{'-'*65}")
for r in results:
    print(f"{r['label']:<45} {'YES' if r['correct'] else 'NO':>8} {r['margin']:>8.3f} {r['confidence']:>12}")

n_correct = sum(r["correct"] for r in results)
low_conf  = sum(r["confidence"] == "LOW" for r in results)
print(f"{'-'*65}")
print(f"Correct: {n_correct}/{len(results)}  |  Low-confidence results: {low_conf}/{len(results)}")
print(f"{'='*65}")
print("\nNOTE: 'LOW' confidence (margin < 0.15) means the metric is unreliable for that case.")
