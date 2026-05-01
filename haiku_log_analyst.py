#!/usr/bin/env python3
"""
Claude Haiku log analyst for CMDLM v2 training.

Reads the last N lines from a SLURM log, sends them to Claude Haiku with
a domain-specific system prompt, and optionally forwards the analysis to
Telegram. Designed to run after each eval checkpoint from cluster_monitor.sh.

Usage:
    python3 haiku_log_analyst.py --log slurm-11108132.out --last 100 --telegram
    python3 haiku_log_analyst.py --log slurm-11108132.out --last 100

Requires:
    pip install anthropic
    Environment: ANTHROPIC_API_KEY, and BOT_TOKEN + CHAT_ID for Telegram
"""

import argparse
import json
import os
import sys
import urllib.request

SYSTEM_PROMPT = """\
You are monitoring a masked-language-model training run (CMDLM v2) targeting the results in \
paper 2602.11047v3 (Table 1, jina-embeddings-v3): 76% token accuracy and val_loss=1.60 at \
step 62,500 of 200,000 total steps.

Reference curve:
- step ~5K:  val_loss ~3.8, token_acc ~0.20
- step ~10K: val_loss ~2.8, token_acc ~0.40
- step ~20K: val_loss ~2.2, token_acc ~0.55
- step ~40K: val_loss ~1.85, token_acc ~0.65
- step ~62K: val_loss ~1.60, token_acc ~0.76  ← paper target

Training log format:
  step N/200000 | loss X | acc X | lr X | ...
  val_loss (ema): X | val_loss (raw): X [BEST]
  token_acc (EMA, 100% mask): X  [paper target: 0.760 @ step 62500]

Analyze the following training log excerpt. Respond in ≤180 tokens covering:
1. Trend (improving / stalled / diverging vs reference)
2. Any anomalies (loss spike, stuck acc, abnormal LR, OOM, errors)
3. One-sentence verdict: ON TRACK / BEHIND / CRITICAL
"""

HAIKU_MODEL = "claude-haiku-4-5-20251001"


def tail_lines(path: str, n: int) -> list[str]:
    with open(path, 'r', errors='replace') as f:
        lines = f.readlines()
    return [l.rstrip() for l in lines[-n:]]


def analyze(lines: list[str]) -> str:
    try:
        import anthropic
    except ImportError:
        print("[haiku_analyst] anthropic package not installed. Run: pip install anthropic",
              file=sys.stderr)
        sys.exit(1)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("[haiku_analyst] ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=HAIKU_MODEL,
        max_tokens=200,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": "\n".join(lines)}],
    )
    return msg.content[0].text.strip()


def send_telegram(text: str, label: str = ""):
    token = os.environ.get("BOT_TOKEN") or os.environ.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get("CHAT_ID") or os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("[telegram] BOT_TOKEN or CHAT_ID not set", file=sys.stderr)
        return
    prefix = f"🤖 Haiku analyst [{label}]\n" if label else "🤖 Haiku analyst\n"
    payload = json.dumps({"chat_id": chat_id, "text": prefix + text}).encode()
    try:
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=15)
    except Exception as e:
        print(f"[telegram] error: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Claude Haiku log analyst for training")
    parser.add_argument("--log", required=True, help="Path to SLURM log file")
    parser.add_argument("--last", type=int, default=100, help="Number of lines to analyze")
    parser.add_argument("--telegram", action="store_true", help="Forward analysis to Telegram")
    parser.add_argument("--label", default="", help="Label for Telegram message (e.g. '2-GPU #11108132')")
    args = parser.parse_args()

    if not os.path.exists(args.log):
        print(f"Log not found: {args.log}", file=sys.stderr)
        sys.exit(1)

    lines = tail_lines(args.log, args.last)
    if not lines:
        print("Log is empty", file=sys.stderr)
        sys.exit(0)

    print(f"Analyzing last {len(lines)} lines with Haiku…", flush=True)
    analysis = analyze(lines)
    print(f"\n{analysis}\n")

    if args.telegram:
        send_telegram(analysis, label=args.label)
        print("Sent to Telegram.")


if __name__ == "__main__":
    main()
