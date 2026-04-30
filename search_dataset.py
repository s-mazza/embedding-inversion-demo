"""
Search sentences in HiTZ/This-is-not-a-dataset by keywords.
Prints matching P/N pairs.

Usage:
  python3 search_dataset.py ice cream
  python3 search_dataset.py --any ice cream        # match either P or N
  python3 search_dataset.py --limit 5 water boil
"""

import argparse
from datasets import load_dataset


def search(keywords: list[str], match_any: bool, limit: int):
    ds = load_dataset("HiTZ/This-is-not-a-dataset", streaming=True, split="test")

    positives = {}
    negatives = {}

    for ex in ds:
        tid = ex["test_id"]
        sentence = ex["sentence"].lower()
        is_positive = (not ex["isDistractor"]) and (ex["label"] is True)

        if is_positive:
            positives[tid] = ex["sentence"]
        else:
            if not ex["isDistractor"]:
                negatives[tid] = ex["sentence"]

    matched = []
    for tid in set(positives) & set(negatives):
        p = positives[tid]
        n = negatives[tid]

        hit_p = all(kw.lower() in p.lower() for kw in keywords)
        hit_n = all(kw.lower() in n.lower() for kw in keywords)

        if match_any and (hit_p or hit_n):
            matched.append((tid, p, n))
        elif not match_any and (hit_p and hit_n):
            matched.append((tid, p, n))

        if len(matched) >= limit:
            break

    if not matched:
        print(f"No matches found for keywords: {keywords}")
        return

    print(f"\nFound {len(matched)} match(es) for {keywords}:\n")
    for tid, p, n in matched:
        print(f"  [id={tid}]")
        print(f"  P: {p}")
        print(f"  N: {n}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("keywords", nargs="+", help="Keywords to search for")
    parser.add_argument("--any", dest="match_any", action="store_true",
                        help="Match if ANY sentence contains all keywords (default: both must match)")
    parser.add_argument("--limit", type=int, default=10, metavar="N",
                        help="Max results to return (default: 10)")
    args = parser.parse_args()
    search(args.keywords, args.match_any, args.limit)
