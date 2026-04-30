from datasets import load_dataset

def load_this_is_not(n: int, use_distractors: bool = False):
    """
    HiTZ/This-is-not-a-dataset (streamed, no local download).
    P = isDistractor=False AND label=True
    N = restricted by use_distractors flag.
    Matched by test_id.
    """
    mode = "with distractors" if use_distractors else "without distractors"
    print(f"Streaming 'HiTZ/This-is-not-a-dataset' ({mode})...")
    ds = load_dataset("HiTZ/This-is-not-a-dataset", streaming=True, split="test")

    positives = {}
    negatives = {}

    for ex in ds:
        tid = ex["test_id"]
        is_positive = (not ex["isDistractor"]) and (ex["label"] is True)

        if is_positive:
            if tid not in positives:
                positives[tid] = ex["sentence"]
        else:
            if not use_distractors and ex["isDistractor"]:
                continue
            if tid not in negatives:
                negatives[tid] = ex["sentence"]

        if len(set(positives) & set(negatives)) >= n:
            break

    matched_ids = sorted(set(positives) & set(negatives))[:n]
    pairs = [{"p": positives[tid], "n": negatives[tid]} for tid in matched_ids]
    print(f"  {len(pairs)} matched pairs loaded.")
    return pairs

def load_jina(n: int):
    """
    jinaai/negation-dataset (streamed, no local download).
    P = entailment, N = negative.
    """
    print("Streaming 'jinaai/negation-dataset'...")
    ds = load_dataset("jinaai/negation-dataset", split="test", streaming=True)
    samples = list(ds.take(n))
    pairs = [{"p": ex["entailment"], "n": ex["negative"]} for ex in samples]
    print(f"  {len(pairs)} pairs loaded.")
    return pairs

def load_pairs(dataset: str, n: int, use_distractors: bool = False):
    """Dispatcher for dataset loading."""
    if dataset == "this_is_not":
        return load_this_is_not(n, use_distractors=use_distractors)
    elif dataset == "jina":
        return load_jina(n)
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}. Choose 'this_is_not' or 'jina'.")
