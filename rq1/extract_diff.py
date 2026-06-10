"""Forward-pass chosen+rejected from a held-out probe through one model, take the
full-chat last-non-pad-token hidden state per layer, and store the per-pair DIFF
d = h(chosen) - h(rejected) as a [L+1, N, H] array.

The diff cloud is the primary RQ1 object: Bradley-Terry training optimized
r(chosen) - r(rejected) = w . (h(chosen) - h(rejected)), so the diff is the exact
coordinate the reward head was trained in. Diff arrays have the SAME shape as the
neutral reps, so rq1/compare.py CKA/mutual-kNN consume them unchanged.

Usage:
    uv run python -m rq1.extract_diff --probe shp --model hh
    uv run python -m rq1.extract_diff --all
"""

import argparse
import json

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from rq1 import config, heldout_config as hc

def _empty_cache(device: str):
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()

def last_nonpad_indices(attention_mask) -> np.ndarray:
    """Index of the last real (non-pad) token per row, assuming right padding.

    attention_mask: [batch, seq] with 1 for real tokens, 0 for padding.
    Returns an int array [batch].
    """
    mask = np.asarray(attention_mask)
    lengths = mask.sum(axis=1)  # number of real tokens per row
    return (lengths - 1).astype(int)


def select_last_token(hidden, attention_mask) -> np.ndarray:
    """Pick the last-non-pad-token vector from each row.

    hidden: [batch, seq, H]; attention_mask: [batch, seq].
    Returns [batch, H].
    """
    hidden = np.asarray(hidden)
    idx = last_nonpad_indices(attention_mask)
    rows = np.arange(hidden.shape[0])
    return hidden[rows, idx, :]

def diff_reps(chosen: np.ndarray, rejected: np.ndarray) -> np.ndarray:
    """[L+1, N, H] chosen and rejected -> [L+1, N, H] difference (chosen-rejected)."""
    return (np.asarray(chosen) - np.asarray(rejected)).astype(np.float32)


def flatten_pairs(pairs):
    """Pairs -> fixed-order flat records, interleaved chosen,rejected per pair.

    Fixed order guarantees every model sees identical inputs index-for-index.
    Returns list of {messages, kind, pair_id}.
    """
    flat = []
    for pid, ex in enumerate(pairs):
        for kind in ("chosen", "rejected"):
            flat.append({"messages": ex[kind], "kind": kind, "pair_id": pid})
    return flat


def split_chosen_rejected(flat_reps: np.ndarray):
    """[L+1, 2*N, H] interleaved (chosen,rejected,...) -> (chosen, rejected) each [L+1,N,H]."""
    chosen = flat_reps[:, 0::2, :]
    rejected = flat_reps[:, 1::2, :]
    return chosen, rejected


def _load_pairs(probe: str):
    pairs = []
    with hc.probe_jsonl(probe).open() as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs[: hc.N_PAIRS]

def extract_flat(model_name: str, flat, batch_size: int = 16, device: str = "cpu") -> np.ndarray:
    """Forward-pass every flat record; return last-token reps [L+1, 2N, H] float32."""
    model_path = config.MODELS[model_name]

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    model = (
        AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=1, torch_dtype=torch.bfloat16,
            output_hidden_states=True,
        ).to(device).eval()
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    per_layer_chunks = None
    for start in range(0, len(flat), batch_size):
        batch = flat[start : start + batch_size]
        texts = [
            tokenizer.apply_chat_template(r["messages"], tokenize=False,
                                          add_generation_prompt=False)
            for r in batch
        ]
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                        max_length=hc.MAX_LENGTH).to(device)
        with torch.no_grad():
            out = model(**enc)
        hidden_states = out.hidden_states
        mask = enc["attention_mask"].cpu().numpy()
        if per_layer_chunks is None:
            per_layer_chunks = [[] for _ in range(len(hidden_states))]
        for li, hs in enumerate(hidden_states):
            hs_np = hs.float().cpu().numpy()
            per_layer_chunks[li].append(select_last_token(hs_np, mask))
        del out, hidden_states
        _empty_cache(device)

    flat_reps = np.stack(
        [np.concatenate(chunks, axis=0) for chunks in per_layer_chunks], axis=0
    )
    del model
    _empty_cache(device)
    return flat_reps.astype(np.float32)


def save_diff(probe: str, model_name: str, batch_size: int = 16, device: str = "cpu"):
    config.REPR_DIR.mkdir(parents=True, exist_ok=True)
    out_path = hc.diff_path(probe, model_name)
    if out_path.exists():
        print(f"{probe}/{model_name}: already exists at {out_path}, skipping")
        return
    pairs = _load_pairs(probe)
    flat = flatten_pairs(pairs)
    flat_reps = extract_flat(model_name, flat, batch_size=batch_size, device=device)
    chosen, rejected = split_chosen_rejected(flat_reps)
    d = diff_reps(chosen, rejected)
    np.save(out_path, d)
    print(f"{probe}/{model_name}: diff {d.shape} -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", choices=list(hc.PROBES))
    ap.add_argument("--model", choices=list(config.MODELS))
    ap.add_argument("--all", action="store_true", help="every probe x every model")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Extracting on {device}")

    if args.all:
        jobs = [(p, m) for p in hc.PROBES for m in config.MODELS]
    elif args.probe and args.model:
        jobs = [(args.probe, args.model)]
    else:
        ap.error("specify --probe + --model, or --all")
    print(f"jobs: {jobs}")
    for probe, model_name in jobs:
        print(f"\nExtracting {probe} x {model_name}...")
        save_diff(probe, model_name, batch_size=args.batch_size, device=device)
