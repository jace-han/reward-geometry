"""
Build shuffled-label NULL datasets for the RQ1 C-control.

Preference signal is destroyed by randomly swapping chosen<->rejected per pair
(p=0.5, fixed seed). EVERYTHING else is preserved byte-for-byte: same texts, same
pairings, same lengths and distributions. The only thing removed is *which response
is preferred* -> a reward model trained on this CANNOT learn preference, by
construction. This is the floor that real-model alignment must beat.

(Per-pair swap, NOT random re-pairing: re-pairing would also destroy pairing
structure, confounding the null. We isolate the label.)

Produces dataset/{key}_shuffled_train.jsonl and _val.jsonl for the chosen sources.
A correctly-nulled model should score ~50% pair accuracy on the shuffled val set.

Usage:
    uv run python dataset/prepare_shuffled_ds.py
    uv run python dataset/prepare_shuffled_ds.py --sources hh_rlhf ultrafeedback
"""

import argparse
import json
import random
from pathlib import Path

DATA_DIR = Path("dataset")
SEED = 42


def shuffle_file(in_path: Path, out_path: Path, rng: random.Random) -> tuple[int, int]:
    """Copy each pair, swapping chosen/rejected with p=0.5. Returns (n_rows, n_swapped)."""
    n, swapped = 0, 0
    with in_path.open() as fin, out_path.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            if rng.random() < 0.5:
                ex["chosen"], ex["rejected"] = ex["rejected"], ex["chosen"]
                swapped += 1
            fout.write(json.dumps(ex) + "\n")
            n += 1
    return n, swapped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sources",
        nargs="+",
        default=["hh_rlhf", "ultrafeedback"],
        help="which datasets to null (C needs 2 for a shuffled<->shuffled pair)",
    )
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    for src in args.sources:
        # independent RNG stream per source+split so swaps are reproducible & decoupled
        for split in ("train", "val"):
            in_path = DATA_DIR / f"{src}_{split}.jsonl"
            out_path = DATA_DIR / f"{src}_shuffled_{split}.jsonl"
            if not in_path.exists():
                print(f"  skip {in_path} (missing)")
                continue
            rng = random.Random(f"{args.seed}-{src}-{split}")
            n, swapped = shuffle_file(in_path, out_path, rng)
            print(f"  {out_path.name}: {n} rows, {swapped} swapped ({swapped / n:.1%})")

    print("\nDone. Train reward models on *_shuffled_train.jsonl for the C null.")


if __name__ == "__main__":
    main()
