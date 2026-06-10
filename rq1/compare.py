"""Compute per-layer CKA and mutual k-NN (with bootstrap CIs) for every model
pair, and write a tidy parquet table.

Run one metric at a time and merge into the shared results table:
    uv run python -m rq1.compare --metric cka
    uv run python -m rq1.compare --metric knn
or both at once (default):
    uv run python -m rq1.compare --metric both
"""

import argparse

import numpy as np
import pandas as pd

from rq1 import config, metrics

METRICS = ("cka", "knn")


def _subsample(n: int, k_sub: int, seed: int) -> np.ndarray:
    if k_sub >= n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return rng.choice(n, size=k_sub, replace=False)


def compute_pair_rows(pair, reps, metric, k_values, knn_subsample, n_boot, seed):
    """Return a list of row dicts for one model pair across all layers.

    reps: dict logical_name -> array [L+1, N, H].
    metric: "cka", "knn", or "both" — which similarity metric(s) to compute.
    CKA uses a Gram-matrix bootstrap (subsampled rows); mutual k-NN uses a fixed
    knn_subsample of examples (for Platonic-comparability) shared across layers.
    """
    do_cka = metric in ("cka", "both")
    do_knn = metric in ("knn", "both")
    A = reps[pair["a"]]
    B = reps[pair["b"]]
    n_layers, n, _ = A.shape
    sub = _subsample(n, knn_subsample, seed)
    rows = []
    for layer in range(n_layers):
        X, Y = A[layer], B[layer]
        if do_cka:
            # CKA via fast Gram-matrix bootstrap (subsampled rows)
            mean, lo, hi = metrics.cka_bootstrap(X, Y, n_boot=n_boot, seed=seed)
            rows.append(
                {
                    "layer": layer,
                    "metric": "cka",
                    "k": None,
                    "pair": f"{pair['a']}__{pair['b']}",
                    "pair_type": pair["pair_type"],
                    "score": mean,
                    "ci_lo": lo,
                    "ci_hi": hi,
                }
            )
        if do_knn:
            # mutual k-NN on the subsample; all k share one matmul per resample
            Xs, Ys = X[sub], Y[sub]
            knn = metrics.knn_bootstrap(Xs, Ys, k_values, n_boot=n_boot, seed=seed)
            for k in k_values:
                mean, lo, hi = knn[k]
                rows.append(
                    {
                        "layer": layer,
                        "metric": "knn",
                        "k": k,
                        "pair": f"{pair['a']}__{pair['b']}",
                        "pair_type": pair["pair_type"],
                        "score": mean,
                        "ci_lo": lo,
                        "ci_hi": hi,
                    }
                )
    return rows


def load_all_reps():
    return {name: np.load(config.REPR_DIR / f"{name}.npy") for name in config.MODELS}

def _load_existing(metric):
    """Existing results minus any rows for the metric(s) we're recomputing.

    Lets a `--metric knn` run merge into a table already holding `cka` rows
    (and vice versa) instead of clobbering it.
    """
    if not config.RESULTS_PATH.exists():
        return pd.DataFrame()
    df = pd.read_parquet(config.RESULTS_PATH)
    drop = METRICS if metric == "both" else (metric,)
    return df[~df["metric"].isin(drop)].copy()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metric",
        choices=(*METRICS, "both"),
        default="both",
        help="Which metric to compute. cka/knn merge into the shared table.",
    )
    parser.add_argument("--n-boot", type=int, default=200)
    args = parser.parse_args()

    config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    reps = load_all_reps()
    kept = _load_existing(args.metric)
    new_rows = []
    for pair in config.PAIRS:
        new_rows.extend(
            compute_pair_rows(
                pair,
                reps,
                metric=args.metric,
                k_values=config.K_VALUES,
                knn_subsample=config.KNN_SUBSAMPLE,
                n_boot=args.n_boot,
                seed=config.SEED,
            )
        )
        # checkpoint after each pair so progress survives interruption
        merged = pd.concat([kept, pd.DataFrame(new_rows)], ignore_index=True)
        merged.to_parquet(config.RESULTS_PATH, index=False)
    print(
        f"Wrote {len(new_rows)} {args.metric} rows "
        f"({len(kept)} existing kept) -> {config.RESULTS_PATH}"
    )
