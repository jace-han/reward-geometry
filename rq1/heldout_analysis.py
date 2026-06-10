"""RQ1 held-out reward-axis analysis.

For each probe, load the per-model diff arrays [L+1, N, H] and compute:
  - real<->real diff alignment (convergence signal)
  - shuffled<->shuffled diff alignment (noise floor)
  - base<->real diff distance (inheritance floor)
  - base<->shuffled diff distance (gradient-descent geometric noise floor)
  - tilt diagnostic: per real model, base->model diff distance; an outlier
    model means the probe is home-field for it.

Diff arrays are fed to the existing compare.compute_pair_rows unchanged.

Usage:
    uv run python -m rq1.heldout_analysis
    uv run python -m rq1.heldout_analysis --n-boot 200 --no-plot
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rq1 import compare, config, heldout_config as hc, metrics
import plotly.subplots as sp
import plotly.graph_objects as go

def _build_heldout_pairs():
    """All pair comparisons for held-out analysis, including base<->shuffled."""
    pairs = list(config.PAIRS)
    for s in config.SHUFFLED:
        if not any(p["a"] == "base" and p["b"] == s for p in pairs):
            pairs.append({"a": "base", "b": s, "pair_type": "base_shuf"})
    return pairs


HELDOUT_PAIRS = _build_heldout_pairs()

def flag_tilt(dists: dict, z_thresh: float = 1.5):
    """Flag models whose mean base-distance is a high-side outlier.

    dists: model -> per-layer base-distance curve.
    A probe equidistant from all training distributions yields flat means.
    A model the probe is home-field for moves further from base -> flagged.
    """
    means = {m: float(np.nanmean(c)) for m, c in dists.items()}
    vals = np.array(list(means.values()))
    mu, sd = vals.mean(), vals.std()
    if sd == 0:
        return []
    return [m for m, v in means.items() if (v - mu) / sd > z_thresh]


def _load(probe, model):
    return np.load(hc.diff_path(probe, model))


def _get_completion_file():
    return config.ARTIFACT_DIR / "heldout_completion.json"


def _get_results_file():
    return config.ARTIFACT_DIR / "heldout_results.jsonl"


def _load_completion():
    """Load set of completed (probe, pair_idx) tuples."""
    cf = _get_completion_file()
    if cf.exists():
        with open(cf) as f:
            return set(tuple(x) for x in json.load(f))
    return set()


def _save_completion(completed):
    """Save set of (probe, pair_idx) tuples to JSON."""
    cf = _get_completion_file()
    with open(cf, "w") as f:
        json.dump([list(x) for x in completed], f)


def _append_rows(rows, results_file):
    """Append rows to JSONL file."""
    if rows:
        with open(results_file, "a") as f:
            for row in rows:
                json.dump(row, f)
                f.write("\n")


def _base_distance_curves(probe, n_boot):
    """real model -> per-layer (1 - CKA(base_diff, model_diff))."""
    base = _load(probe, "base")
    out = {}
    for m in config.REAL:
        md = _load(probe, m)
        n_layers = base.shape[0]
        curve = np.empty(n_layers)
        for layer in range(n_layers):
            mean, _, _ = metrics.cka_bootstrap(
                base[layer], md[layer], n_boot=n_boot, seed=config.SEED
            )
            curve[layer] = 1.0 - mean
        out[m] = curve
    return out


def analyze_probe(probe, n_boot, results_file, completed):
    reps = {m: _load(probe, m) for m in config.MODELS}
    tilt_summary = {}
    for pair_idx, pair in enumerate(HELDOUT_PAIRS):
        if (probe, pair_idx) in completed:
            print(f"  [{probe}] {pair['a']}-{pair['b']} ({pair.get('pair_type', 'unknown')}): SKIPPED (already done)")
            continue

        pair_rows = []
        for r in compare.compute_pair_rows(
            pair, reps, metric="both", k_values=config.K_VALUES,
            knn_subsample=config.KNN_SUBSAMPLE, n_boot=n_boot, seed=config.SEED,
        ):
            r["probe"] = probe
            pair_rows.append(r)

        completed.add((probe, pair_idx))
        _save_completion(completed)
        _append_rows(pair_rows, results_file)
        print(f"  [{probe}] {pair['a']}-{pair['b']} ({pair.get('pair_type', 'unknown')}): {len(pair_rows)} rows")

    base_dists = _base_distance_curves(probe, n_boot)
    flagged = flag_tilt(base_dists)
    tilt_summary[probe] = flagged
    return tilt_summary

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    config.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    results_file = _get_results_file()
    completed = _load_completion()
    tilt_summary = {}

    for probe in hc.PROBES:
        print(f"\nAnalysing probe {probe}...")
        probe_tilt = analyze_probe(probe, args.n_boot, results_file, completed)
        tilt_summary.update(probe_tilt)

    if results_file.exists() and results_file.stat().st_size > 0:
        out = pd.read_json(results_file, lines=True)
        out = out.drop_duplicates(subset=["probe", "pair_type", "layer", "metric", "k"], keep="first")
    else:
        out = pd.DataFrame()
    out_path = config.ARTIFACT_DIR / "heldout_results.parquet"
    out.to_parquet(out_path, index=False)
    print(f"\nwrote {len(out)} rows -> {out_path}")
    print(f"tilt summary: {tilt_summary}")

    if not args.no_plot:
        probes = list(hc.PROBES)
        fig = sp.make_subplots(
            rows=1, cols=len(probes),
            subplot_titles=[f"{p}: diff-CKA convergence" for p in probes],
            horizontal_spacing=0.08
        )

        for col, probe in enumerate(probes, start=1):
            sub = out[(out.probe == probe) & (out.metric == "cka")]
            for ptype, label in (("real_real", "real↔real"),
                                 ("shuf_shuf", "shuffled↔shuffled"),
                                 ("base_real", "base↔real"),
                                 ("base_shuf", "base↔shuffled")):
                g = sub[sub.pair_type == ptype].groupby("layer").score.mean()
                if not g.empty:
                    fig.add_trace(
                        go.Scatter(x=g.index, y=g.values, mode="lines+markers",
                                   name=label, marker=dict(size=4),
                                   legendgroup=label, showlegend=(col == 1)),
                        row=1, col=col
                    )

        fig.update_xaxes(title_text="layer", row=1, col=1)
        for col in range(2, len(probes) + 1):
            fig.update_xaxes(title_text="layer", row=1, col=col)
        fig.update_yaxes(title_text="CKA", row=1, col=1)

        fig.update_layout(height=450, width=6 * len(probes) * 100, hovermode="x unified")
        fig_path = "rq1_heldout_convergence.html"
        fig.write_html(fig_path)
        print(f"saved plot -> {fig_path}")
