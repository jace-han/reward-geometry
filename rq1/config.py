"""Shared constants and the 6-model registry for RQ1."""

from itertools import combinations
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"

SEED = 42
N_EXAMPLES = 4096
KNN_SUBSAMPLE = 1024
K_VALUES = (5, 10, 20)
PRIMARY_K = 10
MAX_LENGTH = 512

# Artifact locations
ARTIFACT_DIR = REPO_ROOT / "rq1" / "artifacts"
FIGURE_DIR = REPO_ROOT / "rq1" / "figures"
REPR_DIR = REPO_ROOT / "representations" / "rq1"
INDICES_PATH = ARTIFACT_DIR / "alpaca_indices.json"
RESULTS_PATH = ARTIFACT_DIR / "results.parquet"
GEOMETRY_PATH = ARTIFACT_DIR / "geometry.parquet"

# logical name -> checkpoint path (or hub id for base)
MODELS = {
    "base": "Qwen/Qwen2.5-3B",
    "hh": str(CHECKPOINT_DIR / "hh_rlhf" / "best"),
    "summarize": str(CHECKPOINT_DIR / "summarize" / "best"),
    "uf": str(CHECKPOINT_DIR / "ultrafeedback" / "best"),
    "hh_shuf": str(CHECKPOINT_DIR / "hh_rlhf_shuffled" / "best"),
    "uf_shuf": str(CHECKPOINT_DIR / "ultrafeedback_shuffled" / "best"),
}

REAL = ("hh", "summarize", "uf")
SHUFFLED = ("hh_shuf", "uf_shuf")

ROLES = {
    "base": {"base"},
    "real": set(REAL),
    "shuffled": set(SHUFFLED),
}


def _build_pairs():
    pairs = []
    # base <-> real (inheritance floor)
    for r in REAL:
        pairs.append({"a": "base", "b": r, "pair_type": "base_real"})
    # real <-> real (convergence signal)
    for a, b in combinations(REAL, 2):
        pairs.append({"a": a, "b": b, "pair_type": "real_real"})
    # shuffled <-> shuffled (null floor)
    for a, b in combinations(SHUFFLED, 2):
        pairs.append({"a": a, "b": b, "pair_type": "shuf_shuf"})
    # real <-> shuffled (objective attribution): matched + cross
    pairs.append({"a": "hh", "b": "hh_shuf", "pair_type": "real_shuf"})
    pairs.append({"a": "uf", "b": "uf_shuf", "pair_type": "real_shuf"})
    pairs.append({"a": "hh", "b": "uf_shuf", "pair_type": "real_shuf"})
    pairs.append({"a": "uf", "b": "hh_shuf", "pair_type": "real_shuf"})
    return pairs


PAIRS = _build_pairs()
