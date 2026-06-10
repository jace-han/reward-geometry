"""Registry + paths for the held-out preference probes (RQ1 reward-axis)."""

from pathlib import Path

from rq1 import config

N_PAIRS = 1024
SEED = config.SEED
MAX_LENGTH = config.MAX_LENGTH

DATA_DIR = config.REPO_ROOT / "dataset"
REPR_DIR = config.REPR_DIR

PROBES = {
    "shp": {"hub_id": "stanfordnlp/SHP", "split": "validation"},
    "pku_safe": {"hub_id": "PKU-Alignment/PKU-SafeRLHF", "split": "test"},
    "arena": {"hub_id": "lmsys/chatbot_arena_conversations", "split": "train"},
}


def probe_jsonl(probe: str) -> Path:
    return DATA_DIR / f"{probe}_heldout.jsonl"


def diff_path(probe: str, model: str) -> Path:
    return REPR_DIR / f"{probe}__{model}__diff.npy"


def shared_prompts_path(probe: str) -> Path:
    return config.ARTIFACT_DIR / f"{probe}_shared_prompts.json"
