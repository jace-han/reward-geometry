"""Format three held-out preference datasets into {chosen, rejected} chat schema.

None were used to train hh / uf / summarize, so they are equally out-of-distribution
probes for the RQ1 reward-axis convergence test.

Datasets and confirmed field names (2026-06-09):
  - SHP (stanfordnlp/SHP): history, human_ref_A, human_ref_B, labels (1->A preferred)
  - PKU-SafeRLHF (PKU-Alignment/PKU-SafeRLHF): prompt, response_0, response_1,
    better_response_id (0 or 1)
  - Chatbot Arena (lmsys/chatbot_arena_conversations): conversation_a, conversation_b,
    winner (model_a / model_b / tie / tie (bothbad))

Usage:
    uv run python -m dataset.prepare_heldout
"""

import json

from datasets import load_dataset
from dotenv import load_dotenv

from rq1 import heldout_config as hc

load_dotenv()


def _user_assistant(prompt: str, response: str):
    return [
        {"role": "user", "content": prompt.strip()},
        {"role": "assistant", "content": response.strip()},
    ]


def format_shp(row):
    """SHP: labels==1 -> human_ref_A preferred, else human_ref_B."""
    prompt = row["history"]
    a, b = row["human_ref_A"], row["human_ref_B"]
    if int(row["labels"]) == 1:
        chosen_resp, rejected_resp = a, b
    else:
        chosen_resp, rejected_resp = b, a
    return {
        "chosen": _user_assistant(prompt, chosen_resp),
        "rejected": _user_assistant(prompt, rejected_resp),
    }


def format_pku(row):
    """PKU-SafeRLHF: better_response_id in {0,1} selects the preferred response."""
    prompt = row["prompt"]
    r0, r1 = row["response_0"], row["response_1"]
    if int(row["better_response_id"]) == 0:
        chosen_resp, rejected_resp = r0, r1
    else:
        chosen_resp, rejected_resp = r1, r0
    return {
        "chosen": _user_assistant(prompt, chosen_resp),
        "rejected": _user_assistant(prompt, rejected_resp),
    }


def _valid_conversation(msgs):
    """Check all messages have non-empty content and valid roles."""
    if not msgs:
        return False
    for m in msgs:
        if not isinstance(m, dict):
            return False
        if not m.get("content", "").strip():
            return False
    return True


def format_arena(row):
    """Chatbot Arena: winner in {model_a, model_b}; ties -> None (dropped)."""
    winner = row["winner"]
    if winner == "model_a":
        chosen, rejected = row["conversation_a"], row["conversation_b"]
    elif winner == "model_b":
        chosen, rejected = row["conversation_b"], row["conversation_a"]
    else:
        return None
    if not _valid_conversation(chosen) or not _valid_conversation(rejected):
        return None
    return {"chosen": chosen, "rejected": rejected}


FORMATTERS = {"shp": format_shp, "pku_safe": format_pku, "arena": format_arena}


def build_probe(probe: str):
    cfg = hc.PROBES[probe]
    ds = load_dataset(cfg["hub_id"], split=cfg["split"]).shuffle(seed=hc.SEED)
    fmt = FORMATTERS[probe]
    out_path = hc.probe_jsonl(probe)
    kept = 0
    with out_path.open("w") as f:
        for row in ds:
            rec = fmt(row)
            if rec is None:
                continue
            if rec["chosen"] == rec["rejected"]:
                continue
            f.write(json.dumps(rec) + "\n")
            kept += 1
            if kept >= hc.N_PAIRS:
                break
    print(f"{probe}: wrote {kept} pairs -> {out_path}")


if __name__ == "__main__":
    for probe in hc.PROBES:
        build_probe(probe)
