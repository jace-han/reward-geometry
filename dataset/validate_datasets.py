"""
Validate prepared reward-model datasets before training.

Checks each dataset/{name}_{train,val}.jsonl for:
  - file exists and is valid JSONL
  - has 'chosen' and 'rejected' keys
  - each is a list of {role, content} messages
  - roles are valid and alternate user/assistant correctly
  - no empty/whitespace-only content
  - chosen != rejected (otherwise the preference pair is meaningless)
  - tokenized length distribution vs. RewardConfig max_length (truncation rate)

Usage:
    uv run python validate_datasets.py
    uv run python validate_datasets.py --max-length 512 --base-model Qwen/Qwen2.5-3B
"""

import argparse
import json
from pathlib import Path
from collections import Counter

DATA_DIR = Path("dataset")
DATASETS = ["hh_rlhf", "ultrafeedback", "summarize"]
SPLITS = ["train", "val"]
VALID_ROLES = {"user", "assistant", "system"}


def validate_messages(msgs, field):
    """Return (fatal_errs, warnings) for one chosen/rejected message list.

    fatal_errs: block training (structure, empty content)
    warnings: quality issues (role alternation)
    """
    fatal_errs = []
    warnings = []

    if not isinstance(msgs, list) or len(msgs) == 0:
        return [f"{field}: not a non-empty list"], []

    prev_role = None
    for i, m in enumerate(msgs):
        if not isinstance(m, dict) or "role" not in m or "content" not in m:
            fatal_errs.append(f"{field}[{i}]: missing role/content")
            continue
        role, content = m["role"], m["content"]
        if role not in VALID_ROLES:
            fatal_errs.append(f"{field}[{i}]: bad role {role!r}")
        if not isinstance(content, str) or not content.strip():
            fatal_errs.append(f"{field}[{i}]: empty content")
        if role == prev_role:
            warnings.append(f"{field}[{i}]: role {role!r} repeats (no alternation)")
        prev_role = role

    if msgs[-1].get("role") != "assistant":
        warnings.append(f"{field}: last message is not 'assistant'")

    return fatal_errs, warnings


def validate_file(path, tokenizer=None, max_length=512):
    print(f"\n{'=' * 60}\n{path}\n{'=' * 60}")
    if not path.exists():
        print(f"  ✗ MISSING FILE")
        return False

    n = 0
    n_fatal = 0  # structural errors that block training
    n_warning = 0  # quality issues (not fatal)
    identical = 0
    role_patterns = Counter()
    chosen_lens, rejected_lens, truncated = [], [], 0
    sample_fatal = []
    sample_warnings = []

    with open(path) as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            n += 1
            try:
                ex = json.loads(line)
            except json.JSONDecodeError as e:
                n_fatal += 1
                if len(sample_fatal) < 5:
                    sample_fatal.append(f"line {line_no}: invalid JSON ({e})")
                continue

            fatal_errs = []
            warn_msgs = []

            for field in ("chosen", "rejected"):
                if field not in ex:
                    fatal_errs.append(f"missing key {field!r}")
                else:
                    fatal, warn = validate_messages(ex[field], field)
                    fatal_errs += fatal
                    warn_msgs += warn

            # chosen vs rejected must differ (structural issue if identical)
            if "chosen" in ex and "rejected" in ex and ex["chosen"] == ex["rejected"]:
                identical += 1
                fatal_errs.append("chosen == rejected")

            if "chosen" in ex and isinstance(ex["chosen"], list):
                role_patterns["/".join(m.get("role", "?") for m in ex["chosen"])] += 1

            # tokenized length (only if no fatal errors)
            if tokenizer is not None and not fatal_errs:
                for field, bucket in (
                    ("chosen", chosen_lens),
                    ("rejected", rejected_lens),
                ):
                    try:
                        toks = tokenizer.apply_chat_template(ex[field], tokenize=True)
                        bucket.append(len(toks))
                        if len(toks) > max_length:
                            truncated += 1
                    except Exception as e:
                        fatal_errs.append(f"chat_template failed on {field}: {e}")

            if fatal_errs:
                n_fatal += 1
                if len(sample_fatal) < 5:
                    sample_fatal.append(f"line {line_no}: " + "; ".join(fatal_errs))
            elif warn_msgs:
                n_warning += 1
                if len(sample_warnings) < 5:
                    sample_warnings.append(f"line {line_no}: " + "; ".join(warn_msgs))

    # ---- report ----
    print(
        f"  rows: {n}   fatal errors: {n_fatal}   warnings: {n_warning}   identical: {identical}"
    )
    print(
        f"  top role patterns (chosen): "
        + ", ".join(f"{p}×{c}" for p, c in role_patterns.most_common(3))
    )
    if chosen_lens:
        alll = sorted(chosen_lens + rejected_lens)
        p50 = alll[len(alll) // 2]
        p95 = alll[int(len(alll) * 0.95)]
        pct = 100 * truncated / (len(chosen_lens) + len(rejected_lens))
        print(f"  token length  median={p50}  p95={p95}  max={alll[-1]}")
        print(f"  >{max_length} tokens (will truncate): {truncated} ({pct:.1f}%)")
        if pct > 20:
            print(f"  ⚠ high truncation — consider raising max_length")
    if sample_fatal:
        print("  fatal issues (block training):")
        for e in sample_fatal:
            print(f"    - {e}")
    if sample_warnings:
        print("  warnings (non-blocking):")
        for e in sample_warnings:
            print(f"    - {e}")

    ok = n > 0 and n_fatal == 0
    print(f"  → {'✓ PASS' if ok else '✗ FAIL'}")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument(
        "--base-model",
        default="Qwen/Qwen2.5-3B",
        help="set to '' to skip tokenization checks",
    )
    args = ap.parse_args()

    tokenizer = None
    if args.base_model:
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(args.base_model)
            print(f"Loaded tokenizer: {args.base_model}")
        except Exception as e:
            print(f"⚠ could not load tokenizer ({e}); skipping length checks")

    all_ok = True
    for name in DATASETS:
        for split in SPLITS:
            ok = validate_file(
                DATA_DIR / f"{name}_{split}.jsonl", tokenizer, args.max_length
            )
            all_ok = all_ok and ok

    print(f"\n{'=' * 60}")
    print(
        "ALL DATASETS VALID ✓"
        if all_ok
        else "VALIDATION FAILED ✗ — fix before training"
    )
    print("='*60" if False else "=" * 60)
