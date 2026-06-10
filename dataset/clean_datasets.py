"""
Clean dataset files by removing invalid preference pairs.

Removes:
  - Rows where chosen == rejected (meaningless preferences)
  - Rows with empty/whitespace-only content
  - Rows with invalid message structure
  - Non-JSON lines

Outputs cleaned versions to dataset/{name}_{split}.jsonl (overwrites)
"""

import json
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path("dataset")
DATASETS = ["hh_rlhf", "ultrafeedback", "summarize"]
SPLITS = ["train", "val"]
VALID_ROLES = {"user", "assistant", "system"}


def validate_messages(msgs):
    """Check if message list is valid. Return True if valid, False otherwise."""
    if not isinstance(msgs, list) or len(msgs) == 0:
        return False
    prev_role = None
    for m in msgs:
        if not isinstance(m, dict) or "role" not in m or "content" not in m:
            return False
        role, content = m["role"], m["content"]
        if role not in VALID_ROLES:
            return False
        if not isinstance(content, str) or not content.strip():
            return False
        if role == prev_role:
            return False
        prev_role = role
    if msgs[-1].get("role") != "assistant":
        return False
    return True


def clean_file(path):
    """Clean a dataset file. Return (cleaned_count, removed_count)."""
    if not path.exists():
        print(f"  ✗ MISSING: {path}")
        return 0, 0

    cleaned = []
    removed_reasons = defaultdict(int)

    with open(path) as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                removed_reasons["invalid_json"] += 1
                continue

            # Check keys exist
            if "chosen" not in ex or "rejected" not in ex:
                removed_reasons["missing_keys"] += 1
                continue

            # Check message structure
            if not validate_messages(ex["chosen"]) or not validate_messages(
                ex["rejected"]
            ):
                removed_reasons["invalid_messages"] += 1
                continue

            # Check chosen != rejected
            if ex["chosen"] == ex["rejected"]:
                removed_reasons["identical_pairs"] += 1
                continue

            cleaned.append(ex)

    # Write cleaned version back
    with open(path, "w") as f:
        for ex in cleaned:
            f.write(json.dumps(ex) + "\n")

    total_removed = sum(removed_reasons.values())
    print(f"  {path.name}: {len(cleaned)} kept, {total_removed} removed")
    for reason, count in sorted(removed_reasons.items(), key=lambda x: -x[1]):
        print(f"    - {reason}: {count}")

    return len(cleaned), total_removed


if __name__ == "__main__":
    print("Cleaning datasets...\n")

    total_cleaned = 0
    total_removed = 0

    for name in DATASETS:
        for split in SPLITS:
            path = DATA_DIR / f"{name}_{split}.jsonl"
            cleaned, removed = clean_file(path)
            total_cleaned += cleaned
            total_removed += removed

    print(f"\n{'=' * 60}")
    print(f"Total: {total_cleaned} rows kept, {total_removed} rows removed")
    print(f"{'=' * 60}")
    print("\nRe-run validation to confirm:")
    print("  python3 validate_datasets.py")
