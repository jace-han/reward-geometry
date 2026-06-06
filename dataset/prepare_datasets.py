from datasets import load_dataset
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

N_TRAIN = 15_000
N_VAL = 15_000
SEED = 42
OUTPUT_DIR = Path("dataset")

def format_hh_rlhf(example):
    """Parse Anthropic hh-rlhf string format into chat messages."""
    def parse_conversation(text):
        messages = []
        parts = text.strip().split("\n\nHuman: ")
        for part in parts:
            if not part:
                continue
            if "\n\nAssistant: " in part:
                human, assistant = part.split("\n\nAssistant: ", 1)
                messages.append({"role": "user", "content": human.strip()})
                messages.append({"role": "assistant", "content": assistant.strip()})
            else:
                messages.append({"role": "user", "content": part.strip()})
        return messages
    return {
        "chosen": parse_conversation(example["chosen"]),
        "rejected": parse_conversation(example["rejected"]),
    }

def format_ultrafeedback(example):
    return {
        "chosen": example["chosen"],
        "rejected": example["rejected"],
    }

def format_summarize_feedback(example):
    post = example["info"]["post"]
    prompt = f"Summarize the following post:\n\n{post}"
    choice_idx = example["choice"]
    rejected_idx = 1 - choice_idx
    return {
        "chosen": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": example["summaries"][choice_idx]["text"]},
        ],
        "rejected": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": example["summaries"][rejected_idx]["text"]},
        ],
    }

if __name__ == "__main__":
    datasets_config = {
        "hh_rlhf": {
            "name": "Anthropic/hh-rlhf",
            "formatter": format_hh_rlhf,
            "train_split": "train",
            "val_split": "test",
        },
        "ultrafeedback": {
            "name": "trl-lib/ultrafeedback_binarized",
            "formatter": format_ultrafeedback,
            "train_split": "train",
            "val_split": "test",
        },
        "summarize": {
            "name": "openai/summarize_from_feedback",
            "formatter": format_summarize_feedback,
            "load_kwargs": {
                "data_files": {
                    "train": "hf://datasets/openai/summarize_from_feedback@refs/convert/parquet/comparisons/train/*.parquet",
                    "validation": "hf://datasets/openai/summarize_from_feedback@refs/convert/parquet/comparisons/validation/*.parquet",
                },
            },
            "loader": "parquet",
            "train_split": "train",
            "val_split": "validation",
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for key, cfg in datasets_config.items():
        print(f"Loading {key}...")
        if cfg.get("loader") == "parquet":
            ds = load_dataset("parquet", data_files=cfg["load_kwargs"]["data_files"])
        else:
            ds = load_dataset(cfg["name"])

        # Load training set
        train_split_name = cfg["train_split"]
        train_raw = ds[train_split_name].shuffle(seed=SEED)
        n_train = min(len(train_raw), N_TRAIN)
        train_split = train_raw.select(range(n_train))

        # Load validation set
        val_split_name = cfg["val_split"]
        val_raw = ds[val_split_name].shuffle(seed=SEED)
        n_val = min(len(val_raw), N_VAL)
        val_split = val_raw.select(range(n_val))

        train_split = train_split.map(cfg["formatter"], remove_columns=train_raw.column_names)
        val_split = val_split.map(cfg["formatter"], remove_columns=val_raw.column_names)
        train_split.to_json(OUTPUT_DIR / f"{key}_train.jsonl")
        val_split.to_json(OUTPUT_DIR / f"{key}_val.jsonl")
        print(f"  {key}: {len(train_split)} train, {len(val_split)} val")

    print("\nAll datasets prepared and saved to", OUTPUT_DIR)