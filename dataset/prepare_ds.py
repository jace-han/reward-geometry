import torch
from datasets import load_dataset, Dataset
N_SAMPLES = 5000
SEED = 42




# ============================================================================
# Format Preference Datasets
# ============================================================================


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
    """UltraFeedback already has chat format."""
    return {
        "chosen": example["chosen"],
        "rejected": example["rejected"],
    }

def format_summarize_feedback(example):
    """Convert summarize_from_feedback comparisons to chat format."""
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
    },
    "ultrafeedback": {
        "name": "trl-lib/ultrafeedback_binarized",
        "formatter": format_ultrafeedback,
    },
    "summarize": {
        "name": "openai/summarize_from_feedback",
        "formatter": format_summarize_feedback,
    },}   

    datasets = {}
    for key, cfg in datasets_config.items():
        print(f"Loading {key}...")
        if key != "summarize":
            ds = load_dataset(cfg["name"])
            datasets[key] = ds
        else:
            datasets[key] = load_dataset(
                "parquet",
                data_files={
                    "train": "hf://datasets/openai/summarize_from_feedback@refs/convert/parquet/comparisons/train/*.parquet",
                    "validation": "hf://datasets/openai/summarize_from_feedback@refs/convert/parquet/comparisons/validation/*.parquet",
                },
            )

    train_ds, val_ds = {}, {}
    for key, ds in datasets.items():
        print(f"Formatting {key}...")
        for split in ds.keys():
            ds_split = ds[split].shuffle(seed=SEED).select(range(N_SAMPLES))
            ds_split = ds_split.map(datasets_config[key]["formatter"], remove_columns=ds[split].column_names)
            if split == "train":
                train_ds[key] = ds_split
            else:
                val_ds[key] = ds_split

    print("\nAll datasets ready.")
    print(f"Train datasets:\n {train_ds}")
    print(f"Validation datasets:\n {val_ds}")