"""
Train Qwen2.5-3B reward models (full fine-tune, Bradley-Terry head).
Designed for Vast.AI A6000 (48GB VRAM).

Usage:
    python train_reward_model.py --dataset hh_rlhf
    python train_reward_model.py --dataset ultrafeedback
    python train_reward_model.py --dataset summarize
    python train_reward_model.py --all
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
)
from trl import RewardTrainer, RewardConfig
import wandb

BASE_MODEL = "Qwen/Qwen2.5-3B"
DATA_DIR = Path("dataset")
CHECKPOINT_DIR = Path("checkpoints")
SEED = 42
WANDB_PROJECT = "reward-geometry"

DATASETS = ["hh_rlhf", "ultrafeedback", "summarize"]


def compute_metrics(eval_pred):
    """Compute preference accuracy: how often chosen_reward > rejected_reward."""
    predictions = eval_pred.predictions
    # RewardTrainer outputs [chosen_reward, rejected_reward] per sample
    if predictions.ndim == 2 and predictions.shape[1] == 2:
        accuracy = (predictions[:, 0] > predictions[:, 1]).mean()
    else:
        accuracy = (predictions > 0).mean()
    return {"accuracy": float(accuracy)}


def train_model(dataset_name: str, resume_from: str = None, smoke_test: bool = False):
    """Train one reward model on the specified dataset."""

    print(f"\n{'='*60}")
    print(f"Training reward model on: {dataset_name}")
    if smoke_test:
        print("  *** SMOKE TEST MODE — 100 samples, 50 steps ***")
    print(f"{'='*60}")

    # Init W&B
    run = wandb.init(
        project=WANDB_PROJECT,
        name=f"qwen2.5-3b-{dataset_name}" + ("-smoke" if smoke_test else ""),
        config={
            "base_model": BASE_MODEL,
            "dataset": dataset_name,
            "n_train": 100 if smoke_test else 10000,
            "n_val": 20 if smoke_test else 1000,
            "full_finetune": True,
            "smoke_test": smoke_test,
        },
        reinit=True,
    )

    # Load data
    train_ds = load_dataset("json", data_files=str(DATA_DIR / f"{dataset_name}_train.jsonl"), split="train")
    val_ds = load_dataset("json", data_files=str(DATA_DIR / f"{dataset_name}_val.jsonl"), split="train")

    if smoke_test:
        train_ds = train_ds.select(range(min(100, len(train_ds))))
        val_ds = val_ds.select(range(min(20, len(val_ds))))

    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=1,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    model.gradient_checkpointing_enable()

    output_dir = CHECKPOINT_DIR / dataset_name

    training_args = RewardConfig(
        output_dir=str(output_dir),
        num_train_epochs=1 if smoke_test else 2,
        max_steps=50 if smoke_test else -1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        max_length=512,
        eval_strategy="steps",
        eval_steps=10 if smoke_test else 250,
        save_strategy="steps",
        save_steps=10 if smoke_test else 250,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        # Logging
        logging_steps=5 if smoke_test else 50,
        report_to="wandb",
        remove_unused_columns=False,
        seed=SEED,
    )

    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Resume from checkpoint if specified
    if resume_from:
        print(f"  Resuming from: {resume_from}")
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        trainer.train()

    # Save final best model
    best_dir = output_dir / "best"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))

    # Log final metrics
    final_metrics = trainer.evaluate()
    eval_acc = final_metrics.get("eval_accuracy", 0)
    print(f"\n  Final eval loss: {final_metrics['eval_loss']:.4f}")
    print(f"  Final eval accuracy: {eval_acc}")

    # W&B alert on completion
    wandb.alert(
        title=f"Training complete: {dataset_name}",
        text=(
            f"Model: qwen2.5-3b-{dataset_name}\n"
            f"Eval loss: {final_metrics['eval_loss']:.4f}\n"
            f"Eval accuracy: {eval_acc:.3f}"
        ),
        level=wandb.AlertLevel.INFO,
    )

    # Smoke test validation: check model actually learned something
    if smoke_test:
        if eval_acc < 0.50:
            wandb.alert(
                title=f"SMOKE TEST FAILED: {dataset_name}",
                text=f"Accuracy {eval_acc:.3f} <= 0.50 — model not learning!",
                level=wandb.AlertLevel.ERROR,
            )
            print(f"  SMOKE TEST FAILED: accuracy {eval_acc:.3f} <= 0.50")
        else:
            print(f"  SMOKE TEST PASSED: accuracy {eval_acc:.3f} > 0.50")

    wandb.finish()

    # Free memory
    del model, trainer
    torch.cuda.empty_cache()

    return final_metrics    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=DATASETS, help="Which dataset to train on")
    parser.add_argument("--all", action="store_true", help="Train all 3 models sequentially")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick validation (100 samples, 20 steps)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    if args.smoke_test:
        print("\n*** SMOKE TEST MODE ***")
        print("Running 1 dataset (hh_rlhf) with 100 samples, 50 steps...")
        print("This should take ~5 minutes on A6000.\n")
        result = train_model("hh_rlhf", smoke_test=True)
        acc = result.get("eval_accuracy", 0)
        if acc > 0.50:
            print(f"\n SMOKE TEST PASSED (accuracy: {acc:.3f})")
            print("Safe to launch full training: python train_reward_model.py --all")
        else:
            print(f"\n SMOKE TEST FAILED (accuracy: {acc:.3f})")
            print("Debug before launching full training!")
        return

    if args.all:
        results = {}
        for ds_name in DATASETS:
            results[ds_name] = train_model(ds_name)

        # Send final completion alert
        summary = "\n".join(f"  {k}: loss={v['eval_loss']:.4f}" for k, v in results.items())
        wandb.init(project=WANDB_PROJECT, name="pipeline-complete", reinit=True)
        wandb.alert(
            title="ALL 3 REWARD MODELS TRAINED",
            text=f"Training pipeline complete!\n\n{summary}\n\nRun extract_representations.py next.",
            level=wandb.AlertLevel.INFO,
        )
        wandb.finish()

        print("\n\n" + "="*60)
        print("ALL TRAINING COMPLETE")
        print("="*60)
        print(summary)
    elif args.dataset:
        train_model(args.dataset, resume_from=args.resume)
    else:
        parser.print_help()