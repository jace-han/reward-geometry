#!/bin/bash
set -e

echo "=== Reward Model Training ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Date: $(date)"
echo ""

# Train all 3 reward models (W&B alerts on each completion)
uv run python train_reward_models.py --all

# Train 2 shuffled models
uv run python train_reward_models.py --all-shuffled

# Package checkpoints for download
echo ""
echo "=== Packaging checkpoints ==="
if [ -d "checkpoints" ] && [ "$(find checkpoints -type d -name best)" ]; then
    tar -czf checkpoints_best.tar.gz checkpoints/*/best/
else
    echo "ERROR: No checkpoints/*/best/ directories found!"
    exit 1
fi

echo ""
echo "=== DONE ==="
echo "Download checkpoints_best.tar.gz (~18GB) to your local machine."
du -sh checkpoints/*/best/