#!/bin/bash
# eval_svgd.sh — merge LoRA adapters then evaluate each on LIBERO-Goal
set -e

PYTHON=/root/miniconda3/envs/openvla-oft/bin/python

CKPT_DIR="/root/autodl-tmp/runs/svgd-K2-r32-lam0.1-accum1+libero_goal_no_noops/step_0050000"
OUT_DIR="/root/autodl-tmp/eval_checkpoints/svgd_step50000"
BASE_MODEL="/root/autodl-tmp/openvla-7b"
LOG_DIR="/root/autodl-tmp/eval_logs"
mkdir -p "$LOG_DIR"

cd /root/openvla-oft

# Step 1: merge LoRA adapters into base model
echo "=== Merging LoRA adapters ==="
$PYTHON vla-scripts/prep_eval_svgd.py \
    --base_model "$BASE_MODEL" \
    --checkpoint_dir "$CKPT_DIR" \
    --output_dir "$OUT_DIR" \
    2>&1 | tee "$LOG_DIR/prep_eval.log"

# Step 2: evaluate each merged checkpoint on LIBERO-Goal
for K in 0 1; do
    echo ""
    echo "=== Evaluating lora_${K} on LIBERO-Goal ==="
    $PYTHON experiments/robot/libero/run_libero_eval.py \
        --pretrained_checkpoint "${OUT_DIR}/merged_lora_${K}" \
        --task_suite_name libero_goal \
        --use_l1_regression True \
        --use_proprio True \
        --num_images_in_input 2 \
        --center_crop True \
        --num_trials_per_task 20 \
        --local_log_dir "$LOG_DIR" \
        --run_id_note "svgd_lora${K}" \
        --use_wandb False \
        2>&1 | tee "$LOG_DIR/eval_lora${K}.log"
done

echo ""
echo "=== All evaluations complete. Logs at: $LOG_DIR ==="
