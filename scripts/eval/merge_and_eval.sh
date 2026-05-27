#!/bin/bash
# Step 1: Merge LoRA into base model for all three 50K checkpoints
# Step 2: Run evaluation sequentially on GPU 0
# 推荐运行方式: bash merge_and_eval.sh 2>&1 | tee runs/logs/merge_and_eval.log
cd /hdd/slwu/test_5_2/openvla-oft
export PYTHONPATH=/hdd/slwu/test_5_2/openvla-oft:/hdd/slwu/LIBERO:$PYTHONPATH

BASE=runs/comparison/openvla-7b+libero_goal_no_noops+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug
OFFICIAL=runs/official_repro/openvla-7b+libero_goal_no_noops+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug

CKPT_BASELINE=${BASE}--baseline--50000_chkpt
CKPT_WEIGHTED=${BASE}--weighted_uncertainty--50000_chkpt
CKPT_OFFICIAL=${OFFICIAL}--official_goal--50000_chkpt

echo "=========================================="
echo "[1/3] Merging LoRA: baseline"
echo "=========================================="
CUDA_VISIBLE_DEVICES=0 python vla-scripts/merge_lora_weights_and_save.py \
  --base_checkpoint openvla/openvla-7b \
  --lora_finetuned_checkpoint_dir ${CKPT_BASELINE}

echo "=========================================="
echo "[2/3] Merging LoRA: weighted_uncertainty"
echo "=========================================="
CUDA_VISIBLE_DEVICES=0 python vla-scripts/merge_lora_weights_and_save.py \
  --base_checkpoint openvla/openvla-7b \
  --lora_finetuned_checkpoint_dir ${CKPT_WEIGHTED}

echo "=========================================="
echo "[3/3] Merging LoRA: official_goal"
echo "=========================================="
CUDA_VISIBLE_DEVICES=0 python vla-scripts/merge_lora_weights_and_save.py \
  --base_checkpoint openvla/openvla-7b \
  --lora_finetuned_checkpoint_dir ${CKPT_OFFICIAL}

echo "=========================================="
echo "All merges done. Starting evaluation..."
echo "=========================================="

echo "=========================================="
echo "[Eval 1/3] baseline (50K)"
echo "=========================================="
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint ${CKPT_BASELINE} \
  --task_suite_name libero_goal \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --lora_rank 32 \
  --center_crop True \
  --num_trials_per_task 50 \
  --unnorm_key libero_goal_no_noops \
  --run_id_note "eval_baseline_50k"

echo "=========================================="
echo "[Eval 2/3] weighted_uncertainty (50K)"
echo "=========================================="
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint ${CKPT_WEIGHTED} \
  --task_suite_name libero_goal \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --lora_rank 32 \
  --center_crop True \
  --num_trials_per_task 50 \
  --unnorm_key libero_goal_no_noops_weighted \
  --run_id_note "eval_weighted_50k"

echo "=========================================="
echo "[Eval 3/3] official_goal (50K)"
echo "=========================================="
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint ${CKPT_OFFICIAL} \
  --task_suite_name libero_goal \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --lora_rank 32 \
  --center_crop True \
  --num_trials_per_task 50 \
  --unnorm_key libero_goal_no_noops \
  --run_id_note "eval_official_50k"

echo "=========================================="
echo "All done! Results in experiments/logs/"
echo "=========================================="
