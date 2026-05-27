#!/bin/bash
# Sequential evaluation of all three 50K checkpoints on GPU 0
# Each run: libero_goal, 10 tasks x 50 trials = 500 rollouts
# 推荐运行方式: bash run_eval_comparison.sh 2>&1 | tee runs/logs/eval_comparison.log
cd /hdd/slwu/test_5_2/openvla-oft
export PYTHONPATH=/hdd/slwu/test_5_2/openvla-oft:/hdd/slwu/LIBERO:$PYTHONPATH

BASE=runs/comparison/openvla-7b+libero_goal_no_noops+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug

echo "=========================================="
echo "[1/3] Evaluating: baseline (50K)"
echo "=========================================="
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint ${BASE}--baseline--50000_chkpt \
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
echo "[2/3] Evaluating: weighted_uncertainty (50K)"
echo "=========================================="
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint ${BASE}--weighted_uncertainty--50000_chkpt \
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
echo "[3/3] Evaluating: official_goal (50K)"
echo "=========================================="
CUDA_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint runs/official_repro/openvla-7b+libero_goal_no_noops+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--official_goal--50000_chkpt \
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
echo "All evaluations done. Results in experiments/logs/"
echo "=========================================="
