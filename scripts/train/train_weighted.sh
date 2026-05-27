#!/bin/bash
# Weighted: uncertainty-weighted fine-tuning
# GPU 1, ~50K steps (~1.5 days)
# 推荐运行方式: bash train_weighted.sh 2>&1 | tee runs/logs/weighted.log
cd /hdd/slwu/test_5_2/openvla-oft
export PYTHONPATH=/hdd/slwu/test_5_2/openvla-oft:$PYTHONPATH

WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /hdd/slwu/test_5_2/openvla-oft/modified_libero_rlds \
  --dataset_name libero_goal_no_noops \
  --run_root_dir runs/comparison \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --lora_rank 32 \
  --batch_size 1 \
  --grad_accumulation_steps 8 \
  --use_gradient_checkpointing False \
  --merge_lora_during_training False \
  --num_steps_before_decay 40000 \
  --max_steps 50000 \
  --save_freq 5000 \
  --save_latest_checkpoint_only False \
  --seed 42 \
  --uncertainty_weight_file /hdd/slwu/test_5_2/openvla-oft/experiments/data/uncertainty_weights.npz \
  --run_id_note "weighted_uncertainty"
