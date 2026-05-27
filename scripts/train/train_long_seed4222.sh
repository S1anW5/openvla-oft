#!/bin/bash
# LIBERO-10 (Long) baseline, seed=4222, GPU 2
# 推荐运行方式: bash train_long_seed4222.sh 2>&1 | tee runs/logs/long_seed4222.log
cd /hdd/slwu/test_5_2/openvla-oft
export PYTHONPATH=/hdd/slwu/test_5_2/openvla-oft:$PYTHONPATH

WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=2 torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /hdd/slwu/test_5_2/openvla-oft/modified_libero_rlds \
  --dataset_name libero_10_no_noops \
  --run_root_dir /hdd/slwu/test_5_2/openvla-oft/runs/long \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --lora_rank 32 \
  --batch_size 1 \
  --grad_accumulation_steps 8 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 80000 \
  --max_steps 100000 \
  --save_freq 10000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --use_gradient_checkpointing False \
  --merge_lora_during_training False \
  --seed 4222 \
  --run_id_note "long_seed4222"
