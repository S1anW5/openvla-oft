#!/bin/bash
# LIBERO-Object, seed=42, 从40K步续训到100K步, GPU 3
# LR 在总第80K步（再跑40K）衰减，总第100K步（再跑60K）结束
# 推荐运行方式: bash scripts/train/resume_object_seed42_from40k.sh 2>&1 | tee runs/logs/object_seed42_resume40k.log
cd /hdd/slwu/test_5_2/openvla-oft
export PYTHONPATH=/hdd/slwu/test_5_2/openvla-oft:$PYTHONPATH

WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=3 torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /hdd/slwu/test_5_2/openvla-oft/modified_libero_rlds \
  --dataset_name libero_object_no_noops \
  --run_root_dir /hdd/slwu/test_5_2/openvla-oft/runs/object \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --lora_rank 32 \
  --batch_size 1 \
  --grad_accumulation_steps 8 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 40000 \
  --max_steps 60000 \
  --save_freq 10000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --use_gradient_checkpointing False \
  --merge_lora_during_training False \
  --seed 42 \
  --run_id_note "object_seed42" \
  --resume True \
  --resume_step 40000 \
  --resume_dir /hdd/slwu/test_5_2/openvla-oft/runs/object/openvla-7b+libero_object_no_noops+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--object_seed42--40000_chkpt
