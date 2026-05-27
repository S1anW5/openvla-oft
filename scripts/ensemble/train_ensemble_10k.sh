#!/bin/bash
# 训练 3 个 ensemble 成员各 10000 步，用于验证 ensemble 多样性
# seed=42 已有 10000 步 checkpoint，只跑 422 和 4222
# 用法: bash train_ensemble_10k.sh

cd /hdd/slwu/test_5_2/openvla-oft
export PYTHONPATH=/hdd/slwu/test_5_2/openvla-oft:$PYTHONPATH

for SEED in 422 4222; do
    echo "=========================================="
    echo "Training ensemble member with seed=${SEED} (10k steps)"
    echo "=========================================="
    WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
        --vla_path openvla/openvla-7b \
        --data_root_dir /hdd/slwu/test_5_2/openvla-oft/modified_libero_rlds \
        --dataset_name libero_goal_no_noops \
        --run_root_dir /hdd/slwu/test_5_2/openvla-oft/runs/ensemble \
        --seed $SEED \
        --use_l1_regression True \
        --use_diffusion False \
        --use_film False \
        --num_images_in_input 2 \
        --use_proprio True \
        --lora_rank 32 \
        --batch_size 1 \
        --grad_accumulation_steps 8 \
        --merge_lora_during_training False \
        --num_steps_before_decay 40000 \
        --max_steps 10000 \
        --save_freq 10000 \
        --save_latest_checkpoint_only False \
        --image_aug True \
        --run_id_note seed${SEED}
    echo "Finished seed=${SEED}"
done

echo "All 10k ensemble members trained!"
