#!/bin/bash
# 推荐运行方式: bash run_infer.sh 2>&1 | tee runs/logs/infer.log
cd /hdd/slwu/test_5_2/openvla-oft
export PYTHONPATH=/hdd/slwu/test_5_2/openvla-oft:$PYTHONPATH

CKPT_BASE=runs/ensemble
DATA_ROOT=/hdd/slwu/test_5_2/openvla-oft/modified_libero_rlds
PRED_DIR=experiments/data/preds
mkdir -p $PRED_DIR

for SEED in 42 422 4222; do
    CKPT_DIR="${CKPT_BASE}/openvla-7b+libero_goal_no_noops+b8+lr-0.0005+lora-r32+dropout-0.0--image_aug--seed${SEED}--10000_chkpt"
    echo "=========================================="
    echo "Inferring seed=${SEED}"
    echo "=========================================="
    CUDA_VISIBLE_DEVICES=0 python -u vla-scripts/compute_ensemble_uncertainty.py \
        --mode infer \
        --vla_path openvla/openvla-7b \
        --data_root_dir $DATA_ROOT \
        --dataset_name libero_goal_no_noops \
        --checkpoint_dir $CKPT_DIR \
        --output_path ${PRED_DIR}/preds_seed${SEED}.npy \
        --unnorm_key libero_goal_no_noops \
        --cuda_device 0
    echo "Finished seed=${SEED}"
done
echo "All inference done!"
