#!/bin/bash
set -e

PYTHON=/hdd/slwu/miniconda3/envs/openvla/bin/python
BASE_MODEL="/mnt/hdd_1/share/huggingface/hub/models--openvla--openvla-7b/snapshots/47a0ec7fc4ec123775a391911046cf33cf9ed83f"
CKPT_DIR="/hdd/slwu/test_5_2/openvla-oft/runs/svgd/svgd-K2-r32-lam0.1-accum1+libero_goal_no_noops/step_0050000"
OUT_DIR="/hdd/slwu/test_5_2/openvla-oft/runs/svgd/eval_checkpoints"
LOG_DIR="/hdd/slwu/test_5_2/openvla-oft/runs/svgd/logs"

export PYTHONPATH=/hdd/slwu/test_5_2/openvla-oft:/hdd/slwu/LIBERO:$PYTHONPATH
export MUJOCO_GL=egl

mkdir -p "$OUT_DIR" "$LOG_DIR"

# ── Step 1: merge LoRA adapters ──────────────────────────────────────────────
echo "=== Merging LoRA adapters ==="
CUDA_VISIBLE_DEVICES=1 $PYTHON vla-scripts/prep_eval_svgd.py \
    --base_model "$BASE_MODEL" \
    --checkpoint_dir "$CKPT_DIR" \
    --output_dir "$OUT_DIR" \
    --num_particles 2

# copy tokenizer/processor files from base model (prep script only copies from ckpt_dir)
for K in 0 1; do
    MERGED="$OUT_DIR/merged_lora_${K}"
    for f in added_tokens.json preprocessor_config.json processor_config.json \
              processing_prismatic.py special_tokens_map.json tokenizer.json \
              tokenizer.model tokenizer_config.json; do
        [ -f "$BASE_MODEL/$f" ] && cp "$BASE_MODEL/$f" "$MERGED/$f"
    done
    echo "Tokenizer files copied to merged_lora_${K}"
done

# ── Step 2: eval both in parallel (lora_0 on GPU 0, lora_1 on GPU 1) ─────────
echo "=== Starting eval: lora_0 on GPU 2, lora_1 on GPU 3 ==="

CUDA_VISIBLE_DEVICES=2 $PYTHON experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint "$OUT_DIR/merged_lora_0" \
    --task_suite_name libero_goal \
    --use_l1_regression True --use_diffusion False --use_film False \
    --num_images_in_input 2 --use_proprio True \
    --center_crop True --num_trials_per_task 20 \
    --unnorm_key libero_goal_no_noops \
    --run_id_note svgd_local_lora0 --use_wandb False --local_log_dir "$LOG_DIR" \
    2>&1 | tee "$LOG_DIR/eval_lora0.log" &

CUDA_VISIBLE_DEVICES=3 $PYTHON experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint "$OUT_DIR/merged_lora_1" \
    --task_suite_name libero_goal \
    --use_l1_regression True --use_diffusion False --use_film False \
    --num_images_in_input 2 --use_proprio True \
    --center_crop True --num_trials_per_task 20 \
    --unnorm_key libero_goal_no_noops \
    --run_id_note svgd_local_lora1 --use_wandb False --local_log_dir "$LOG_DIR" \
    2>&1 | tee "$LOG_DIR/eval_lora1.log" &

wait
echo "=== All done ==="
grep -E "Total successes|Overall" "$LOG_DIR/eval_lora0.log" "$LOG_DIR/eval_lora1.log" 2>/dev/null
