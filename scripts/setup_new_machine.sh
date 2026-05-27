#!/bin/bash
# One-shot setup script for a new AutoDL machine.
# Usage:
#   bash scripts/setup_new_machine.sh [DATASET]
#
# DATASET: which LIBERO suite to download (default: all)
#   all | libero_goal | libero_spatial | libero_object | libero_10
#
# After this script completes, launch training with:
#   CUDA_VISIBLE_DEVICES=0 python vla-scripts/finetune_svgd_ensemble.py \
#     --dataset_name libero_goal_no_noops ...

set -e
DATASET=${1:-all}
AUTODL=/root/autodl-tmp
CONDA_ENV=openvla-oft

echo "============================================================"
echo "  OpenVLA-OFT setup  |  dataset=$DATASET"
echo "============================================================"

# ── 1. conda environment ─────────────────────────────────────────
echo ""
echo "[1/6] Conda environment..."
source /root/miniconda3/etc/profile.d/conda.sh

if conda env list | grep -q "^$CONDA_ENV "; then
    echo "  env '$CONDA_ENV' already exists, skipping creation"
else
    conda create -n $CONDA_ENV python=3.10 -y
fi
conda activate $CONDA_ENV

# ── 2. clone repo ────────────────────────────────────────────────
echo ""
echo "[2/6] Cloning repo..."
if [ ! -d "/root/openvla-oft" ]; then
    git clone https://github.com/S1anW5/openvla-oft.git /root/openvla-oft
else
    echo "  /root/openvla-oft already exists, pulling latest..."
    git -C /root/openvla-oft pull
fi
cd /root/openvla-oft

# ── 3. install dependencies ──────────────────────────────────────
echo ""
echo "[3/6] Installing dependencies..."

# Install PyTorch with CUDA 12.8 support (required for Blackwell RTX PRO 6000)
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "  Installing PyTorch (CUDA 12.8)..."
    pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu128 -q
else
    echo "  PyTorch already installed: $(python -c 'import torch; print(torch.__version__)')"
fi

pip install -e . -q

# Flash Attention: not required (training uses attn_implementation=sdpa).
# Skip on Blackwell (SM_120) since flash-attn 2.5.5 does not support it.
GPU_ARCH=$(python -c "import torch; print(torch.cuda.get_device_capability()[0])" 2>/dev/null || echo "0")
if ! python -c "import flash_attn" 2>/dev/null; then
    if [ "$GPU_ARCH" -ge 12 ] 2>/dev/null; then
        echo "  Skipping flash_attn on Blackwell (SM_${GPU_ARCH}x) — sdpa is used instead"
    else
        echo "  Installing flash_attn..."
        pip install packaging ninja -q
        pip install "flash-attn==2.5.5" --no-build-isolation -q || \
            echo "  flash_attn install failed — training will use sdpa (fine)"
    fi
else
    echo "  flash_attn already installed"
fi

# LIBERO
if [ ! -d "/root/LIBERO" ]; then
    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /root/LIBERO
    pip install -e /root/LIBERO -q
    pip install -r /root/openvla-oft/experiments/robot/libero/libero_requirements.txt -q
else
    echo "  /root/LIBERO already exists"
fi

# ── 4. base model ────────────────────────────────────────────────
echo ""
echo "[4/6] Downloading base model (openvla-7b)..."
if [ ! -f "$AUTODL/openvla-7b/config.json" ]; then
    mkdir -p $AUTODL/openvla-7b
    huggingface-cli download openvla/openvla-7b \
        --local-dir $AUTODL/openvla-7b \
        --local-dir-use-symlinks False
else
    echo "  $AUTODL/openvla-7b already exists"
fi

# ── 5. LIBERO datasets ────────────────────────────────────────────
echo ""
echo "[5/6] Downloading LIBERO RLDS datasets (dataset=$DATASET)..."
mkdir -p $AUTODL/modified_libero_rlds

download_suite() {
    local name=$1
    local dest=$AUTODL/modified_libero_rlds/$name
    if [ -d "$dest" ]; then
        echo "  $name already exists, skipping"
        return
    fi
    echo "  Downloading $name ..."
    huggingface-cli download openvla/modified_libero_rlds \
        --repo-type dataset \
        --local-dir $AUTODL/modified_libero_rlds \
        --local-dir-use-symlinks False \
        --include "${name}/*"
}

case $DATASET in
    all)
        for suite in libero_goal_no_noops libero_spatial_no_noops libero_object_no_noops libero_10_no_noops; do
            download_suite $suite
        done
        ;;
    libero_goal)    download_suite libero_goal_no_noops ;;
    libero_spatial) download_suite libero_spatial_no_noops ;;
    libero_object)  download_suite libero_object_no_noops ;;
    libero_10)      download_suite libero_10_no_noops ;;
    *)
        echo "Unknown dataset: $DATASET"
        echo "Valid options: all | libero_goal | libero_spatial | libero_object | libero_10"
        exit 1
        ;;
esac

# ── 6. smoke test ─────────────────────────────────────────────────
echo ""
echo "[6/6] Running 5-step smoke test..."
cd /root/openvla-oft
python vla-scripts/smoke_test_svgd.py
if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "  Setup complete!"
    echo ""
    echo "  Launch SVGD training:"
    echo "  CUDA_VISIBLE_DEVICES=0 python vla-scripts/finetune_svgd_ensemble.py \\"
    echo "    --dataset_name libero_goal_no_noops \\"
    echo "    --max_steps 50000 \\"
    echo "    --num_steps_before_decay 40000"
    echo "============================================================"
else
    echo "  Smoke test FAILED — check the output above"
    exit 1
fi
