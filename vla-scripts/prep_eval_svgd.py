"""
prep_eval_svgd.py

Merges each SVGD-LoRA adapter into the base model and prepares a standard
eval checkpoint directory that run_libero_eval.py can consume directly.

Usage:
    python vla-scripts/prep_eval_svgd.py \
        --base_model /root/autodl-tmp/openvla-7b \
        --checkpoint_dir /root/autodl-tmp/runs/svgd-K2-r32-lam0.1-accum1+libero_goal_no_noops/step_0050000 \
        --output_dir /root/autodl-tmp/eval_checkpoints/svgd_step50000
"""

import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import draccus
import torch
from peft import PeftModel
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.robot.openvla_utils import update_auto_map, check_model_logic_mismatch
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor


# Files to copy verbatim from the training checkpoint dir into each merged dir
COPY_FILES = [
    "dataset_statistics.json",
    "added_tokens.json",
    "preprocessor_config.json",
    "processor_config.json",
    "processing_prismatic.py",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
]


@dataclass
class PrepConfig:
    base_model: Union[str, Path] = "/root/autodl-tmp/openvla-7b"
    checkpoint_dir: Union[str, Path] = ""   # step_XXXXXXX training checkpoint
    output_dir: Union[str, Path] = ""       # where merged checkpoints are written
    num_particles: int = 2


@draccus.wrap()
def main(cfg: PrepConfig) -> None:
    ckpt_dir = Path(cfg.checkpoint_dir)
    out_dir = Path(cfg.output_dir)
    assert ckpt_dir.exists(), f"checkpoint_dir not found: {ckpt_dir}"

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    update_auto_map(str(cfg.base_model))
    check_model_logic_mismatch(str(cfg.base_model))

    for k in range(cfg.num_particles):
        adapter_path = ckpt_dir / f"lora_{k}" / f"lora_{k}"
        merged_dir = out_dir / f"merged_lora_{k}"
        merged_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Processing lora_{k}")
        print(f"  adapter : {adapter_path}")
        print(f"  output  : {merged_dir}")

        # Load base model
        print("  Loading base model...")
        t0 = time.time()
        vla = AutoModelForVision2Seq.from_pretrained(
            str(cfg.base_model),
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

        # Merge LoRA
        print("  Merging LoRA adapter...")
        peft_model = PeftModel.from_pretrained(vla, str(adapter_path))
        peft_model = peft_model.to("cuda")
        merged = peft_model.merge_and_unload()
        merged.save_pretrained(str(merged_dir))
        del peft_model, merged, vla
        torch.cuda.empty_cache()
        print(f"  Merge done in {time.time()-t0:.1f}s")

        # Copy tokenizer / processor files
        print("  Copying support files...")
        for fname in COPY_FILES:
            src = ckpt_dir / fname
            if src.exists():
                shutil.copy2(src, merged_dir / fname)
            else:
                print(f"  WARNING: {fname} not found in checkpoint_dir, skipping")

        # Copy per-particle action_head and proprio_projector.
        # New format: action_head_{k}--{step}_checkpoint.pt
        # Old format (shared): action_head--{step}_checkpoint.pt (fallback)
        matched = list(ckpt_dir.glob(f"action_head_{k}--*.pt"))
        if not matched:
            matched = list(ckpt_dir.glob("action_head--*.pt"))  # old shared format
        for pt_file in matched:
            shutil.copy2(pt_file, merged_dir / pt_file.name)
            print(f"  Copied {pt_file.name}")

        matched_pp = list(ckpt_dir.glob(f"proprio_projector_{k}--*.pt"))
        if not matched_pp:
            matched_pp = list(ckpt_dir.glob("proprio_projector--*.pt"))  # old shared format
        for pt_file in matched_pp:
            shutil.copy2(pt_file, merged_dir / pt_file.name)
            print(f"  Copied {pt_file.name}")

        print(f"  Ready: {merged_dir}")

    print("\nAll adapters merged. Eval checkpoints at:")
    for k in range(cfg.num_particles):
        print(f"  lora_{k}: {out_dir}/merged_lora_{k}")


if __name__ == "__main__":
    main()
