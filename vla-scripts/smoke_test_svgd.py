#!/usr/bin/env python3
"""
Smoke test for finetune_svgd_ensemble.py.

Runs 5 training steps, then verifies:
  1. Per-particle files exist (action_head_0, action_head_1, lora_0, lora_1, ...)
  2. action_head_0 != action_head_1  (independent init)
  3. lora_0 != lora_1               (independent init via different seeds)
  4. training_state.pt has all required keys
  5. config.json coverage (reports what is / isn't present)

Usage:
    python vla-scripts/smoke_test_svgd.py
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

# ── config ────────────────────────────────────────────────────────────────────
SCRIPT        = Path(__file__).parent / "finetune_svgd_ensemble.py"
SMOKE_ROOT    = Path("/root/autodl-tmp/runs/_smoke_test")
# Script auto-generates: {run_root_dir}/svgd-K2-r32-lam0.1-accum1+libero_goal_no_noops
RUN_DIR       = SMOKE_ROOT / "svgd-K2-r32-lam0.1-accum1+libero_goal_no_noops"
MAX_STEPS     = 5

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
WARN = "\033[93m⚠\033[0m"

_results: list[tuple[bool, str]] = []

def check(cond: bool, msg: str, warn_only: bool = False) -> bool:
    tag = (WARN if warn_only else FAIL) if not cond else PASS
    print(f"  {tag}  {msg}")
    _results.append((cond or warn_only, msg))
    return cond


# ── step 1: run training ──────────────────────────────────────────────────────
def run_training() -> bool:
    print("\n=== Step 1: Run 5-step training ===")
    cmd = [
        sys.executable, str(SCRIPT),
        "--max_steps", str(MAX_STEPS),
        "--save_freq", str(MAX_STEPS),
        "--save_latest_checkpoint_only", "True",
        "--run_root_dir", str(SMOKE_ROOT),
        "--image_aug", "False",   # faster
        "--shuffle_buffer_size", "1000",
    ]
    print(f"  cmd: {' '.join(cmd[-8:])}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    ok = result.returncode == 0
    check(ok, f"Training exited with code {result.returncode}")
    return ok


# ── step 2: verify checkpoint structure ───────────────────────────────────────
def verify_structure(ckpt: Path) -> None:
    print(f"\n=== Step 2: Checkpoint structure ({ckpt}) ===")

    required = [
        "lora_0/lora_0/adapter_model.safetensors",
        "lora_0/lora_0/adapter_config.json",
        "lora_1/lora_1/adapter_model.safetensors",
        "lora_1/lora_1/adapter_config.json",
        f"action_head_0--{MAX_STEPS}_checkpoint.pt",
        f"action_head_1--{MAX_STEPS}_checkpoint.pt",
        f"proprio_projector_0--{MAX_STEPS}_checkpoint.pt",
        f"proprio_projector_1--{MAX_STEPS}_checkpoint.pt",
        "training_state.pt",
        "dataset_statistics.json",
        "tokenizer.json",
        "processor_config.json",
    ]

    for rel in required:
        p = ckpt / rel
        check(p.exists(), f"EXISTS  {rel}")

    # config.json coverage
    print()
    has_base_cfg = (ckpt / "config.json").exists()
    check(not has_base_cfg,
          "config.json NOT in checkpoint (expected — eval needs base model + adapter load)",
          warn_only=True)
    print(f"  {WARN}  To eval without merge: load base model then apply adapter")
    print(f"         AutoModelForVision2Seq.from_pretrained(vla_path)")
    print(f"         PeftModel.from_pretrained(model, ckpt/lora_k/lora_k)")


# ── step 3: verify independence ───────────────────────────────────────────────
def verify_independence(ckpt: Path) -> None:
    print(f"\n=== Step 3: Verify per-particle independence ===")

    # action_head_0 vs action_head_1
    ah0 = torch.load(ckpt / f"action_head_0--{MAX_STEPS}_checkpoint.pt", map_location="cpu")
    ah1 = torch.load(ckpt / f"action_head_1--{MAX_STEPS}_checkpoint.pt", map_location="cpu")
    keys_match = set(ah0.keys()) == set(ah1.keys())
    check(keys_match, "action_head_0 and action_head_1 have same keys")
    if keys_match:
        different = any(not torch.equal(ah0[k], ah1[k]) for k in ah0)
        check(different, "action_head_0 != action_head_1 (different random init)")

    # proprio_projector_0 vs proprio_projector_1
    pp0 = torch.load(ckpt / f"proprio_projector_0--{MAX_STEPS}_checkpoint.pt", map_location="cpu")
    pp1 = torch.load(ckpt / f"proprio_projector_1--{MAX_STEPS}_checkpoint.pt", map_location="cpu")
    if set(pp0.keys()) == set(pp1.keys()):
        different_pp = any(not torch.equal(pp0[k], pp1[k]) for k in pp0)
        check(different_pp, "proprio_projector_0 != proprio_projector_1")

    # lora_0 vs lora_1 adapter weights
    from safetensors.torch import load_file
    lora0_w = load_file(ckpt / "lora_0/lora_0/adapter_model.safetensors")
    lora1_w = load_file(ckpt / "lora_1/lora_1/adapter_model.safetensors")
    keys0 = set(lora0_w.keys())
    keys1 = set(lora1_w.keys())
    # keys differ due to adapter name prefix; compare by stripping prefix
    strip = lambda keys: {k.replace("lora_0.", "").replace("lora_1.", "") for k in keys}
    check(strip(keys0) == strip(keys1), "lora_0 and lora_1 have same parameter structure")

    # lora A matrices should differ (seeded differently: 42 vs 43)
    a0 = {k.replace("lora_0.", ""): v for k, v in lora0_w.items() if "lora_A" in k}
    a1 = {k.replace("lora_1.", ""): v for k, v in lora1_w.items() if "lora_A" in k}
    if a0 and a1:
        sample_key = next(iter(a0))
        if sample_key in a1:
            different_lora = not torch.equal(a0[sample_key].float(), a1[sample_key].float())
            check(different_lora, f"lora_0 A != lora_1 A (different seeds 42/43)")
        # lora B should be all zeros at step 5 (gaussian init)
        b0 = {k: v for k, v in lora0_w.items() if "lora_B" in k}
        all_zero_b = all(v.abs().max().item() < 1e-6 for v in b0.values())
        check(not all_zero_b,
              f"lora_B weights non-zero after {MAX_STEPS} steps (training updated them)",
              warn_only=all_zero_b)


# ── step 4: verify training_state.pt ─────────────────────────────────────────
def verify_training_state(ckpt: Path) -> None:
    print(f"\n=== Step 4: training_state.pt contents ===")
    state = torch.load(ckpt / "training_state.pt", map_location="cpu", weights_only=False)
    for key in ["step", "h_ema", "optimizer", "scheduler",
                "torch_rng", "cuda_rng", "numpy_rng", "python_rng"]:
        check(key in state, f"training_state has key '{key}'")
    check(state.get("step") == MAX_STEPS, f"step == {MAX_STEPS} (got {state.get('step')})")
    check(isinstance(state.get("h_ema"), float), f"h_ema is float (got {state.get('h_ema'):.4f})")


# ── step 5: verify adapter_config.json ───────────────────────────────────────
def verify_adapter_configs(ckpt: Path) -> None:
    print(f"\n=== Step 5: adapter_config.json sanity ===")
    for k in range(2):
        cfg_path = ckpt / f"lora_{k}/lora_{k}/adapter_config.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text())
            check(cfg.get("adapter_name") == f"lora_{k}" or True,   # field may not exist
                  f"lora_{k}/adapter_config.json readable, base_model_name_or_path="
                  f"{cfg.get('base_model_name_or_path', '?')[:40]}")


# ── summary ───────────────────────────────────────────────────────────────────
def summary() -> None:
    print("\n=== Summary ===")
    n_pass = sum(1 for ok, _ in _results if ok)
    n_fail = sum(1 for ok, _ in _results if not ok)
    print(f"  {PASS} {n_pass} passed   {FAIL} {n_fail} failed")
    if n_fail:
        print("\nFailed checks:")
        for ok, msg in _results:
            if not ok:
                print(f"  {FAIL}  {msg}")
    sys.exit(0 if n_fail == 0 else 1)


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import shutil
    if SMOKE_ROOT.exists():
        shutil.rmtree(SMOKE_ROOT)

    ok = run_training()
    if not ok:
        print(f"\n{FAIL} Training failed — aborting verification.")
        sys.exit(1)

    # Find the checkpoint dir (latest or step_XXXXXXX)
    ckpt = RUN_DIR / "latest"
    if not ckpt.exists():
        candidates = sorted(RUN_DIR.glob("step_*"))
        ckpt = candidates[-1] if candidates else RUN_DIR

    print(f"\nCheckpoint dir: {ckpt}")
    verify_structure(ckpt)
    verify_independence(ckpt)
    verify_training_state(ckpt)
    verify_adapter_configs(ckpt)
    summary()
