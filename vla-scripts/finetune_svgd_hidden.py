"""
finetune_svgd_hidden.py

SVGD ensemble with repulsion in LLM hidden state space.

repulsion 作用在 action token 位置的 LLM 隐状态上（L2 归一化后），
而非最终的动作预测。这样多样性来自特征表示空间，两个粒子都可以
用同一 action head 解码出正确的动作，避免一个粒子被推偏。

  L_total = (L_task_0 + L_task_1 + λ(t) · L_repulsion) / grad_accum_steps

  L_repulsion = exp(-d_h / max(h_ema, 1e-6))
  d_h = mean_over_batch ||norm(mean_seq(h_0)) - norm(mean_seq(h_1))||_2^2
  h_0, h_1: (B, chunk*dim, llm_dim)  action token 位置的隐状态

  per-particle action_heads 保留（与 ensemble 版本对齐，只改 repulsion 空间）。
"""

import os
import random
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import queue
import threading

import draccus
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

sys.path.insert(0, "/root/LIBERO")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from experiments.robot.openvla_utils import check_model_logic_mismatch, update_auto_map
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.action_heads import L1RegressionActionHead
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.projectors import ProprioProjector
from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    NUM_ACTIONS_CHUNK,
    PROPRIO_DIM,
)
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics



class _PrefetchLoader:
    """Overlap data loading with GPU compute by prefetching in a background thread."""

    def __init__(self, loader, buffer_size: int = 2):
        self._loader = loader
        self._buffer_size = buffer_size

    def __iter__(self):
        q: queue.Queue = queue.Queue(maxsize=self._buffer_size)
        sentinel = object()

        def _worker():
            try:
                for item in self._loader:
                    q.put(item)
            finally:
                q.put(sentinel)

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        while True:
            item = q.get()
            if item is sentinel:
                break
            yield item
        t.join()

    def __len__(self):
        return len(self._loader)


@dataclass
class SVGDEnsembleConfig:
    # fmt: off
    vla_path: str                   = "/root/autodl-tmp/openvla-7b"
    data_root_dir: Path             = Path("/root/autodl-tmp/modified_libero_rlds")
    dataset_name: str               = "libero_goal_no_noops"
    run_root_dir: Path              = Path("/root/autodl-tmp/runs")
    shuffle_buffer_size: int        = 50_000

    # Architecture
    num_images_in_input: int        = 2       # 2 = third-person + wrist
    use_proprio: bool               = True

    # Training
    batch_size: int                 = 8       # micro-batch; 96 GB GPU
    grad_accum_steps: int           = 1       # effective batch = batch_size * grad_accum_steps
    learning_rate: float            = 5e-4
    num_steps_before_decay: int     = 40_000  # 80 % of max_steps
    max_steps: int                  = 50_000
    save_freq: int                  = 10_000
    save_latest_checkpoint_only: bool = False
    image_aug: bool                 = True
    seed: int                       = 42
    use_compile: bool               = False   # torch.compile action_heads for ~10% speedup

    # LoRA
    lora_rank: int                  = 32
    lora_dropout: float             = 0.0
    num_particles: int              = 2

    # SVGD
    svgd_lambda: float              = 0.1     # λ_max
    lambda_warmup_steps: int        = 5_000   # effective steps with λ=0
    lambda_ramp_steps: int          = 5_000   # effective steps to ramp 0→λ_max

    # Logging
    use_wandb: bool                 = False
    wandb_entity: str               = "your-entity"
    wandb_project: str              = "openvla-svgd"
    wandb_log_freq: int             = 100
    run_id_note: Optional[str]      = None
    # fmt: on


# ── helpers ──────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_lambda(step: int, cfg: SVGDEnsembleConfig) -> float:
    """λ schedule based on effective gradient steps."""
    if step < cfg.lambda_warmup_steps:
        return 0.0
    ramp_progress = min(
        (step - cfg.lambda_warmup_steps) / max(cfg.lambda_ramp_steps, 1), 1.0
    )
    return cfg.svgd_lambda * ramp_progress


def enable_all_lora_params(vla) -> None:
    """Restore requires_grad=True for ALL LoRA weights.

    PEFT's set_adapter() freezes non-active adapters.  After K forward passes
    we must re-enable all LoRA params so backward() fills .grad for every
    adapter regardless of which one was active last.
    """
    for name, param in vla.named_parameters():
        if "lora_" in name:
            param.requires_grad = True


def compute_grad_norm(params) -> float:
    total_sq = sum(
        p.grad.detach().float().norm() ** 2
        for p in params
        if p.grad is not None
    )
    return float(total_sq ** 0.5)


# ── repulsion in hidden state space ───────────────────────────────────────────

def compute_repulsion(
    h_0: torch.Tensor,
    h_1: torch.Tensor,
    h_ema: float,
) -> Tuple[torch.Tensor, float]:
    """
    RBF-kernel repulsion on L2-normalized action-token hidden states.

    h_0, h_1: (B, chunk*dim, llm_dim) — action position hidden states.
    先对 seq 维取均值得 (B, llm_dim)，再 L2 归一化到单位球，
    然后计算 RBF kernel。距离范围 [0, 4]，与 llm_dim 无关。

    detach trick 保证梯度各自流入对应 LoRA。
    """
    # mean over sequence → (B, llm_dim), 归一化到单位球
    h0 = F.normalize(h_0.float().mean(dim=1), dim=-1)   # (B, D)
    h1 = F.normalize(h_1.float().mean(dim=1), dim=-1)   # (B, D)

    h_safe = max(h_ema, 1e-6)

    d_fwd = ((h0 - h1.detach()) ** 2).sum(dim=1).mean()
    d_rev = ((h0.detach() - h1) ** 2).sum(dim=1).mean()

    k_fwd = torch.exp(-d_fwd / h_safe)
    k_rev = torch.exp(-d_rev / h_safe)
    loss_rep = (k_fwd + k_rev) * 0.5

    with torch.no_grad():
        d_sym = ((h0 - h1) ** 2).sum(dim=1).mean()
        k_val = float(torch.exp(-d_sym / h_safe))

    return loss_rep, k_val


# ── single-particle forward ───────────────────────────────────────────────────

def forward_one_particle(
    vla,
    action_head: L1RegressionActionHead,
    proprio_projector: Optional[nn.Module],
    batch: dict,
    device_id: int,
    num_patches: int,
    use_proprio: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    One forward pass with the currently-active LoRA adapter.
    Returns (l1_loss, predicted_actions (B, chunk, action_dim),
             actions_hidden (B, chunk*dim, llm_dim)).
    actions_hidden 用于 hidden state repulsion。
    """
    gt_actions = batch["actions"].to(device_id).to(torch.bfloat16)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = vla(
            input_ids=batch["input_ids"].to(device_id),
            attention_mask=batch["attention_mask"].to(device_id),
            pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
            labels=batch["labels"].to(device_id),
            output_hidden_states=True,
            proprio=batch["proprio"].to(device_id) if use_proprio else None,
            proprio_projector=proprio_projector if use_proprio else None,
            use_film=False,
        )

    last_hidden = output.hidden_states[-1]        # (B, seq, D)
    text_hidden = last_hidden[:, num_patches:-1]   # strip vision prefix + last token
    gt_token_ids = batch["labels"][:, 1:].to(device_id)
    cur_mask = get_current_action_mask(gt_token_ids)
    nxt_mask = get_next_actions_mask(gt_token_ids)
    B = batch["input_ids"].shape[0]

    actions_hidden = (
        text_hidden[cur_mask | nxt_mask]
        .reshape(B, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
        .to(torch.bfloat16)
    )  # (B, chunk*dim, D)

    pred = action_head.predict_action(actions_hidden)   # (B, chunk, action_dim)
    loss = F.l1_loss(pred, gt_actions)
    return loss, pred, actions_hidden


# ── checkpoint ────────────────────────────────────────────────────────────────

def save_checkpoint(
    cfg: SVGDEnsembleConfig,
    run_dir: Path,
    step: int,
    vla,
    processor,
    action_heads: List[nn.Module],
    proprio_projectors: List[Optional[nn.Module]],
    train_dataset,
    optimizer,
    scheduler,
    h_ema: float,
) -> None:
    if cfg.save_latest_checkpoint_only:
        ckpt_dir = run_dir / "latest"
    else:
        ckpt_dir = run_dir / f"step_{step:07d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    processor.save_pretrained(ckpt_dir)
    save_dataset_statistics(train_dataset.dataset_statistics, ckpt_dir)

    # Save each LoRA adapter independently
    for k in range(cfg.num_particles):
        adapter_dir = ckpt_dir / f"lora_{k}"
        adapter_dir.mkdir(exist_ok=True)
        vla.save_pretrained(str(adapter_dir), selected_adapters=[f"lora_{k}"])

    # Save per-particle action_head and proprio_projector
    for k in range(cfg.num_particles):
        torch.save(
            action_heads[k].state_dict(),
            ckpt_dir / f"action_head_{k}--{step}_checkpoint.pt",
        )
        if cfg.use_proprio and proprio_projectors[k] is not None:
            torch.save(
                proprio_projectors[k].state_dict(),
                ckpt_dir / f"proprio_projector_{k}--{step}_checkpoint.pt",
            )

    # Full training state for resume
    torch.save(
        {
            "step": step,
            "h_ema": h_ema,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
        },
        ckpt_dir / "training_state.pt",
    )

    print(f"[step {step:6d}] Saved → {ckpt_dir}")


# ── sanity check ──────────────────────────────────────────────────────────────

def sanity_check_diversity(
    vla,
    action_heads: List[nn.Module],
    proprio_projectors: List[Optional[nn.Module]],
    batch: dict,
    device_id: int,
    num_patches: int,
    use_proprio: bool,
    step: int,
) -> None:
    """Verify the two LoRA adapters produce different predictions at step ~100."""
    with torch.no_grad():
        preds = []
        for k in range(2):
            vla.set_adapter(f"lora_{k}")
            _, pred = forward_one_particle(
                vla, action_heads[k], proprio_projectors[k],
                batch, device_id, num_patches, use_proprio,
            )
            preds.append(pred.float())
        div = ((preds[0] - preds[1]) ** 2).sum(dim=-1).mean().sqrt().item()
        identical = torch.allclose(preds[0], preds[1], atol=1e-5)
        status = "IDENTICAL (B still ~0?)" if identical else "DIFFERENT — diversity OK"
        print(f"[sanity @ step {step}] pred_diversity={div:.6f}  {status}")


# ── main ──────────────────────────────────────────────────────────────────────

@draccus.wrap()
def train(cfg: SVGDEnsembleConfig) -> None:
    print(
        f"SVGD-LoRA Ensemble  dataset={cfg.dataset_name}  "
        f"K={cfg.num_particles}  λ_max={cfg.svgd_lambda}  r={cfg.lora_rank}  "
        f"grad_accum={cfg.grad_accum_steps}  eff_batch={cfg.batch_size * cfg.grad_accum_steps}"
    )

    set_seed(cfg.seed)

    device_id = 0
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

    run_id = (
        f"svgd-hidden-K{cfg.num_particles}-r{cfg.lora_rank}"
        f"-lam{cfg.svgd_lambda}-accum{cfg.grad_accum_steps}"
        f"+{cfg.dataset_name}"
    )
    if cfg.run_id_note:
        run_id += f"--{cfg.run_id_note}"
    run_dir = Path(cfg.run_root_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")

    if cfg.use_wandb:
        import wandb
        wandb.init(
            entity=cfg.wandb_entity, project=cfg.wandb_project,
            name=run_id, config=vars(cfg),
        )

    # ── Load base model ───────────────────────────────────────────────────────
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    update_auto_map(cfg.vla_path)
    check_model_logic_mismatch(cfg.vla_path)

    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="sdpa",
    ).to(device_id)

    # Must set BEFORE PEFT wrapping (vision_backbone inaccessible after)
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)

    # ── Attach K LoRA adapters with explicit per-adapter seeds ────────────────
    lora_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=min(cfg.lora_rank, 16),
        lora_dropout=cfg.lora_dropout,
        target_modules="all-linear",
        init_lora_weights="gaussian",
    )

    torch.manual_seed(42)
    vla = get_peft_model(vla, lora_config, adapter_name="lora_0")

    torch.manual_seed(43)
    vla.add_adapter("lora_1", lora_config)

    set_seed(cfg.seed)  # restore global seed after per-adapter init

    # PEFT's add_adapter freezes non-active adapters; re-enable all before optimizer
    enable_all_lora_params(vla)
    lora_params = sum(p.numel() for p in vla.parameters() if p.requires_grad)
    print(f"LoRA trainable params ({cfg.num_particles} adapters): {lora_params:,}")

    # ── Per-particle action heads and proprio projectors ──────────────────────
    llm_dim = vla.model.language_model.config.hidden_size

    action_heads: List[nn.Module] = []
    proprio_projectors: List[Optional[nn.Module]] = []

    for k in range(cfg.num_particles):
        ah = (
            L1RegressionActionHead(input_dim=llm_dim, hidden_dim=llm_dim, action_dim=ACTION_DIM)
            .to(torch.bfloat16)
            .to(device_id)
        )
        if cfg.use_compile:
            ah = torch.compile(ah)
        action_heads.append(ah)

        pp = (
            ProprioProjector(llm_dim=llm_dim, proprio_dim=PROPRIO_DIM).to(device_id)
            if cfg.use_proprio
            else None
        )
        proprio_projectors.append(pp)

    print(f"Created {cfg.num_particles} independent action_heads and proprio_projectors")

    # ── Number of vision patches ──────────────────────────────────────────────
    NUM_PATCHES = (
        vla.model.vision_backbone.get_num_patches()
        * vla.model.vision_backbone.get_num_images_in_input()
    )
    if cfg.use_proprio:
        NUM_PATCHES += 1
    print(f"NUM_PATCHES={NUM_PATCHES}")

    # ── Optimizer ────────────────────────────────────────────────────────────
    trainable = [p for p in vla.parameters() if p.requires_grad]
    for k in range(cfg.num_particles):
        trainable += list(action_heads[k].parameters())
        if cfg.use_proprio and proprio_projectors[k] is not None:
            trainable += list(proprio_projectors[k].parameters())
    print(f"Total trainable params: {sum(p.numel() for p in trainable):,}")

    optimizer = AdamW(trainable, lr=cfg.learning_rate)
    scheduler = MultiStepLR(optimizer, milestones=[cfg.num_steps_before_decay], gamma=0.1)

    # ── Dataset ───────────────────────────────────────────────────────────────
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=(cfg.num_images_in_input > 1),
        use_proprio=cfg.use_proprio,
    )
    train_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.model.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )
    save_dataset_statistics(train_dataset.dataset_statistics, run_dir)

    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side="right",
    )
    dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )

    # ── Training state ────────────────────────────────────────────────────────
    h_ema: float = 1.0  # EMA bandwidth, updated every micro-batch

    metric_keys = [
        "loss_total", "loss_task", "loss_task_0", "loss_task_1",
        "task_loss_diff", "loss_repulsion", "kernel_value",
        "dist_sq", "h_ema", "lambda", "pred_diversity",
        "grad_norm_lora_0", "grad_norm_lora_1",
    ]
    recent: Dict[str, deque] = {k: deque(maxlen=cfg.wandb_log_freq) for k in metric_keys}

    # param groups for per-adapter grad norm
    lora_0_params = [p for n, p in vla.named_parameters() if "lora_0" in n]
    lora_1_params = [p for n, p in vla.named_parameters() if "lora_1" in n]
    all_lora_params = lora_0_params + lora_1_params

    # ── Train ────────────────────────────────────────────────────────────────
    vla.train()
    for k in range(cfg.num_particles):
        action_heads[k].train()
        if proprio_projectors[k] is not None:
            proprio_projectors[k].train()

    optimizer.zero_grad()
    global_step = 0
    micro_step = 0
    t0 = time.time()

    prefetch_loader = _PrefetchLoader(dataloader, buffer_size=4)
    with tqdm.tqdm(total=cfg.max_steps, desc="SVGD-LoRA") as pbar:
        for batch in prefetch_loader:
            micro_step += 1

            # ── K forward passes (per-particle action_head, different LoRA) ───
            task_losses: List[torch.Tensor] = []
            all_preds: List[torch.Tensor] = []
            all_hiddens: List[torch.Tensor] = []

            for k in range(cfg.num_particles):
                vla.set_adapter(f"lora_{k}")
                loss_k, pred_k, hidden_k = forward_one_particle(
                    vla, action_heads[k], proprio_projectors[k],
                    batch, device_id, NUM_PATCHES, cfg.use_proprio,
                )
                task_losses.append(loss_k)
                all_preds.append(pred_k)
                all_hiddens.append(hidden_k)

            # ── Re-enable all LoRA grads before backward ──────────────────────
            for p in all_lora_params:
                p.requires_grad = True

            # ── Detached stats for EMA (hidden state distances) ───────────────
            with torch.no_grad():
                h0_n = F.normalize(all_hiddens[0].float().mean(dim=1), dim=-1)
                h1_n = F.normalize(all_hiddens[1].float().mean(dim=1), dim=-1)
                d_sq_val = ((h0_n - h1_n) ** 2).sum(dim=1).mean().item()
                # action prediction diversity for monitoring only
                p0 = all_preds[0].float().reshape(all_preds[0].shape[0], -1)
                p1 = all_preds[1].float().reshape(all_preds[1].shape[0], -1)
                diversity = ((p0 - p1) ** 2).sum(dim=1).sqrt().mean().item()

            # EMA bandwidth 跟踪 hidden state 距离（范围 [0,4]，初值 1.0 合理）
            h_ema = 0.9 * h_ema + 0.1 * d_sq_val

            # ── λ schedule (based on effective step counter) ──────────────────
            lam = get_lambda(global_step, cfg)

            # ── Total loss with optional repulsion ────────────────────────────
            loss_task_sum = sum(task_losses)

            if lam > 0.0:
                # repulsion 作用在 hidden state 上
                loss_rep, k_val = compute_repulsion(all_hiddens[0], all_hiddens[1], h_ema)
                loss_total = (loss_task_sum + lam * loss_rep) / cfg.grad_accum_steps
            else:
                loss_rep = torch.zeros(1, device=all_preds[0].device)
                k_val = float(torch.exp(torch.tensor(-d_sq_val / max(h_ema, 1e-6))))
                loss_total = loss_task_sum / cfg.grad_accum_steps

            loss_total.backward()

            # ── Optimizer step every grad_accum_steps micro-batches ───────────
            if micro_step % cfg.grad_accum_steps == 0:
                # Grad norms before zero_grad
                gn0 = compute_grad_norm(lora_0_params)
                gn1 = compute_grad_norm(lora_1_params)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                loss_t0 = task_losses[0].item()
                loss_t1 = task_losses[1].item()
                loss_task_avg = (loss_t0 + loss_t1) / 2.0
                current_lr = optimizer.param_groups[0]["lr"]

                recent["loss_total"].append(loss_total.item() * cfg.grad_accum_steps)
                recent["loss_task"].append(loss_task_avg)
                recent["loss_task_0"].append(loss_t0)
                recent["loss_task_1"].append(loss_t1)
                recent["task_loss_diff"].append(abs(loss_t0 - loss_t1))
                recent["loss_repulsion"].append(loss_rep.item())
                recent["kernel_value"].append(k_val)
                recent["dist_sq"].append(d_sq_val)
                recent["h_ema"].append(h_ema)
                recent["lambda"].append(lam)
                recent["pred_diversity"].append(diversity)
                recent["grad_norm_lora_0"].append(gn0)
                recent["grad_norm_lora_1"].append(gn1)

                pbar.update()
                pbar.set_postfix({
                    "task": f"{loss_task_avg:.4f}",
                    "rep": f"{loss_rep.item():.4f}",
                    "div": f"{diversity:.4f}",
                    "λ": f"{lam:.3f}",
                })

                if global_step % cfg.wandb_log_freq == 0:
                    log = {k: sum(v) / len(v) for k, v in recent.items() if v}
                    log["lr"] = current_lr
                    log["steps_per_sec"] = global_step / (time.time() - t0)
                    print(
                        f"[{global_step:6d}/{cfg.max_steps}] "
                        f"task={log['loss_task']:.4f} "
                        f"(t0={log['loss_task_0']:.4f} t1={log['loss_task_1']:.4f} "
                        f"Δ={log['task_loss_diff']:.4f})  "
                        f"rep={log['loss_repulsion']:.4f}  "
                        f"k={log['kernel_value']:.4f}  h={log['h_ema']:.4f}  "
                        f"λ={log['lambda']:.3f}  div={log['pred_diversity']:.4f}  "
                        f"lr={current_lr:.2e}  ({log['steps_per_sec']:.2f} it/s)"
                    )
                    if cfg.use_wandb:
                        import wandb
                        wandb.log({f"train/{k}": v for k, v in log.items()},
                                  step=global_step)

                # Sanity check at step 100: B should be non-zero after 100 effective steps
                if global_step == 100:
                    sanity_check_diversity(
                        vla, action_heads, proprio_projectors,
                        batch, device_id, NUM_PATCHES, cfg.use_proprio, global_step,
                    )

                if global_step % cfg.save_freq == 0:
                    save_checkpoint(
                        cfg, run_dir, global_step, vla, processor,
                        action_heads, proprio_projectors, train_dataset,
                        optimizer, scheduler, h_ema,
                    )

                if global_step >= cfg.max_steps:
                    print(f"Reached max_steps={cfg.max_steps}, stopping.")
                    break

            if global_step >= cfg.max_steps:
                break

    save_checkpoint(
        cfg, run_dir, global_step, vla, processor,
        action_heads, proprio_projectors, train_dataset,
        optimizer, scheduler, h_ema,
    )
    print("Training complete.")


if __name__ == "__main__":
    train()
