"""
compute_ensemble_uncertainty.py

对 LIBERO-10 数据集的所有帧，用 Deep Ensemble 的每个成员各推理一次，
计算预测动作的方差作为 epistemic uncertainty 分数，保存为 .npz 文件。

用法示例：
    CUDA_VISIBLE_DEVICES=0 python vla-scripts/compute_ensemble_uncertainty.py \
        --vla_path openvla/openvla-7b \
        --data_root_dir /PATH/TO/modified_libero_rlds/ \
        --dataset_name libero_10_no_noops \
        --checkpoint_dirs runs/ensemble/seed42 runs/ensemble/seed1337 runs/ensemble/seed7 \
        --output_path uncertainty_scores.npz
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import draccus
import numpy as np
import torch
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from experiments.robot.openvla_utils import (
    check_model_logic_mismatch,
    find_checkpoint_file,
    load_component_state_dict,
    model_is_on_hf_hub,
    update_auto_map,
)
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.action_heads import L1RegressionActionHead
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


@dataclass
class EnsembleUncertaintyConfig:
    # fmt: off
    vla_path: str = "openvla/openvla-7b"                      # 基础模型路径（HF Hub 或本地）
    data_root_dir: Path = Path("datasets/rlds")               # RLDS 数据集根目录
    dataset_name: str = "libero_10_no_noops"                  # 数据集名称
    checkpoint_dirs: List[str] = field(default_factory=list)  # ensemble 成员 checkpoint 目录列表
    output_path: str = "uncertainty_scores.npz"               # 输出文件路径
    lora_rank: int = 32                                       # LoRA rank（必须与训练时一致）
    use_proprio: bool = True                                  # 是否使用本体感知状态
    num_images_in_input: int = 2                              # 输入图像数量（1=仅第三人称，2=含腕部）
    unnorm_key: str = "libero_10_no_noops"                    # 动作反归一化 key
    # fmt: on


class RLDSDatasetWithTimestep(RLDSDataset):
    """RLDSDataset 子类，在 yield 的 batch 中额外附加 timestep 字段。

    timestep 来自 rlds_batch["observation"]["timestep"][0]，由 RLDS pipeline 注入。
    用于在推理脚本中确定性地分配 (episode_idx, step_idx) 帧 ID。
    """

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            sample = self.batch_transform(rlds_batch)
            # 附加 timestep，用于 episode 边界检测
            sample["_timestep"] = int(rlds_batch["observation"]["timestep"][0])
            yield sample


def register_openvla_if_local(vla_path: str):
    """如果是本地 checkpoint，注册 OpenVLA 到 HF Auto Classes。"""
    if not model_is_on_hf_hub(vla_path):
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
        update_auto_map(vla_path)
        check_model_logic_mismatch(vla_path)


def load_ensemble_member(
    vla_path: str, checkpoint_dir: str, lora_rank: int
) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """加载单个 ensemble 成员（基础模型 + LoRA adapter + action head）。"""
    from peft import LoraConfig, get_peft_model

    print(f"  Loading ensemble member from: {checkpoint_dir}")
    register_openvla_if_local(vla_path)

    # 加载基础模型
    vla = AutoModelForVision2Seq.from_pretrained(
        vla_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # 应用 LoRA 配置（必须与训练时完全一致）
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=min(lora_rank, 16),
        lora_dropout=0.0,
        target_modules="all-linear",
        init_lora_weights="gaussian",
    )
    vla = get_peft_model(vla, lora_config)

    # 加载保存的 LoRA adapter 权重（finetune.py 保存的是完整 state_dict）
    adapter_checkpoint = find_checkpoint_file(checkpoint_dir, "adapter_model")
    state_dict = torch.load(adapter_checkpoint, map_location="cpu")
    # 只加载 LoRA 参数（过滤掉非 LoRA 的 key）
    lora_state = {k: v for k, v in state_dict.items() if "lora_" in k}
    vla.load_state_dict(lora_state, strict=False)

    vla.eval()
    vla = vla.to(DEVICE)

    # 加载 dataset stats 用于动作反归一化
    stats_path = os.path.join(checkpoint_dir, "dataset_statistics.json")
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            vla.norm_stats = json.load(f)

    # 加载 action head
    action_head = L1RegressionActionHead(
        input_dim=vla.llm_dim, hidden_dim=vla.llm_dim, action_dim=ACTION_DIM
    )
    action_head_path = find_checkpoint_file(checkpoint_dir, "action_head")
    action_head.load_state_dict(load_component_state_dict(action_head_path))
    action_head = action_head.to(torch.bfloat16).to(DEVICE)
    action_head.eval()

    return vla, action_head


def run_inference_on_batch(
    vla: torch.nn.Module,
    action_head: torch.nn.Module,
    batch: Dict[str, Any],
    unnorm_key: str,
) -> np.ndarray:
    """对单帧运行推理，返回预测的 action chunk，shape: (NUM_ACTIONS_CHUNK, ACTION_DIM)。"""
    with torch.inference_mode():
        input_ids = batch["input_ids"].unsqueeze(0).to(DEVICE)
        attention_mask = batch["attention_mask"].unsqueeze(0).to(DEVICE)
        pixel_values = batch["pixel_values"].unsqueeze(0).to(DEVICE, dtype=torch.bfloat16)

        proprio = None
        if "proprio" in batch and batch["proprio"] is not None:
            proprio_val = batch["proprio"]
            if not isinstance(proprio_val, torch.Tensor):
                proprio_val = torch.tensor(proprio_val)
            proprio = proprio_val.unsqueeze(0).to(DEVICE, dtype=torch.bfloat16)

        _, predicted_actions = vla.predict_action(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            unnorm_key=unnorm_key,
            do_sample=False,
            proprio=proprio,
            action_head=action_head,
        )

    # predicted_actions: list of np.ndarray, each shape (ACTION_DIM,)
    return np.stack(predicted_actions, axis=0)  # (NUM_ACTIONS_CHUNK, ACTION_DIM)


def compute_ensemble_uncertainty(cfg: EnsembleUncertaintyConfig):
    assert len(cfg.checkpoint_dirs) >= 2, "需要至少 2 个 ensemble 成员"

    # 加载 processor（所有成员共用）
    print(f"Loading processor from: {cfg.vla_path}")
    register_openvla_if_local(cfg.vla_path)
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)

    # 获取 image_sizes（加载第一个模型后立即卸载）
    print("Loading first model to get image_sizes...")
    vla_tmp, _ = load_ensemble_member(cfg.vla_path, cfg.checkpoint_dirs[0], cfg.lora_rank)
    resize_resolution = tuple(vla_tmp.config.image_sizes)
    del vla_tmp
    torch.cuda.empty_cache()

    # 构建确定性数据集（shuffle_buffer_size=1，不做图像增强）
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=(cfg.num_images_in_input > 1),
        use_proprio=cfg.use_proprio,
    )
    print(f"Building dataset: {cfg.dataset_name} (shuffle disabled)")
    dataset = RLDSDatasetWithTimestep(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=resize_resolution,
        shuffle_buffer_size=1,
        image_aug=False,
        train=True,
    )

    # 第一次遍历：收集所有帧的 (episode_idx, step_idx) 和 batch
    # 注意：整个数据集的 batch 会占用较多内存（~几GB），如果内存不足可改为每个模型单独遍历
    print("First pass: collecting all frames...")
    all_batches = []
    episode_indices_list = []
    step_indices_list = []
    episode_idx = 0
    prev_step = None

    for batch in dataset:
        current_step = batch.pop("_timestep")  # 取出 timestep，不传给模型
        if prev_step is not None and current_step == 0:
            episode_idx += 1
        prev_step = current_step
        episode_indices_list.append(episode_idx)
        step_indices_list.append(current_step)
        all_batches.append(batch)

    n_frames = len(all_batches)
    print(f"Total frames: {n_frames} across {episode_idx + 1} episodes")

    # 对每个 ensemble 成员推理
    all_predictions = []  # list of (N_frames, NUM_ACTIONS_CHUNK, ACTION_DIM)

    for i, ckpt_dir in enumerate(cfg.checkpoint_dirs):
        print(f"\n[{i+1}/{len(cfg.checkpoint_dirs)}] Running inference: {ckpt_dir}")
        vla, action_head = load_ensemble_member(cfg.vla_path, ckpt_dir, cfg.lora_rank)

        member_preds = []
        for j, batch in enumerate(all_batches):
            if j % 500 == 0:
                print(f"  Frame {j}/{n_frames}")
            pred = run_inference_on_batch(vla, action_head, batch, cfg.unnorm_key)
            member_preds.append(pred)

        all_predictions.append(np.stack(member_preds, axis=0))  # (N_frames, chunk, action_dim)

        del vla, action_head
        torch.cuda.empty_cache()

    # 计算每帧方差：在 model 维度上计算，然后对 chunk 和 action_dim 取均值
    print("\nComputing per-frame variance...")
    preds_array = np.stack(all_predictions, axis=0)  # (N_models, N_frames, chunk, action_dim)
    variances = preds_array.var(axis=0).mean(axis=(1, 2))  # (N_frames,)

    print(f"\nVariance stats:")
    print(f"  min:  {variances.min():.6f}")
    print(f"  max:  {variances.max():.6f}")
    print(f"  mean: {variances.mean():.6f}")
    print(f"  std:  {variances.std():.6f}")
    print(f"  frames with var > mean: {(variances > variances.mean()).sum()} / {n_frames}")

    np.savez(
        cfg.output_path,
        episode_indices=np.array(episode_indices_list),
        step_indices=np.array(step_indices_list),
        variances=variances,
    )
    print(f"\nSaved to: {cfg.output_path}")


@draccus.wrap()
def main(cfg: EnsembleUncertaintyConfig):
    compute_ensemble_uncertainty(cfg)


if __name__ == "__main__":
    main()
