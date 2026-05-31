"""
compute_ensemble_uncertainty.py

三种模式：
  infer    — 单张卡推理一个 ensemble 成员，保存 .npy 预测结果
  merge    — 汇总多个 .npy，计算每帧方差，保存 uncertainty_scores.npz
  parallel — 双卡同时推理 K 个成员（lora_k → GPU k），完成后自动 merge（推荐）

推荐用法（parallel，一条命令搞定推理+汇总）：
    python vla-scripts/compute_ensemble_uncertainty.py \
        --mode parallel \
        --vla_path /root/autodl-tmp/openvla-7b \
        --data_root_dir /root/autodl-tmp/modified_libero_rlds \
        --dataset_name libero_goal_no_noops \
        --checkpoint_dirs /root/autodl-tmp/eval_checkpoints/svgd_step50000/merged_lora_0 \
                          /root/autodl-tmp/eval_checkpoints/svgd_step50000/merged_lora_1 \
        --preds_dir /root/autodl-tmp/preds \
        --output_path /root/autodl-tmp/preds/uncertainty_scores.npz \
        --unnorm_key libero_goal_no_noops

手动模式（单卡串行，调试用）：
    CUDA_VISIBLE_DEVICES=0 python vla-scripts/compute_ensemble_uncertainty.py \
        --mode infer --cuda_device 0 \
        --checkpoint_dir /path/to/merged_lora_0 --output_path preds_lora0.npy ...

    python vla-scripts/compute_ensemble_uncertainty.py \
        --mode merge \
        --pred_files preds_lora0.npy preds_lora1.npy \
        --meta_file preds_lora0_meta.npz \
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
from tqdm import tqdm
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from experiments.robot.openvla_utils import (
    check_model_logic_mismatch,
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
from prismatic.vla.datasets.rlds import make_interleaved_dataset


@dataclass
class EnsembleUncertaintyConfig:
    # fmt: off
    mode: str = "infer"                                        # "infer" | "merge" | "parallel"

    # infer 模式参数
    vla_path: str = "openvla/openvla-7b"                      # 基础模型路径（HF Hub 或本地）
    data_root_dir: Path = Path("datasets/rlds")               # RLDS 数据集根目录
    dataset_name: str = "libero_goal_no_noops"                # 数据集名称
    checkpoint_dir: str = ""                                  # 单个 ensemble 成员的 checkpoint 目录
    output_path: str = "preds.npy"                            # 推理结果保存路径（infer: .npy；merge/parallel: .npz）
    lora_rank: int = 32                                       # LoRA rank（必须与训练时一致）
    use_proprio: bool = True                                  # 是否使用本体感知状态
    num_images_in_input: int = 2                              # 输入图像数量（1=仅第三人称，2=含腕部）
    unnorm_key: str = "libero_goal_no_noops"                  # 动作反归一化 key
    cuda_device: int = 0                                      # 使用的 GPU 编号（infer 模式）
    infer_batch_size: int = 16                                # 批量推理 batch size

    # parallel 模式参数
    checkpoint_dir_0: str = ""   # lora_0 checkpoint 目录 → GPU 0
    checkpoint_dir_1: str = ""   # lora_1 checkpoint 目录 → GPU 1
    preds_dir: str = ""          # 中间 .npy 文件保存目录（默认与 output_path 同级）
    max_episodes: int = -1       # 最多处理的 episode 数（-1=全量）

    # merge 模式参数
    pred_files: List[str] = field(default_factory=list)       # 各成员的 .npy 预测文件列表
    meta_file: str = ""                                       # infer 模式保存的 meta 文件（含 episode/step 索引）
    # fmt: on


class RLDSDatasetWithTimestep(RLDSDataset):
    """RLDSDataset 子类，在 yield 的 batch 中额外附加 timestep 字段。

    覆盖 make_dataset 以强制 shuffle_files=False，保证每次运行的文件读取顺序相同，
    使推理结果可以按帧位置直接对应到训练时的权重。
    """

    def make_dataset(self, rlds_config):
        for kwargs in rlds_config["dataset_kwargs_list"]:
            kwargs["shuffle"] = False
        return make_interleaved_dataset(**rlds_config)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            sample = self.batch_transform(rlds_batch)
            sample["_timestep"] = int(rlds_batch["observation"]["timestep"][0])
            task_bytes = rlds_batch.get("task", {}).get("language_instruction", b"")
            sample["_task"] = task_bytes.decode("utf-8") if isinstance(task_bytes, bytes) else str(task_bytes)
            yield sample


def register_openvla_if_local(vla_path: str):
    if not model_is_on_hf_hub(vla_path):
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
        update_auto_map(vla_path)
        check_model_logic_mismatch(vla_path)


def load_ensemble_member(
    vla_path: str, checkpoint_dir: str, lora_rank: int, device: torch.device
) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """加载单个 ensemble 成员（基础模型 + LoRA adapter + action head）。"""
    print(f"  Loading ensemble member from: {checkpoint_dir}")
    register_openvla_if_local(vla_path)

    # 直接用本地类加载，避免 trust_remote_code 加载 HF 缓存的旧版本
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig as LocalOpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction as LocalOpenVLA

    config = LocalOpenVLAConfig.from_pretrained(vla_path)
    vla = LocalOpenVLA.from_pretrained(
        vla_path,
        config=config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        ignore_mismatched_sizes=True,
    )

    # 加载模型权重：
    #   - 若 checkpoint_dir 内有 lora_adapter/，则通过 PEFT 加载后 merge（旧格式）
    #   - 否则视为已合并的完整模型，直接从 checkpoint_dir 加载（新格式 / prep_eval_svgd 输出）
    adapter_dir = os.path.join(checkpoint_dir, "lora_adapter")
    if os.path.isdir(adapter_dir):
        from peft import PeftModel
        vla = PeftModel.from_pretrained(vla, adapter_dir)
        vla = vla.merge_and_unload()
    else:
        # 已合并的模型：直接从 checkpoint_dir 加载权重覆盖 base model
        from safetensors.torch import load_file
        import glob
        shard_files = sorted(glob.glob(os.path.join(checkpoint_dir, "model-*.safetensors")))
        if not shard_files:
            shard_files = glob.glob(os.path.join(checkpoint_dir, "*.safetensors"))
        assert shard_files, f"No safetensors found in {checkpoint_dir}"
        state_dict = {}
        for shard in shard_files:
            state_dict.update(load_file(shard))
        missing, unexpected = vla.load_state_dict(state_dict, strict=False)
        if unexpected:
            print(f"  WARNING: unexpected keys: {unexpected[:5]}")

    vla.eval()
    vla = vla.to(device)

    # 设置 norm_stats
    stats_path = os.path.join(checkpoint_dir, "dataset_statistics.json")
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            vla.norm_stats = json.load(f)

    # 加载 action head（支持旧格式 action_head--*.pt 和新格式 action_head_0--*.pt）
    action_head_files = [
        f for f in os.listdir(checkpoint_dir)
        if "action_head" in f and "checkpoint" in f
    ]
    assert len(action_head_files) == 1, f"Expected 1 action_head checkpoint, found: {action_head_files}"
    action_head_path = os.path.join(checkpoint_dir, action_head_files[0])

    llm_dim = vla.config.text_config.hidden_size
    action_head = L1RegressionActionHead(
        input_dim=llm_dim, hidden_dim=llm_dim, action_dim=ACTION_DIM
    )
    action_head.load_state_dict(load_component_state_dict(action_head_path))
    action_head = action_head.to(torch.bfloat16).to(device)
    action_head.eval()

    return vla, action_head


def run_inference_on_batch(
    vla: torch.nn.Module,
    action_head: torch.nn.Module,
    batch: Dict[str, Any],
    unnorm_key: str,
    device: torch.device,
) -> np.ndarray:
    """单帧推理，返回预测 action chunk，shape: (NUM_ACTIONS_CHUNK, ACTION_DIM)。"""
    with torch.inference_mode():
        input_ids = batch["input_ids"].unsqueeze(0).to(device)
        attention_mask = torch.ones_like(input_ids)
        pixel_values = batch["pixel_values"].unsqueeze(0).to(device, dtype=torch.bfloat16)

        proprio = None
        if "proprio" in batch and batch["proprio"] is not None:
            pv = batch["proprio"]
            if not isinstance(pv, torch.Tensor):
                pv = torch.tensor(pv)
            proprio = pv.unsqueeze(0).to(device, dtype=torch.bfloat16)

        predicted_actions, _ = vla.predict_action(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            unnorm_key=unnorm_key,
            do_sample=False,
            proprio=proprio,
            action_head=action_head,
        )

    return predicted_actions  # np.ndarray (NUM_ACTIONS_CHUNK, ACTION_DIM)


def run_infer(cfg: EnsembleUncertaintyConfig):
    """单成员推理模式：对全量数据集推理，保存预测结果和帧索引元数据。"""
    assert cfg.checkpoint_dir, "--checkpoint_dir 不能为空"

    device = torch.device(f"cuda:{cfg.cuda_device}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载 processor
    print(f"Loading processor from: {cfg.vla_path}")
    register_openvla_if_local(cfg.vla_path)
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)

    # 加载模型获取 image_sizes
    vla, action_head = load_ensemble_member(cfg.vla_path, cfg.checkpoint_dir, cfg.lora_rank, device)
    resize_resolution = tuple(vla.config.image_sizes)

    # 构建确定性数据集（shuffle 关闭，image_aug 关闭）
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=(cfg.num_images_in_input > 1),
        use_proprio=cfg.use_proprio,
    )
    print(f"Building dataset: {cfg.dataset_name} (shuffle_files=False, image_aug=False)")
    dataset = RLDSDatasetWithTimestep(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=resize_resolution,
        shuffle_buffer_size=1,
        image_aug=False,
        train=True,
    )

    # 从 dataset_statistics.json 读取总帧数，用于精确截止
    import glob
    stats_files = glob.glob(
        str(cfg.data_root_dir / cfg.dataset_name / "**" / "dataset_statistics*.json"),
        recursive=True,
    )
    num_transitions = None
    num_episodes = None
    if stats_files:
        with open(stats_files[0]) as _f:
            _stats = json.load(_f)
        num_transitions = _stats.get("num_transitions")
        num_episodes = _stats.get("num_trajectories")
    if num_transitions is None:
        raise RuntimeError("无法从 dataset_statistics.json 读取 num_transitions，请手动指定 --num_transitions")
    print(f"Dataset: {num_transitions} frames across {num_episodes} episodes")

    # 推理（逐帧）
    all_preds = []
    episode_indices_list = []
    step_indices_list = []
    task_names_list = []
    episode_idx = 0
    prev_step = None

    print("Running inference...")
    pbar = tqdm(total=num_transitions, unit="frame", dynamic_ncols=True)
    for j, batch in enumerate(dataset):
        if j >= num_transitions:
            break

        current_step = batch.pop("_timestep")
        task_name = batch.pop("_task", "")

        if prev_step is not None and current_step == 0:
            episode_idx += 1
        prev_step = current_step
        episode_indices_list.append(episode_idx)
        step_indices_list.append(current_step)
        task_names_list.append(task_name)

        pred = run_inference_on_batch(vla, action_head, batch, cfg.unnorm_key, device)
        all_preds.append(pred)
        pbar.update(1)

        if j % 100 == 0:
            pbar.set_postfix(episode=episode_idx, step=current_step)
    pbar.close()

    n_frames = len(all_preds)
    print(f"Total frames: {n_frames} across {episode_idx + 1} episodes")

    # 保存预测结果
    preds_array = np.stack(all_preds, axis=0)  # (N_frames, NUM_ACTIONS_CHUNK, ACTION_DIM)
    np.save(cfg.output_path, preds_array)
    print(f"Predictions saved to: {cfg.output_path}")

    # 保存帧索引元数据（供 merge 模式使用）
    meta_path = cfg.output_path.replace(".npy", "_meta.npz")
    np.savez(
        meta_path,
        episode_indices=np.array(episode_indices_list),
        step_indices=np.array(step_indices_list),
        task_names=np.array(task_names_list, dtype=object),
    )
    print(f"Frame metadata saved to: {meta_path}")


def run_dual_infer(cfg: EnsembleUncertaintyConfig):
    """双模型单卡推理：两个模型同时加载，数据集只过一遍，直接输出 uncertainty_scores.npz。"""
    assert cfg.checkpoint_dir_0 and cfg.checkpoint_dir_1, \
        "--checkpoint_dir_0 和 --checkpoint_dir_1 不能为空"

    device = torch.device(f"cuda:{cfg.cuda_device}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    register_openvla_if_local(cfg.vla_path)
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)

    print("Loading model 0...")
    vla_0, ah_0 = load_ensemble_member(cfg.vla_path, cfg.checkpoint_dir_0, cfg.lora_rank, device)
    print("Loading model 1...")
    vla_1, ah_1 = load_ensemble_member(cfg.vla_path, cfg.checkpoint_dir_1, cfg.lora_rank, device)
    resize_resolution = tuple(vla_0.config.image_sizes)

    action_tokenizer = ActionTokenizer(processor.tokenizer)
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=(cfg.num_images_in_input > 1),
        use_proprio=cfg.use_proprio,
    )
    dataset = RLDSDatasetWithTimestep(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=resize_resolution,
        shuffle_buffer_size=1,
        image_aug=False,
        train=True,
    )

    import glob
    stats_files = glob.glob(
        str(cfg.data_root_dir / cfg.dataset_name / "**" / "dataset_statistics*.json"),
        recursive=True,
    )
    if not stats_files:
        raise RuntimeError("无法找到 dataset_statistics.json")
    with open(stats_files[0]) as f:
        stats = json.load(f)
    num_transitions = stats["num_transitions"]
    num_episodes = stats.get("num_trajectories")
    print(f"Dataset: {num_transitions} frames across {num_episodes} episodes")

    all_preds_0, all_preds_1 = [], []
    episode_indices_list, step_indices_list, task_names_list, gripper_states_list = [], [], [], []
    episode_idx = 0
    prev_step = None

    print("Running dual inference (one pass)...")
    pbar = tqdm(total=num_transitions, unit="frame", dynamic_ncols=True)
    for j, batch in enumerate(dataset):
        if j >= num_transitions:
            break

        current_step = batch.pop("_timestep")
        task_name = batch.pop("_task", "")

        if prev_step is not None and current_step == 0:
            episode_idx += 1
            if cfg.max_episodes > 0 and episode_idx >= cfg.max_episodes:
                break
        prev_step = current_step
        episode_indices_list.append(episode_idx)
        step_indices_list.append(current_step)
        task_names_list.append(task_name)

        # gripper state: proprio dim 6 (gripper width), or NaN if not available
        proprio = batch.get("proprio", None)
        if proprio is not None:
            p = proprio.squeeze()
            gripper_states_list.append(float(p[6]) if p.ndim >= 1 and p.shape[-1] > 6 else float('nan'))
        else:
            gripper_states_list.append(float('nan'))

        all_preds_0.append(run_inference_on_batch(vla_0, ah_0, batch, cfg.unnorm_key, device))
        all_preds_1.append(run_inference_on_batch(vla_1, ah_1, batch, cfg.unnorm_key, device))
        pbar.update(1)
        if j % 100 == 0:
            pbar.set_postfix(episode=episode_idx, step=current_step)
    pbar.close()

    n_frames = len(all_preds_0)
    print(f"Total frames: {n_frames} across {episode_idx + 1} episodes")

    preds_array = np.stack([np.stack(all_preds_0), np.stack(all_preds_1)], axis=0)
    variances = preds_array.var(axis=0).mean(axis=(1, 2))

    print(f"Variance stats: min={variances.min():.6f}  max={variances.max():.6f}  "
          f"mean={variances.mean():.6f}  p99={np.percentile(variances, 99):.6f}")

    np.savez(
        cfg.output_path,
        episode_indices=np.array(episode_indices_list),
        step_indices=np.array(step_indices_list),
        task_names=np.array(task_names_list, dtype=object),
        variances=variances,
        gripper_states=np.array(gripper_states_list, dtype=np.float32),
    )
    print(f"Saved: {cfg.output_path}")


def run_merge(cfg: EnsembleUncertaintyConfig):
    """汇总模式：加载所有成员的预测，计算每帧方差，保存 uncertainty_scores.npz。"""
    assert len(cfg.pred_files) >= 2, "--pred_files 需要至少 2 个文件"
    assert cfg.meta_file, "--meta_file 不能为空"

    print(f"Loading {len(cfg.pred_files)} prediction files...")
    all_preds = []
    for f in cfg.pred_files:
        preds = np.load(f)
        print(f"  {f}: shape {preds.shape}")
        all_preds.append(preds)

    # 验证所有成员帧数一致
    n_frames_list = [p.shape[0] for p in all_preds]
    assert len(set(n_frames_list)) == 1, f"各成员帧数不一致: {n_frames_list}"
    n_frames = n_frames_list[0]

    # 加载帧索引元数据
    meta = np.load(cfg.meta_file, allow_pickle=True)
    episode_indices = meta["episode_indices"]
    step_indices = meta["step_indices"]
    task_names = meta["task_names"] if "task_names" in meta else np.array([""] * n_frames, dtype=object)
    assert len(episode_indices) == n_frames, "元数据帧数与预测帧数不匹配"

    # 计算每帧方差
    print("Computing per-frame variance...")
    preds_array = np.stack(all_preds, axis=0)  # (N_models, N_frames, chunk, action_dim)
    variances = preds_array.var(axis=0).mean(axis=(1, 2))  # (N_frames,)

    print(f"\nVariance stats:")
    print(f"  min:    {variances.min():.6f}")
    print(f"  max:    {variances.max():.6f}")
    print(f"  mean:   {variances.mean():.6f}")
    print(f"  std:    {variances.std():.6f}")
    print(f"  p99:    {np.percentile(variances, 99):.6f}")
    print(f"  frames with var > mean: {(variances > variances.mean()).sum()} / {n_frames}")

    np.savez(
        cfg.output_path,
        episode_indices=episode_indices,
        step_indices=step_indices,
        task_names=task_names,
        variances=variances,
    )
    print(f"\nSaved to: {cfg.output_path}")


def run_parallel(cfg: EnsembleUncertaintyConfig):
    """双卡并行模式：lora_0 → GPU 0，lora_1 → GPU 1，推理完成后自动 merge。"""
    import subprocess
    import sys

    assert cfg.checkpoint_dir_0 and cfg.checkpoint_dir_1, \
        "--checkpoint_dir_0 和 --checkpoint_dir_1 不能为空"

    ckpt_dirs = [cfg.checkpoint_dir_0, cfg.checkpoint_dir_1]

    preds_dir = Path(cfg.preds_dir) if cfg.preds_dir else Path(cfg.output_path).parent / "preds_tmp"
    preds_dir.mkdir(parents=True, exist_ok=True)

    pred_paths: List[str] = []
    procs: List[subprocess.Popen] = []
    log_handles = []

    script_path = os.path.abspath(__file__)

    for k, ckpt_dir in enumerate(ckpt_dirs):
        out_npy = str(preds_dir / f"preds_lora{k}.npy")
        pred_paths.append(out_npy)

        cmd = [
            sys.executable, script_path,
            "--mode", "infer",
            "--vla_path", str(cfg.vla_path),
            "--data_root_dir", str(cfg.data_root_dir),
            "--dataset_name", cfg.dataset_name,
            "--checkpoint_dir", ckpt_dir,
            "--output_path", out_npy,
            "--unnorm_key", cfg.unnorm_key,
            "--lora_rank", str(cfg.lora_rank),
            "--use_proprio", str(cfg.use_proprio),
            "--num_images_in_input", str(cfg.num_images_in_input),
            "--cuda_device", str(k),
        ]
        log_path = str(preds_dir / f"infer_lora{k}.log")
        lf = open(log_path, "w")
        log_handles.append(lf)
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
        procs.append(proc)
        print(f"[parallel] lora_{k} started on GPU {k}  →  log: {log_path}")

    # 等待所有子进程完成
    failed = False
    for k, proc in enumerate(procs):
        ret = proc.wait()
        log_handles[k].close()
        if ret != 0:
            print(f"[parallel] ERROR: lora_{k} inference failed (exit {ret}). "
                  f"Check {preds_dir}/infer_lora{k}.log")
            failed = True
        else:
            print(f"[parallel] lora_{k} inference done")

    if failed:
        raise RuntimeError("One or more inference jobs failed — see logs above")

    # 自动 merge
    meta_file = pred_paths[0].replace(".npy", "_meta.npz")
    print(f"\n[parallel] Merging {len(pred_paths)} prediction files...")
    merge_cfg = EnsembleUncertaintyConfig(
        mode="merge",
        pred_files=pred_paths,
        meta_file=meta_file,
        output_path=cfg.output_path,
    )
    run_merge(merge_cfg)


@draccus.wrap()
def main(cfg: EnsembleUncertaintyConfig):
    if cfg.mode == "infer":
        run_infer(cfg)
    elif cfg.mode == "merge":
        run_merge(cfg)
    elif cfg.mode == "parallel":
        run_parallel(cfg)
    elif cfg.mode == "dual_infer":
        run_dual_infer(cfg)
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}. Use 'infer', 'merge', 'parallel', or 'dual_infer'.")


if __name__ == "__main__":
    main()
