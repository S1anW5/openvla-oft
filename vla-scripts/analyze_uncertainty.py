"""
analyze_uncertainty.py

读取 compute_ensemble_uncertainty.py 生成的方差文件，
画直方图验证 ensemble 多样性，归一化到 [0, 1] 并保存权重文件。

用法：
    python vla-scripts/analyze_uncertainty.py \
        --input_path uncertainty_scores.npz \
        --output_path uncertainty_weights.npz \
        --histogram_path uncertainty_histogram.png
"""

from dataclasses import dataclass
from pathlib import Path

import draccus
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class AnalyzeConfig:
    # fmt: off
    input_path: str = "uncertainty_scores.npz"    # compute_ensemble_uncertainty.py 的输出
    output_path: str = "uncertainty_weights.npz"  # 归一化后的权重文件
    histogram_path: str = "uncertainty_histogram.png"  # 直方图保存路径
    clip_percentile: float = 99.0                 # 截断异常值的百分位数
    # fmt: on


def analyze_uncertainty(cfg: AnalyzeConfig):
    data = np.load(cfg.input_path)
    variances = data["variances"]
    episode_indices = data["episode_indices"]
    step_indices = data["step_indices"]

    n_frames = len(variances)
    n_episodes = int(episode_indices.max()) + 1

    print(f"Dataset summary:")
    print(f"  Total frames:   {n_frames}")
    print(f"  Total episodes: {n_episodes}")
    print(f"  Avg frames/ep:  {n_frames / n_episodes:.1f}")

    print(f"\nVariance stats (raw):")
    print(f"  min:    {variances.min():.6f}")
    print(f"  max:    {variances.max():.6f}")
    print(f"  mean:   {variances.mean():.6f}")
    print(f"  median: {np.median(variances):.6f}")
    print(f"  std:    {variances.std():.6f}")
    print(f"  p99:    {np.percentile(variances, 99):.6f}")

    # 多样性检验
    if variances.mean() < 1e-4:
        print("\n[WARNING] Mean variance < 1e-4: ensemble 可能已坍缩！")
        print("  建议：增大 lora_dropout（0.0 -> 0.05）或使用差异更大的种子。")
    else:
        print("\n[OK] Ensemble 多样性检验通过。")

    # 归一化：截断 top clip_percentile% 异常值，再缩放到 [0, 1]
    p_clip = np.percentile(variances, cfg.clip_percentile)
    weights = np.clip(variances, 0.0, p_clip) / (p_clip + 1e-8)

    print(f"\nNormalized weight stats (after {cfg.clip_percentile}th percentile clip):")
    print(f"  min:    {weights.min():.4f}")
    print(f"  max:    {weights.max():.4f}")
    print(f"  mean:   {weights.mean():.4f}")
    print(f"  frames with weight > 0.5: {(weights > 0.5).sum()} / {n_frames} ({100*(weights>0.5).mean():.1f}%)")
    print(f"  frames with weight > 0.8: {(weights > 0.8).sum()} / {n_frames} ({100*(weights>0.8).mean():.1f}%)")

    # 画直方图
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(variances, bins=100, log=True, color="steelblue", edgecolor="none")
    axes[0].axvline(variances.mean(), color="red", linestyle="--", label=f"mean={variances.mean():.4f}")
    axes[0].axvline(p_clip, color="orange", linestyle="--", label=f"p{cfg.clip_percentile:.0f}={p_clip:.4f}")
    axes[0].set_xlabel("Variance")
    axes[0].set_ylabel("Count (log scale)")
    axes[0].set_title("Raw Variance Distribution")
    axes[0].legend()

    axes[1].hist(weights, bins=100, color="steelblue", edgecolor="none")
    axes[1].axvline(weights.mean(), color="red", linestyle="--", label=f"mean={weights.mean():.3f}")
    axes[1].set_xlabel("Normalized Weight")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Normalized Weight Distribution")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(cfg.histogram_path, dpi=150)
    print(f"\nHistogram saved to: {cfg.histogram_path}")

    # 保存归一化权重
    np.savez(
        cfg.output_path,
        episode_indices=episode_indices,
        step_indices=step_indices,
        weights=weights,
    )
    print(f"Normalized weights saved to: {cfg.output_path}")


@draccus.wrap()
def main(cfg: AnalyzeConfig):
    analyze_uncertainty(cfg)


if __name__ == "__main__":
    main()
