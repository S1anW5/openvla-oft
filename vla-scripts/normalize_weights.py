#!/usr/bin/env python3
"""
Per-episode normalize uncertainty weights.
w_t = variance_t / mean(variance within episode)
=> mean(w) = 1 within every episode.
"""
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", default="uncertainty_scores.npz")
parser.add_argument("--output_path", default="uncertainty_weights.npz")
args = parser.parse_args()

d = np.load(args.input_path)
variances   = d["variances"].astype(np.float64)
episode_idx = d["episode_indices"]
step_idx    = d["step_indices"]

weights = np.ones(len(variances), dtype=np.float32)
for ep in np.unique(episode_idx):
    mask   = episode_idx == ep
    mean_v = variances[mask].mean()
    if mean_v > 1e-10:
        weights[mask] = (variances[mask] / mean_v).astype(np.float32)

print(f"weights: min={weights.min():.4f}  mean={weights.mean():.4f}  "
      f"max={weights.max():.4f}  std={weights.std():.4f}")

ok = all(abs(weights[episode_idx == ep].mean() - 1.0) < 1e-4
         for ep in np.unique(episode_idx))
print(f"Per-episode mean == 1: {ok}")

np.savez(args.output_path, weights=weights,
         episode_indices=episode_idx, step_indices=step_idx)
print(f"Saved {args.output_path}")
