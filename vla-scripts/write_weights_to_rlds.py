#!/usr/bin/env python3
"""
write_weights_to_rlds.py

Reads libero_goal_no_noops in canonical order (shuffle_files=False), adds per-frame
uncertainty weights as 'steps/frame_weight' to each step, and writes a new dataset
libero_goal_no_noops_weighted/ with the same 16-shard structure.

Includes an ordering verification step: compares TFDS episode order vs raw TFRecord
order by matching episode lengths and first actions. If they don't match, abort.

Usage:
  python vla-scripts/write_weights_to_rlds.py \
    [--data_root /hdd/slwu/test_5_2/openvla-oft/modified_libero_rlds] \
    [--weights_file /hdd/slwu/test_5_2/openvla-oft/uncertainty_weights.npz]
"""

import argparse
import glob
import json
import os
import shutil

import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")
import tensorflow_datasets as tfds


SRC_DATASET = "libero_goal_no_noops"
DST_DATASET = "libero_goal_no_noops_weighted"
VERSION = "1.0.0"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        default="/hdd/slwu/test_5_2/openvla-oft/modified_libero_rlds",
    )
    parser.add_argument(
        "--weights_file",
        default="/hdd/slwu/test_5_2/openvla-oft/uncertainty_weights.npz",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Step 1: build fingerprint → weight array from TFDS canonical order
# ---------------------------------------------------------------------------
# TFDS shuffle_files=False interleaves shards in a strided pattern, NOT in
# simple sorted-shard order.  We therefore do NOT rely on raw-file reading
# order matching the TFDS order.  Instead we build a lookup table keyed by
# each episode's first action (float32 bytes, 28 bytes → effectively unique
# across 428 episodes).  The raw TFRecord write step then matches episodes by
# fingerprint regardless of shard/record position.

def build_fingerprint_weights(data_root, dataset_name, weights):
    print("[1/4] Building fingerprint→weights table from TFDS canonical order ...")
    builder = tfds.builder(dataset_name, data_dir=data_root)
    ds = builder.as_dataset(split="train", shuffle_files=False)

    fp_to_weights = {}
    frame_offset = 0
    for i, ep in enumerate(ds):
        for steps in ep["steps"].batch(100000):
            T = int(steps["action"].shape[0])
            first_act = steps["action"][0].numpy()  # shape (7,)
        # (T, first_action_bytes) is unique across all 428 LIBERO-Goal episodes
        fp = (T, first_act.tobytes())
        assert fp not in fp_to_weights, f"Fingerprint collision at ep {i}! Use more data."
        fp_to_weights[fp] = weights[frame_offset : frame_offset + T].copy()
        frame_offset += T
        if i % 100 == 0:
            print(f"    ep {i}: T={T}", flush=True)

    print(f"    Built table for {len(fp_to_weights)} episodes, {frame_offset} frames.")
    assert frame_offset == len(weights), (
        f"Frame count mismatch: TFDS={frame_offset}, weights={len(weights)}"
    )
    return fp_to_weights


# ---------------------------------------------------------------------------
# Step 2: write weighted TFRecords (match by fingerprint, shard order free)
# ---------------------------------------------------------------------------

def add_frame_weight(raw_bytes: bytes, ep_weights: np.ndarray) -> bytes:
    ex = tf.train.Example()
    ex.ParseFromString(raw_bytes)
    T_check = len(ex.features.feature["steps/is_first"].int64_list.value)
    assert len(ep_weights) == T_check, (
        f"Weight array length {len(ep_weights)} != episode length {T_check}"
    )
    ex.features.feature["steps/frame_weight"].float_list.value.extend(
        ep_weights.tolist()
    )
    return ex.SerializeToString()


def write_weighted_shards(src_dir, dst_dir, fp_to_weights):
    os.makedirs(dst_dir, exist_ok=True)
    shard_files = sorted(glob.glob(os.path.join(src_dir, "*.tfrecord-*-of-*")))
    print(f"[2/4] Writing {len(shard_files)} weighted shards to:\n    {dst_dir}")
    total_eps, total_frames = 0, 0
    for shard_path in shard_files:
        # Rename: original 'libero_goal-train.tfrecord-*' → '{DST_DATASET}-train.tfrecord-*'
        shard_name = os.path.basename(shard_path)
        src_prefix = shard_name.rsplit("-train.", 1)[0]
        dst_shard_name = shard_name.replace(src_prefix, DST_DATASET, 1)
        dst_path = os.path.join(dst_dir, dst_shard_name)
        print(f"    {shard_name} ...", flush=True)
        with tf.io.TFRecordWriter(dst_path) as writer:
            for raw in tf.data.TFRecordDataset(shard_path):
                b = raw.numpy()
                ex_peek = tf.train.Example()
                ex_peek.ParseFromString(b)
                T_raw = len(ex_peek.features.feature["steps/is_first"].int64_list.value)
                first_act = np.array(
                    ex_peek.features.feature["steps/action"].float_list.value[:7],
                    dtype=np.float32,
                )
                fp = (T_raw, first_act.tobytes())
                ep_weights = fp_to_weights[fp]
                writer.write(add_frame_weight(b, ep_weights))
                total_eps += 1
                total_frames += len(ep_weights)
    print(f"    Done: {total_eps} episodes, {total_frames} frames written.")


# ---------------------------------------------------------------------------
# Step 5: copy and update metadata files
# ---------------------------------------------------------------------------

def update_features_json(src_dir, dst_dir):
    with open(os.path.join(src_dir, "features.json")) as f:
        feat = json.load(f)
    steps_features = (
        feat["featuresDict"]["features"]["steps"]
        ["sequence"]["feature"]["featuresDict"]["features"]
    )
    steps_features["frame_weight"] = {
        "pythonClassName": "tensorflow_datasets.core.features.scalar.Scalar",
        "tensor": {
            "shape": {},
            "dtype": "float32",
            "encoding": "none",
        },
        "description": "Per-frame epistemic uncertainty weight (normalized to [0,1]).",
    }
    with open(os.path.join(dst_dir, "features.json"), "w") as f:
        json.dump(feat, f, indent=4)
    print("    features.json updated with 'frame_weight'.")


def copy_metadata(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    print("[3/4] Copying metadata files ...")
    for fname in os.listdir(src_dir):
        if not fname.endswith(".json"):
            continue
        if fname == "features.json":
            update_features_json(src_dir, dst_dir)
            continue
        src_path = os.path.join(src_dir, fname)
        dst_path = os.path.join(dst_dir, fname)
        if fname == "dataset_info.json":
            with open(src_path) as f:
                info = json.load(f)
            info["name"] = DST_DATASET
            info["moduleName"] = ""  # ReadOnlyBuilder doesn't need a module
            with open(dst_path, "w") as f:
                json.dump(info, f, indent=2)
            print(f"    dataset_info.json updated (name → {DST_DATASET}).")
        else:
            shutil.copy2(src_path, dst_path)
            print(f"    {fname} copied.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    src_dir = os.path.join(args.data_root, SRC_DATASET, VERSION)
    dst_dir = os.path.join(args.data_root, DST_DATASET, VERSION)

    if os.path.exists(dst_dir):
        print(f"WARNING: destination {dst_dir} already exists. Removing...")
        shutil.rmtree(dst_dir)

    print(f"Loading weights from {args.weights_file} ...")
    data = np.load(args.weights_file)
    weights = data["weights"].astype(np.float32)
    print(f"  weights shape={weights.shape}, range=[{weights.min():.4f}, {weights.max():.4f}]")

    # Build fingerprint table from TFDS canonical order
    fp_to_weights = build_fingerprint_weights(args.data_root, SRC_DATASET, weights)

    # Write weighted shards (order-independent: matched by fingerprint)
    copy_metadata(src_dir, dst_dir)  # copy metadata first so dst_dir exists
    write_weighted_shards(src_dir, dst_dir, fp_to_weights)

    print("\n[4/4] Verifying new dataset loads correctly ...")
    builder = tfds.builder(DST_DATASET, data_dir=args.data_root)
    ds = builder.as_dataset(split="train", shuffle_files=False)
    # Quick spot-check: first 5 steps of first episode should have frame_weight
    for ep in ds.take(1):
        for step in ep["steps"].take(5):
            w = float(step["frame_weight"].numpy())
            t = bool(step["is_first"].numpy())
            print(f"    is_first={t}, frame_weight={w:.4f}")
    print("    ✓ frame_weight present in new dataset.")

    print("\n✓ All done!")
    print(f"  New dataset: {dst_dir}")
    print("  Run train_weighted.sh — WeightedRLDSDataset reads frame_weight from each batch.")


if __name__ == "__main__":
    main()
