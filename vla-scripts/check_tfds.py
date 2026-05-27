import argparse, sys, tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import tensorflow_datasets as tfds
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", required=True)
parser.add_argument("--dataset", default="libero_goal_no_noops")
args = parser.parse_args()

builder = tfds.builder(args.dataset, data_dir=args.data_root)

def read_all(shuffle):
    ds = builder.as_dataset(split='train', shuffle_files=shuffle)
    fps, Ts, first_actions = [], [], []
    for i, ep in enumerate(ds):
        fp = ep['episode_metadata']['file_path'].numpy().decode()
        for steps in ep['steps'].batch(10000):
            T = int(steps['action'].shape[0])
            first_act = steps['action'][0].numpy()
            break
        fps.append(fp)
        Ts.append(T)
        first_actions.append(first_act)
        if i % 50 == 0:
            print(f"  ep {i}: T={T}, fp={fp.split('/')[-1]}", flush=True)
    return fps, Ts, first_actions

print("=== 第一次读取 (shuffle=False) ===", flush=True)
fps1, Ts1, acts1 = read_all(shuffle=False)
print(f"Total episodes: {len(fps1)}, total frames: {sum(Ts1)}", flush=True)
print(f"Unique file_paths: {len(set(fps1))}", flush=True)

print("\n=== 第二次读取 (shuffle=False) ===", flush=True)
fps2, Ts2, acts2 = read_all(shuffle=False)

print("\n=== 两次读取一致性验证 ===", flush=True)
fp_match  = all(a == b for a, b in zip(fps1, fps2))
T_match   = all(a == b for a, b in zip(Ts1,  Ts2))
act_match = all(np.allclose(a, b) for a, b in zip(acts1, acts2))
print(f"file_path 顺序一致: {fp_match}", flush=True)
print(f"episode 长度一致:   {T_match}", flush=True)
print(f"第一帧 action 一致: {act_match}", flush=True)
print(f"\n前5个 file_path: {[f.split('/')[-1] for f in fps1[:5]]}", flush=True)
print(f"前5个长度: {Ts1[:5]}", flush=True)
