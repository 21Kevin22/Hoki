"""C7 step 1: for every init state of the chosen tasks, settle 10 steps exactly like run_closed_loop.py, then save what
Qwen sees (agentview RGB, upright) and what the 3D mapping needs (metric depth in OpenCV orientation, intrinsics,
camera-to-world) -> main/qwen/t0/<task>_<split><idx>.png / .npz.  Usage: python main/capture_t0.py"""
import os, sys
import numpy as np
import torch
_tl = torch.load
torch.load = lambda *a, **k: _tl(*a, **{**k, "weights_only": False})
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/ubuntu/slocal1/third_party/stowsafe_kit_v5/stowsafe")
from PIL import Image
from libero.libero.envs import OffScreenRenderEnv
from stowsafe.libero_env import camera_views
from tasks import TASKS, bddl_path

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHOSEN = ("T1", "T2", "T3", "T5", "T6")
out = os.path.join(HERE, "main", "qwen", "t0"); os.makedirs(out, exist_ok=True)
for task in CHOSEN:
    bddl, _ = bddl_path(*TASKS[task][:2])
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256, camera_depths=True,
                             camera_names=["agentview", "robot0_eye_in_hand"])
    env.seed(7)
    S = np.load(os.path.join(HERE, "main", "init_states", f"{task}.npz"))
    for split in ("dev", "eval"):
        for i, st in enumerate(S[split]):
            env.reset(); obs = env.set_init_state(st)
            for _ in range(10):
                obs, *_ = env.step([0.0] * 6 + [-1.0])
            stem = os.path.join(out, f"{task}_{split}{i:02d}")
            # true pinhole image (raw[::-1]; the policy's raw[::-1, ::-1] is mirrored left-right): same pixels as depth
            Image.fromarray(np.ascontiguousarray(obs["agentview_image"][::-1])).save(stem + ".png")
            v = camera_views(env, obs)[0]                                       # agentview, OpenCV orientation
            np.savez_compressed(stem + ".npz", depth=v.depth_m, K=v.intrinsics, T=v.camera_to_world)
    print(task, "captured", len(S["dev"]) + len(S["eval"]), flush=True)
    env.close()
