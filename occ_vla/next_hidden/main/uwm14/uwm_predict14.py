"""PREREG (14): frozen UWM (libero_90.pt) forward dynamics on the replayed H3 decision points.
Runs in third_party/unified-world-model/.venv_uwm. For every decision point writes, in the model's 224x224 image
domain (resize 240x240 + center crop 224, eval mode): VAE(current agentview), VAE(true future agentview t+16) and the
UWM-predicted future agentview (first of the 2 predicted frames), as uint8. Images go in as the LIBERO hdf5 convention
the model was trained on (robosuite frames with rows flipped = our upright mujoco.Renderer frames).
Usage: CUDA_VISIBLE_DEVICES=0 .venv_uwm/bin/python uwm_predict14.py [--limit N] [--device cuda]"""
import argparse, glob, os, sys
import numpy as np
import torch
from einops import rearrange

UWM = "/home/ubuntu/slocal1/third_party/unified-world-model"
sys.path.insert(0, UWM)
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(HERE, "main", "data14")
OUT = os.path.join(DATA, "uwm"); os.makedirs(OUT, exist_ok=True)

ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--device", default="cuda")
ap.add_argument("--bs", type=int, default=16); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
torch.manual_seed(a.seed)
with initialize_config_dir(config_dir=os.path.join(UWM, "configs"), version_base=None):
    cfg = compose(config_name="train_uwm_robomimic", overrides=["dataset=libero_90"])
model = instantiate(cfg.model)
ck = torch.load(os.path.join(UWM, "checkpoints", "libero_90.pt"), map_location="cpu", weights_only=False)
print(model.load_state_dict(ck["model"]))
norm = ck["action_normalizer"]
model = model.to(a.device).eval()
enc = model.obs_encoder
keys = sorted(enc.rgb_keys)                      # ['agentview_rgb', 'eye_in_hand_rgb']
print("rgb keys", keys, "views", enc.num_views)


def to8(x):                                      # (B, C, H, W) in [0,1] -> uint8 HWC
    return (x.clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()


files = sorted(glob.glob(os.path.join(DATA, "*.npz")))
if a.limit:
    files = files[: a.limit]
for f in files:
    out = os.path.join(OUT, os.path.basename(f))
    if os.path.exists(out):
        continue
    z = np.load(f)
    n = len(z["t"]); cur, fut, pred = [], [], []
    for s in range(0, n, a.bs):
        sl = slice(s, min(n, s + a.bs))
        obs = {"agentview_rgb": torch.from_numpy(np.stack([z["agent_prev"][sl], z["agent"][sl]], 1)),
               "eye_in_hand_rgb": torch.from_numpy(np.stack([z["wrist_prev"][sl], z["wrist"][sl]], 1))}
        nxt = {"agentview_rgb": torch.from_numpy(np.stack([z["agent_future"][sl], z["agent_future2"][sl]], 1)),
               "eye_in_hand_rgb": torch.from_numpy(np.stack([z["wrist"][sl], z["wrist"][sl]], 1))}   # wrist future unused
        obs = {k: v.to(a.device) for k, v in obs.items()}; nxt = {k: v.to(a.device) for k, v in nxt.items()}
        act = torch.from_numpy(norm(z["actions"][sl].astype(np.float32))).float().to(a.device)
        with torch.no_grad():
            p = model.sample_forward_dynamics(obs, act)                       # (B, V, c, T, h, w) latents
            p = enc.apply_vae(p, inverse=True)                                  # (B, V, 3, T, 224, 224)
            tf = enc.apply_vae(enc.encode_next_obs(nxt), inverse=True)
            cu = enc.apply_vae(enc.apply_vae(enc.apply_transform(obs)), inverse=True)
        v = keys.index("agentview_rgb")
        pred.append(to8(p[:, v, :, 0])); fut.append(to8(tf[:, v, :, 0])); cur.append(to8(cu[:, v, :, 1]))
    np.savez_compressed(out, pred=np.concatenate(pred), fut=np.concatenate(fut), cur=np.concatenate(cur), t=z["t"])
    print("done", os.path.basename(f), n, flush=True)
