"""PREREG (14) amendment: frozen Cosmos Policy LIBERO (nvidia/Cosmos-Policy-LIBERO-Predict2-2B) as an action-conditioned
world model. Runs in third_party/cosmos-policy/.venv. For each replayed decision point:
  1. get_action() on the real current observation (256-px agentview + wrist, upright = LIBERO flipud, proprio 9-d);
  2. the action latent of that sample is overwritten with OUR executed 16-step chunk (normalized with the checkpoint's
     dataset stats, inverse of unnormalize_actions) via replace_latent_with_action_chunk;
  3. get_future_state_prediction() -> predicted agentview image at t+16 (model's own decoder);
  4. the model's encode/decode roundtrip of the current image and of the TRUE t+16 image (a second get_action call with
     the true future as the current image) -> 'cur' and 'fut' in the same degraded domain.
Writes main/data14/cosmos/<run>.npz with pred / fut / cur (uint8, model resolution) and t.
Usage: CUDA_VISIBLE_DEVICES=0 .venv/bin/python cosmos_predict14.py [--limit N] [--every K]"""
import argparse, glob, os, sys, json
import numpy as np
import torch

CP = "/home/ubuntu/slocal1/third_party/cosmos-policy"
sys.path.insert(0, CP); os.chdir(CP)
from cosmos_policy.experiments.robot.libero.run_libero_eval import PolicyEvalConfig
from cosmos_policy.experiments.robot.cosmos_utils import (get_action, get_model, load_dataset_stats,
                                                         init_t5_text_embeddings_cache, get_t5_embedding_from_cache,
                                                         get_future_state_prediction)
from cosmos_policy.models.policy_text2world_model import replace_latent_with_action_chunk

HERE = "/home/ubuntu/slocal1/Hoki/occ_vla/next_hidden"
DATA = os.path.join(HERE, "main", "data14"); OUT = os.path.join(DATA, "cosmos"); os.makedirs(OUT, exist_ok=True)
TASK = "put the yellow and white mug in the microwave and close it"
ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--every", type=int, default=1)
a = ap.parse_args()

cfg = PolicyEvalConfig(
    config="cosmos_predict2_2b_480p_libero__inference_only", ckpt_path="nvidia/Cosmos-Policy-LIBERO-Predict2-2B",
    config_file="cosmos_policy/config/config.py",
    dataset_stats_path="nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_dataset_statistics.json",
    t5_text_embeddings_path="nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_t5_embeddings.pkl",
    use_wrist_image=True, use_proprio=True, normalize_proprio=True, unnormalize_actions=True, chunk_size=16,
    num_open_loop_steps=16, trained_with_image_aug=True, use_jpeg_compression=True, flip_images=True,
    num_denoising_steps_action=5, num_denoising_steps_future_state=1, num_denoising_steps_value=1)
stats = load_dataset_stats(cfg.dataset_stats_path)
init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
model, cosmos_config = get_model(cfg)
emb = get_t5_embedding_from_cache(TASK)
amin, amax = np.asarray(stats["actions_min"]), np.asarray(stats["actions_max"])


def norm_actions(A):
    return 2.0 * (A - amin) / (amax - amin) - 1.0


def run_one(img, wrist, prop, actions=None):
    obs = {"primary_image": img, "wrist_image": wrist, "proprio": prop}
    r = get_action(cfg, model, stats, obs, emb, seed=0, num_denoising_steps_action=cfg.num_denoising_steps_action,
                   generate_future_state_and_value_in_parallel=False)
    li = r["latent_indices"]
    # decode the clean (VAE-encoded) input frames -> model-domain current image
    dec = ((model.decode(r["orig_clean_latent_frames"].clone()) + 1.0) * 127.5).clamp(0, 255)
    dec = dec.permute(0, 2, 3, 4, 1).to(torch.uint8).cpu().numpy()[0]           # (T, H, W, C)
    out = {"cur": dec[(3 - 1) * 4 + 1]}                                          # latent 3 = current primary image
    if actions is not None:
        lat = r["generated_latent"].clone()
        idx = torch.full((1,), li["action_latent_idx"], dtype=torch.int64, device=lat.device)
        act = torch.from_numpy(norm_actions(actions)).to(lat.device, lat.dtype)[None]
        lat = replace_latent_with_action_chunk(lat, act, idx)
        f = get_future_state_prediction(cfg, model=model, data_batch=r["data_batch"], generated_latent_with_action=lat,
                                        orig_clean_latent_frames=r["orig_clean_latent_frames"],
                                        future_proprio_latent_idx=li["future_proprio_latent_idx"],
                                        future_wrist_image_latent_idx=li["future_wrist_image_latent_idx"],
                                        future_wrist_image2_latent_idx=li["future_wrist_image2_latent_idx"],
                                        future_image_latent_idx=li["future_image_latent_idx"],
                                        future_image2_latent_idx=li["future_image2_latent_idx"],
                                        num_denoising_steps_future_state=cfg.num_denoising_steps_future_state)
        out["pred"] = np.asarray(f["future_image_predictions"]["future_image"])
    return out


files = sorted(glob.glob(os.path.join(DATA, "*.npz")))
files = files[: a.limit] if a.limit else files
for fpath in files:
    out_path = os.path.join(OUT, os.path.basename(fpath))
    if os.path.exists(out_path):
        continue
    z = np.load(fpath)
    keep = list(range(0, len(z["t"]), a.every))
    P, F, C = [], [], []
    for j in keep:
        o = run_one(z["agent256"][j], z["wrist256"][j], z["proprio"][j], z["actions"][j])
        tr = run_one(z["agent256_future"][j], z["wrist256"][j], z["proprio"][j])   # roundtrip of the true future
        P.append(o["pred"]); C.append(o["cur"]); F.append(tr["cur"])
    np.savez_compressed(out_path, pred=np.array(P), fut=np.array(F), cur=np.array(C), t=z["t"][keep])
    print("done", os.path.basename(fpath), len(keep), flush=True)
