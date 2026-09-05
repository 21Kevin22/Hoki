"""Standalone, faithful (as-scoped) V-JEPA2-AC CEM+MPC control loop on a
real LIBERO(-Occ) task, per arXiv:2506.09985's actual method: sample
candidate action sequences, roll them forward through the frozen
encoder + action-conditioned predictor's LATENT dynamics, score by L1
distance to a goal image's own encoded latent, refine the sampling
distribution toward the best candidates (Cross-Entropy Method), execute
the first action, replan (receding horizon).

OpenVLA-OFT is NOT used at all in this script -- this bypasses it
entirely, unlike every other condition in run_libero_occluded_oracle_
headroom.py. This is Option "standalone V-JEPA2-AC control loop" from
the 2026-08-31 scoping discussion.

Deliberate, disclosed approximations (real V-JEPA2-AC's own published
setup uses a real Franka/Droid robot + hand-specified per-phase horizon
+ a scale calibrated to their action-magnitude convention; none of
that is re-derivable from the public checkpoint alone):
  - Action/state 7-dim convention: LIBERO's own OSC_POSE normalized
    delta (dx,dy,dz,drx,dry,drz,gripper) -- confirmed to MATCH the
    checkpoint's action_embed_dim=7 in dimensionality (see CLAUDE.md),
    NOT confirmed to match its units/scale.
  - State vector: eef_pos[3] + axis-angle[3] + gripper_qpos[0] (7-dim,
    dropping LIBERO's 2nd gripper-finger dim to fit 7) -- an
    approximation of the checkpoint's own (undocumented) proprio format.
  - Same current state repeated across the whole planning horizon
    (no per-future-step state estimate) -- a known simplification.
  - Latent rollout: single real 2-frame context clip encoded once per
    replan step; each candidate's future latent is produced by calling
    the predictor ONCE PER HORIZON STEP, feeding back its own predicted
    latent as the next step's context (recursive latent rollout) --
    this is the natural reading of the predictor's per-call API
    (context tokens + one action/state token in, one predicted latent
    out), not copied from any released rollout utility in the repo.

Not implemented: task-varying horizon, the paper's L1-ball action
clipping (a generic magnitude clip is used instead), goal switching
across sub-phases. This is a smoke-test-scale implementation meant to
answer "does the mechanism run and produce a sane, non-degenerate
control signal" -- not a claim of matching the paper's own reported
zero-shot success rates.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
VJEPA2_DIR = REPO_ROOT / "thirdparty" / "vjepa2"
sys.path.insert(0, str(VJEPA2_DIR))

SCRIPTS_DIR = str(Path(__file__).resolve().parent)
sys.path.insert(0, SCRIPTS_DIR)
OFT_ROOT = str(REPO_ROOT / "thirdparty" / "openvla-oft")
sys.path.insert(0, OFT_ROOT)
import os  # noqa: E402
os.chdir(OFT_ROOT)
os.environ.setdefault("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero_oft"))

import register_libero_occ_suites  # noqa: E402
from libero.libero import benchmark  # noqa: E402
from run_libero_occluded_oracle_headroom import get_libero_env_seg  # noqa: E402

OSC_POSE_MAX_DELTA_M = 0.05
ACTION_CLIP = 1.0  # generic magnitude clip (stand-in for the paper's L1-ball radius)


def quat2axisangle(quat):
    # same convention as openvla-oft's own proprio construction
    quat = quat / (np.linalg.norm(quat) + 1e-8)
    w, x, y, z = quat[3], quat[0], quat[1], quat[2]
    angle = 2 * np.arccos(np.clip(w, -1.0, 1.0))
    s = np.sqrt(max(1e-8, 1 - w * w))
    axis = np.array([x, y, z]) / s if s > 1e-6 else np.array([1.0, 0.0, 0.0])
    return axis * angle


def load_vjepa2(device):
    from src.hub.backbones import vjepa2_ac_vit_giant
    encoder, predictor = vjepa2_ac_vit_giant(pretrained=True)
    encoder = encoder.to(device).eval()
    predictor = predictor.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    for p in predictor.parameters():
        p.requires_grad_(False)
    return encoder, predictor


def rgb_to_clip(arr_uint8, device, size=256):
    img = Image.fromarray(arr_uint8).resize((size, size))
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    frame = torch.from_numpy(arr).permute(2, 0, 1)
    return frame.to(device)


def two_frame_clip(frame_prev, frame_cur):
    # (C,H,W) x2 -> (1,C,2,H,W)
    clip = torch.stack([frame_prev, frame_cur], dim=1).unsqueeze(0)
    return clip


@torch.no_grad()
def rollout_latent(predictor, z_context, action_seq, state_vec, device):
    """z_context: (256,1408) current context tokens.
    action_seq: (H,7) numpy. state_vec: (7,) numpy, repeated each step.
    Returns z_final: (256,1408) predicted latent after H steps."""
    z = z_context.unsqueeze(0)  # (1,256,1408)
    for h in range(action_seq.shape[0]):
        a = torch.from_numpy(action_seq[h]).float().to(device).view(1, 1, 7)
        s = torch.from_numpy(state_vec).float().to(device).view(1, 1, 7)
        z = predictor(z, a, s)  # (1,256,1408)
    return z[0]


def cem_plan(predictor, z_context, state_vec, z_goal, device,
             horizon, n_candidates, n_iters, elite_frac, init_std):
    mean = np.zeros((horizon, 7), dtype=np.float32)
    std = np.full((horizon, 7), init_std, dtype=np.float32)
    n_elite = max(1, int(n_candidates * elite_frac))
    best_action_seq = mean.copy()
    for it in range(n_iters):
        candidates = mean[None] + std[None] * np.random.randn(n_candidates, horizon, 7).astype(np.float32)
        candidates[..., :6] = np.clip(candidates[..., :6], -ACTION_CLIP, ACTION_CLIP)
        candidates[..., 6] = np.sign(candidates[..., 6])  # gripper: discrete open/close
        energies = np.zeros(n_candidates, dtype=np.float32)
        for i in range(n_candidates):
            z_pred = rollout_latent(predictor, z_context, candidates[i], state_vec, device)
            energies[i] = torch.mean(torch.abs(z_pred - z_goal)).item()
        elite_idx = np.argsort(energies)[:n_elite]
        elite = candidates[elite_idx]
        mean = elite.mean(axis=0)
        std = elite.std(axis=0) + 1e-3
        best_action_seq = mean
    return best_action_seq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", type=int, default=1)
    ap.add_argument("--n-episodes", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--replan-every", type=int, default=1)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--n-candidates", type=int, default=8)
    ap.add_argument("--n-cem-iters", type=int, default=2)
    ap.add_argument("--elite-frac", type=float, default=0.25)
    ap.add_argument("--init-std", type=float, default=0.5)
    ap.add_argument("--goal-image", type=str, default="/tmp/vjepa2_domain_check_extra/task1_goal_success.png")
    ap.add_argument("--results-dir", type=str, default="cem_control_smoke")
    args = ap.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[setup] device={device}")

    encoder, predictor = load_vjepa2(device)
    print("[setup] real V-JEPA2-AC loaded (encoder+predictor, frozen)")

    goal_arr = np.asarray(Image.open(args.goal_image).convert("RGB"))
    goal_frame = rgb_to_clip(goal_arr, device)
    with torch.no_grad():
        z_goal = encoder(two_frame_clip(goal_frame, goal_frame))[0]  # (256,1408)
    print(f"[setup] goal latent encoded from {args.goal_image}")

    occluded_suite = benchmark.get_benchmark_dict()["libero_10_occluded"]()
    task = occluded_suite.get_task(args.task_id)
    env = get_libero_env_seg(task, resolution=256)
    env.seed(0)
    env.reset()
    init_states = occluded_suite.get_task_init_states(args.task_id)

    results = []
    for ep in range(args.n_episodes):
        env.reset()
        obs = env.set_init_state(init_states[ep])
        prev_frame = None
        n_replans = 0
        t0 = time.time()
        for t in range(args.max_steps):
            agentview = obs["agentview_image"][::-1, ::-1].copy()
            cur_frame = rgb_to_clip(agentview, device)
            if prev_frame is None:
                prev_frame = cur_frame
            if t % args.replan_every == 0:
                with torch.no_grad():
                    z_context = encoder(two_frame_clip(prev_frame, cur_frame))[0]
                eef_pos = np.array(obs["robot0_eef_pos"], dtype=np.float32)
                eef_quat = np.array(obs["robot0_eef_quat"], dtype=np.float32)
                axang = quat2axisangle(eef_quat).astype(np.float32)
                gripper_qpos = np.array(obs["robot0_gripper_qpos"], dtype=np.float32)
                state_vec = np.concatenate([eef_pos, axang, gripper_qpos[:1]]).astype(np.float32)
                action_seq = cem_plan(
                    predictor, z_context, state_vec, z_goal, device,
                    args.horizon, args.n_candidates, args.n_cem_iters,
                    args.elite_frac, args.init_std,
                )
                n_replans += 1
            action = action_seq[min(t % args.replan_every, action_seq.shape[0] - 1)].copy()
            action[:6] = np.clip(action[:6], -1.0, 1.0)
            obs, reward, done, info = env.step(action.tolist())
            prev_frame = cur_frame
            if env.check_success():
                break
        success = bool(env.check_success())
        elapsed = time.time() - t0
        print(f"  ep{ep}: success={success} steps={t+1} n_replans={n_replans} "
              f"wall_s={elapsed:.1f}")
        results.append({"episode": ep, "success": success, "steps": t + 1,
                         "n_replans": n_replans, "wall_s": elapsed})

    import json
    with open(os.path.join(args.results_dir, "results.json"), "w") as f:
        json.dump({"task_id": args.task_id, "config": vars(args), "results": results}, f, indent=2)
    n_success = sum(r["success"] for r in results)
    print(f"\n=== DONE: {n_success}/{len(results)} success ===")


if __name__ == "__main__":
    main()
