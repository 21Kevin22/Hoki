"""Generates paired (arm-visible, arm-removed) ground-truth images for
a future LoRA fine-tune of MMaDA's arm-removal inpainting, using
robosuite/MuJoCo's privileged simulator access: render the identical
sim state twice -- once normally, once with the robot's geoms hidden
(alpha=0) -- giving a perfectly-aligned pair with zero generative
guessing involved, unlike anything MMaDA has zero-shot produced this
session (see occ_vla/CLAUDE.md's blob-collapse investigation).
Confirmed working via a manual spot check (2026-07-17): the "clear"
render cleanly reveals real background geometry, no artifacts.

No GPU/pi0.5 needed (per user preference -- a low-cost distribution
check before committing to LoRA training infra): arm poses come from
small random action deltas, not a real policy rollout. Also saves the
arm's segmentation mask (captured before hiding) since a future
masked-token training loop will need it.

Run: python3 scripts/collect_arm_removal_pairs.py [--episodes-per-task N] [--samples-per-episode N]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_orig_torch_load = torch.load
torch.load = lambda *a, **k: _orig_torch_load(*a, **{**k, "weights_only": False})

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "third_party/openpi/third_party/libero"))

from occ_vla.eval.libero_occ_env import AGENTVIEW_KEY, AGENTVIEW_SEGMENTATION_KEY, LiberoOccEnv, LiberoOccEnvConfig  # noqa: E402
from occ_vla.eval.metrics import Difficulty  # noqa: E402

NUM_STEPS_WAIT = 10
SEED = 7
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
EPISODES_PER_TASK = 3
SAMPLES_PER_EPISODE = 15
STEPS_BETWEEN_SAMPLES = 8
ACTION_POS_SCALE = 0.3
ACTION_ROT_SCALE = 0.2

TASKS = [
    {"suite": "libero_10", "task_id": 8, "label": "moka_pots"},
    {"suite": "libero_10", "task_id": 3, "label": "bowl_in_drawer"},
    {"suite": "libero_10", "task_id": 9, "label": "mug_in_microwave"},
    {"suite": "libero_spatial", "task_id": 4, "label": "bowl_top_drawer"},
    {"suite": "libero_10", "task_id": 4, "label": "mugs_on_plates"},
]

OUT_DIR = _ROOT / "arm_removal_pairs"


def robot_geom_ids(model) -> list[int]:
    body_names = [model.body_id2name(i) for i in range(model.nbody)]
    robot_body_ids = {
        i for i, n in enumerate(body_names) if n and any(k in n.lower() for k in ("robot", "panda", "gripper", "mount"))
    }
    geom_bodyid = model.geom_bodyid
    return [g for g in range(model.ngeom) if geom_bodyid[g] in robot_body_ids]


def render_pair(occ_env) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (arm_visible_image, arm_free_image, arm_pixel_mask) for
    the *current* sim state (no time advance)."""
    env = occ_env._env  # noqa: SLF001
    model = env.sim.model
    geom_ids = robot_geom_ids(model)

    state = env.get_sim_state()
    visible_obs = env.regenerate_obs_from_state(state)
    visible_img = visible_obs[AGENTVIEW_KEY].copy()

    seg_dict = env.get_segmentation_instances(visible_obs[AGENTVIEW_SEGMENTATION_KEY])
    arm_mask = (seg_dict["robot"].squeeze(-1) != 0) if "robot" in seg_dict else np.zeros(visible_img.shape[:2], dtype=bool)

    orig_rgba = model.geom_rgba[geom_ids].copy()
    model.geom_rgba[geom_ids, 3] = 0.0
    clear_obs = env.regenerate_obs_from_state(state)
    clear_img = clear_obs[AGENTVIEW_KEY].copy()
    model.geom_rgba[geom_ids] = orig_rgba

    return visible_img, clear_img, arm_mask


def collect_task(suite: str, task_id: int, label: str, episodes: int, samples_per_episode: int) -> list[dict]:
    from libero.libero import benchmark  # noqa: PLC0415

    bench = benchmark.get_benchmark(suite)()
    instruction = bench.get_task(task_id).language
    init_states = bench.get_task_init_states(task_id)
    rng = np.random.default_rng(hash(label) % (2**32))

    manifest = []
    for episode_idx in range(episodes):
        config = LiberoOccEnvConfig(
            benchmark_suite=suite, task_id=task_id, difficulty=Difficulty.LIGHT,
            init_state_idx=episode_idx % len(init_states), seed=SEED, place_occluder=False,
        )
        occ_env = LiberoOccEnv(config, libero_root=str(_ROOT / "third_party/openpi/third_party/libero"))
        obs = occ_env.reset()
        for _ in range(NUM_STEPS_WAIT):
            obs, _, _, _ = occ_env.step(LIBERO_DUMMY_ACTION)

        gripper = -1.0
        for sample_idx in range(samples_per_episode):
            for _ in range(STEPS_BETWEEN_SAMPLES):
                pos = rng.uniform(-ACTION_POS_SCALE, ACTION_POS_SCALE, size=3)
                rot = rng.uniform(-ACTION_ROT_SCALE, ACTION_ROT_SCALE, size=3)
                if rng.random() < 0.05:
                    gripper *= -1.0
                action = np.concatenate([pos, rot, [gripper]]).tolist()
                obs, _, done, _ = occ_env.step(action)
                if done:
                    break

            visible_img, clear_img, arm_mask = render_pair(occ_env)
            if not arm_mask.any():
                continue  # arm not in frame at all -- not a useful "occlusion" pair

            stub = f"{label}_ep{episode_idx:02d}_s{sample_idx:03d}"
            Image.fromarray(visible_img).save(OUT_DIR / f"{stub}_armvis.png")
            Image.fromarray(clear_img).save(OUT_DIR / f"{stub}_armfree.png")
            Image.fromarray((arm_mask.astype(np.uint8) * 255)).save(OUT_DIR / f"{stub}_armmask.png")
            manifest.append({"stub": stub, "label": label, "instruction": instruction, "arm_px": int(arm_mask.sum())})

    print(f"[{label}] {len(manifest)} pairs saved", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-per-task", type=int, default=EPISODES_PER_TASK)
    parser.add_argument("--samples-per-episode", type=int, default=SAMPLES_PER_EPISODE)
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    manifest = []
    for task in TASKS:
        manifest.extend(collect_task(task["suite"], task["task_id"], task["label"], args.episodes_per_task, args.samples_per_episode))
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\ndone -- {len(manifest)} total pairs in {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
