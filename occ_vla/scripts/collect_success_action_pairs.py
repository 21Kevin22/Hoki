"""occ_vla addition (2026-08-22), per user's revised self-contained
fine-tuning data plan (avoids the LIBERO-Occ external-data blocker
found this session -- litsh/Libero-Occ ships no training HDF5/
checkpoints): runs the REAL VLA policy on the CLEAN (stock, non-
occluded) suite, where it already succeeds 95-100% of the time, and at
every replan step saves a triple:

  I_clean     -- the real, unmodified agentview frame
  I_occ       -- I_clean with a REAL occluder sprite (captured once
                 from the occluded-suite version of the SAME task, via
                 the exact technique already validated for
                 `composite_visual_only`) composited on top
  a_clean     -- the action the policy ACTUALLY output for I_clean
                 (the real control signal driving this successful
                 rollout, not a guess)

No teleop, no expert-trajectory generation problem: labels come from
the model's OWN already-successful behavior on the easy (clean)
condition. Reuses run_libero_occluded_oracle_headroom.py's own
validated helper functions rather than reimplementing them.
"""
import argparse
import json
import os
import sys

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)
OFT_ROOT = os.path.normpath(os.path.join(SCRIPTS_DIR, "..", "thirdparty", "openvla-oft"))
sys.path.insert(0, OFT_ROOT)
os.chdir(OFT_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

_orig_load = torch.load
torch.load = lambda *a, **k: _orig_load(*a, **{**k, "weights_only": False})

import register_libero_occ_suites  # noqa: E402
from libero.libero import benchmark  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402

from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_wrist_image, quat2axisangle  # noqa: E402
from experiments.robot.libero.run_libero_eval import GenerateConfig, TASK_MAX_STEPS, check_unnorm_key, process_action  # noqa: E402
from experiments.robot.openvla_utils import get_action_head, get_proprio_projector, get_vla, get_vla_action, get_processor  # noqa: E402

from run_libero_occluded_oracle_headroom import (  # noqa: E402
    get_libero_env_seg, find_occluder_body_names, geom_ids_for_bodies,
    get_agentview_frames, find_segmentation_ids_for_bodies,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-ids", type=int, nargs="+", default=[1, 6], help="OCCLUDED-suite task ids (sprite source); stock-suite id resolved automatically")
    ap.add_argument("--n-episodes", type=int, default=20)
    ap.add_argument("--resolution", type=int, default=224)
    ap.add_argument("--checkpoint", default=os.path.expanduser("~/slocal1/Hoki/occ_vla/checkpoints/openvla-7b-oft-libero10-vjepa"))
    ap.add_argument("--out-dir", default="success_action_pairs")
    args = ap.parse_args()
    if not os.path.isabs(args.out_dir):
        args.out_dir = os.path.join(SCRIPTS_DIR, args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    bm = benchmark.get_benchmark_dict()
    occ_suite = bm["libero_10_occluded"]()
    stock_suite = bm["libero_10"]()

    cfg = GenerateConfig(
        pretrained_checkpoint=args.checkpoint,
        use_l1_regression=True, use_diffusion=False, use_film=False,
        num_images_in_input=2, use_proprio=True,
        load_in_8bit=False, load_in_4bit=False,
        center_crop=True, num_open_loop_steps=8, task_suite_name="libero_10", seed=7,
    )
    model = get_vla(cfg)
    check_unnorm_key(cfg, model)
    processor = get_processor(cfg)
    action_head = get_action_head(cfg, model.llm_dim)
    proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)

    manifest = []
    for occ_task_id in args.task_ids:
        occ_task = occ_suite.get_task(occ_task_id)
        # Resolve the matching stock task_id by shared bddl filename (same
        # convention established throughout this session).
        stock_task_id = None
        for i in range(10):
            if stock_suite.get_task(i).bddl_file == occ_task.bddl_file:
                stock_task_id = i
                break
        if stock_task_id is None:
            print(f"  [skip] task{occ_task_id}: no matching stock task found")
            continue
        stock_task = stock_suite.get_task(stock_task_id)
        task_description = stock_task.language

        # --- capture the static occluder sprite once, from the occluded-suite task ---
        sprite_env = get_libero_env_seg(occ_task, resolution=args.resolution)
        sprite_env.seed(0)
        sprite_env.reset()
        occluder_names = find_occluder_body_names(occ_task, stock_suite)
        # occ_vla bug fix (2026-08-22, same documented pattern as
        # run_libero_occluded_oracle_headroom.py's own 2026-08-18 fix):
        # find_occluder_body_names opens/closes 2 SEPARATE temp
        # OffScreenRenderEnv instances internally -- MuJoCo/robosuite's
        # offscreen EGL rendering shares process-global context state, so
        # closing those temp envs leaves THIS env's own render state
        # stale unless it's reset again afterward. Caught by a real,
        # reproduced symptom (occluder_pixel_mask.sum()==0 in the first
        # smoke test -- the composited "occluded" frames were silently
        # identical to the clean ones), not by inspection.
        sprite_env.reset()
        sprite_sim = sprite_env.env.sim
        occluder_geom_ids = geom_ids_for_bodies(sprite_sim, set(occluder_names)) if occluder_names else []
        if not occluder_geom_ids:
            print(f"  [skip] task{occ_task_id}: no occluder resolved")
            sprite_env.close()
            continue
        occluder_seg_ids = find_segmentation_ids_for_bodies(sprite_env, sprite_sim, occluder_geom_ids)
        sprite_color, sprite_seg = get_agentview_frames(sprite_env, args.resolution)
        occluder_pixel_mask = np.isin(sprite_seg, occluder_seg_ids) if occluder_seg_ids else np.zeros_like(sprite_seg, dtype=bool)
        sprite_env.close()
        print(f"  task{occ_task_id}: occluder sprite captured, mask px={int(occluder_pixel_mask.sum())}")

        # --- run real policy rollouts on the CLEAN (stock) task ---
        env = get_libero_env_seg(stock_task, resolution=args.resolution)
        env.seed(0)
        init_states = stock_suite.get_task_init_states(stock_task_id)
        max_steps = TASK_MAX_STEPS.get("libero_10", 520)

        n_saved = 0
        for ep in range(min(args.n_episodes, len(init_states))):
            env.reset()
            obs = env.set_init_state(init_states[ep])
            t = 0
            from collections import deque
            action_queue = deque(maxlen=cfg.num_open_loop_steps)
            for _ in range(10):
                obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
            success = False
            while t < max_steps + 10:
                agentview_color, _ = get_agentview_frames(env, args.resolution)
                wrist_img = get_libero_wrist_image(obs).copy()
                if len(action_queue) == 0:
                    observation = {
                        "full_image": agentview_color,
                        "wrist_image": wrist_img,
                        "state": np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])),
                    }
                    actions = get_vla_action(
                        cfg, model, processor, observation, task_description,
                        action_head=action_head, proprio_projector=proprio_projector,
                        noisy_action_projector=None, use_film=cfg.use_film,
                    )
                    action_queue.extend(actions)

                    # save the pair for THIS observation (before acting)
                    i_occ = agentview_color.copy()
                    i_occ[occluder_pixel_mask] = sprite_color[occluder_pixel_mask]
                    uid = f"task{occ_task_id}_ep{ep}_t{t:05d}"
                    Image.fromarray(agentview_color).save(os.path.join(args.out_dir, f"{uid}_clean.png"))
                    Image.fromarray(i_occ).save(os.path.join(args.out_dir, f"{uid}_occ.png"))
                    manifest.append({
                        "uid": uid, "task_id": occ_task_id, "episode": ep, "t": t,
                        "clean_path": f"{uid}_clean.png", "occ_path": f"{uid}_occ.png",
                        "action_clean": [float(x) for x in np.asarray(actions[0], dtype=float)],
                    })
                    n_saved += 1

                action = action_queue.popleft()
                action = process_action(action, cfg.model_family)
                obs, reward, done, info = env.step(action.tolist())
                if done:
                    success = True
                    break
                t += 1
            print(f"    ep{ep}: success={success} done_step={t}")
        env.close()
        print(f"  task{occ_task_id}: {n_saved} pairs saved")

    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\ntotal pairs: {len(manifest)}, saved to {args.out_dir}/manifest.json")


if __name__ == "__main__":
    main()
