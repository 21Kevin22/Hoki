"""
run_libero_occ_benchmark.py

Batch natural-occlusion baseline scan across ALL libero_10 tasks (occ_vla,
2026-08-12), per the user's explicit request: find whether a "true 0%
task" exists elsewhere in libero_10 (mug_in_microwave, task_id=9, turned
out to be a 90-95% ceiling case for this checkpoint under REAL natural
occlusion -- see this session's own earlier findings), and whether
occlusion severity (S_occ) actually correlates with failure on it, before
picking a main testbed.

REAL BUG FOUND AND FIXED before the full run (2026-08-12): the first
version of this script measured self-occlusion as "any wrist-camera pixel
that changes when the robot's own geoms are hidden" -- but the wrist
camera is MOUNTED ON the gripper, so the gripper's own fingers are
essentially ALWAYS partially in its own frame by mounting geometry alone,
regardless of whether the manipulation TARGET is occluded. The smoke test
confirmed this: occ_mean sat at a suspiciously constant ~0.19-0.21 and
frac_above_threshold ~0.98-1.00 across every task, including ones with
totally different scenes/objects -- a signature of measuring "how much
screen area does my own gripper occupy" (roughly constant), not "is the
target occluded" (should vary a lot by task/scene geometry).

Fixed to the ALREADY-ESTABLISHED correct formula, per
[[feedback_segmentation_occlusion_measurement]]: capture the target
object's CLEAR (first-observed, presumed-unoccluded) segmentation
footprint once early in the episode, then at every step measure
S_occ = |clear_target_mask & robot_mask_now| / |clear_target_mask| --
the fraction of the ORIGINALLY-visible target region now covered by the
robot. Uses real per-camera instance segmentation
(`camera_segmentations="instance"`, confirmed available on LIBERO's
OffScreenRenderEnv -- NOT used anywhere else in this project's own
scripts before this one, all of which relied on the body-position-
translate/geom_rgba-hide diff trick instead). Segmentation IDs for the
robot and the per-task target are resolved empirically (hide the
candidate geoms, see which segmentation value's pixel count drops to
~0) rather than assumed, since robosuite's raw instance-ID numbering
isn't documented anywhere in this project.

Records per episode: success, done_step, occ_max, occ_mean,
frac_steps_above_threshold, and a best-effort "object_dropped" flag
(target object's final height notably below its own first-observed
height -- a crude proxy for task-destruction/drop failure modes; genuine
contact-force-based destruction detection is NOT implemented, no
force/torque sensors are wired into this project).

Usage:
  python scripts/run_libero_occ_benchmark.py --task-ids 0 1 2 --n-episodes 20 \\
      --results-dir libero_occ_benchmark_results
"""
import os
import sys
import json
import argparse
from collections import deque

sys.path.insert(0, os.path.dirname(__file__))
OFT_ROOT = os.path.join(os.path.dirname(__file__), "..", "thirdparty", "openvla-oft")
sys.path.insert(0, OFT_ROOT)
os.chdir(OFT_ROOT)

import numpy as np  # noqa: E402
import register_libero_occ_suites  # noqa: E402
from libero.libero import benchmark, get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402

from experiments.robot.libero.libero_utils import (  # noqa: E402
    get_libero_image, get_libero_wrist_image, get_libero_dummy_action, quat2axisangle,
)
from experiments.robot.libero.run_libero_eval import (  # noqa: E402
    GenerateConfig, check_unnorm_key, process_action, TASK_MAX_STEPS,
)
from experiments.robot.openvla_utils import (  # noqa: E402
    get_processor, get_vla_action, get_action_head, get_proprio_projector,
)
from experiments.robot.robot_utils import get_model, get_image_resize_size, set_seed_everywhere  # noqa: E402
import torch  # noqa: E402

CHECKPOINT = "/home/ubuntu/slocal/occ_vla/checkpoints/openvla-7b-oft-libero10-vjepa"
OCCLUSION_TRIGGER_THRESHOLD = 0.15
SEG_KEY = "robot0_eye_in_hand_segmentation_instance"  # wrist camera -- matches this project's own
                                                        # established framing (wrist-camera occlusion),
                                                        # e.g. task 9's mug_in_microwave precedent.


def get_libero_env_seg(task, resolution=256):
    """Same as experiments/robot/libero/libero_utils.py::get_libero_env, but requests
    per-camera instance segmentation too -- NOT changing that shared helper (used by many
    other scripts in this project) since this is the only script that needs it."""
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=task_bddl_file, camera_heights=resolution, camera_widths=resolution,
                              camera_segmentations="instance")
    env.seed(0)
    return env


def geom_ids_by_substring(sim, substrings):
    ids = []
    for i in range(sim.model.ngeom):
        body_id = sim.model.geom_bodyid[i]
        body_name = (sim.model.body_id2name(body_id) or "").lower()
        if any(s in body_name for s in substrings):
            ids.append(i)
    return ids


def get_wrist_seg(env):
    obs = env.env._get_observations(force_update=True)
    seg = obs[SEG_KEY][::-1, ::-1, 0].copy()  # same 180-degree flip as get_libero_wrist_image
    return seg


def find_segmentation_ids(env, sim, geom_ids):
    """Empirically determine which instance-segmentation pixel values belong to the given
    geoms: render, hide those geoms (alpha=0), render again, and see which values' pixel
    counts collapsed to ~0. Avoids assuming robosuite's internal instance-ID numbering,
    which isn't documented anywhere in this project."""
    seg_before = get_wrist_seg(env)
    counts_before = {int(v): int((seg_before == v).sum()) for v in np.unique(seg_before)}

    orig_alpha = sim.model.geom_rgba[geom_ids, 3].copy()
    sim.model.geom_rgba[geom_ids, 3] = 0.0
    sim.forward()
    seg_after = get_wrist_seg(env)
    sim.model.geom_rgba[geom_ids, 3] = orig_alpha
    sim.forward()
    counts_after = {int(v): int((seg_after == v).sum()) for v in np.unique(seg_after)}

    ids = [v for v, c in counts_before.items() if counts_after.get(v, 0) < 0.1 * c]
    return ids


def run_episode(cfg, env, task_description, model, processor, action_head, proprio_projector,
                 init_state, robot_geom_ids_, target_seg_ids, obj_body_name, max_steps):
    env.reset()
    obs = env.set_init_state(init_state)
    if hasattr(model, "reset_vjepa_state"):
        model.reset_vjepa_state()
    sim = env.env.sim  # re-fetch AFTER reset (stale-reference bug, established earlier this session)

    initial_obj_z, final_obj_z = None, None
    try:
        body_id = sim.model.body_name2id(obj_body_name)
        initial_obj_z = float(sim.data.body_xpos[body_id][2])
    except Exception:
        pass

    action_queue = deque(maxlen=cfg.num_open_loop_steps)
    t = 0
    success = False
    occlusion_trace = []
    revealed_px_trace = []  # diagnostic: how many target pixels WOULD be visible sans-robot each step

    for _ in range(cfg.num_steps_wait):
        obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
        t += 1

    while t < max_steps + cfg.num_steps_wait:
        # LIVE per-step hide-and-reveal (2026-08-12 fix #2): the wrist camera MOVES WITH THE ARM,
        # so a "clear target footprint" captured once in screen-space coordinates (t=0) stops
        # meaning anything the instant the camera moves -- comparing it against later frames'
        # screen-space robot position is comparing two different cameras' worth of pixels as if
        # they were the same. Fixed to compare, at THIS SAME step/camera-pose, the target's
        # currently-visible pixel count against what it WOULD be if the robot were hidden RIGHT
        # NOW (a fresh render every step, matching derive_natural_mask_and_gt's own established
        # per-step cost pattern) -- both renders share the identical (current) camera pose, so the
        # comparison is actually valid regardless of how much the wrist camera has moved.
        target_px_now = int(np.isin(get_wrist_seg(env), target_seg_ids).sum())
        orig_alpha = sim.model.geom_rgba[robot_geom_ids_, 3].copy()
        sim.model.geom_rgba[robot_geom_ids_, 3] = 0.0
        sim.forward()
        target_px_revealed = int(np.isin(get_wrist_seg(env), target_seg_ids).sum())
        sim.model.geom_rgba[robot_geom_ids_, 3] = orig_alpha
        sim.forward()
        s_occ = float(1.0 - target_px_now / target_px_revealed) if target_px_revealed > 0 else 0.0
        occlusion_trace.append(s_occ)
        revealed_px_trace.append(target_px_revealed)

        if len(action_queue) == 0:
            observation = {
                "full_image": get_libero_image(obs).copy(),
                "wrist_image": get_libero_wrist_image(obs).copy(),
                "state": np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])),
            }
            with torch.inference_mode():
                actions = get_vla_action(
                    cfg, model, processor, observation, task_description,
                    action_head=action_head, proprio_projector=proprio_projector,
                    noisy_action_projector=None, use_film=cfg.use_film, occlusion_mask=None,
                )
            action_queue.extend(actions)
        action = np.array(action_queue.popleft(), dtype=float)
        action = process_action(action, cfg.model_family)
        obs, reward, done, info = env.step(action.tolist())
        t += 1
        if done:
            success = True
            break

    if initial_obj_z is not None:
        try:
            body_id = sim.model.body_name2id(obj_body_name)
            final_obj_z = float(sim.data.body_xpos[body_id][2])
        except Exception:
            pass
    object_dropped = (
        initial_obj_z is not None and final_obj_z is not None and final_obj_z < initial_obj_z - 0.05
    )

    return {
        "success": success,
        "done_step": t - cfg.num_steps_wait,
        "timeout": (not success) and (t - cfg.num_steps_wait >= max_steps),
        "mean_revealed_target_px": float(np.mean(revealed_px_trace)) if revealed_px_trace else 0.0,
        "frac_steps_target_never_visible": sum(1 for p in revealed_px_trace if p == 0) / len(revealed_px_trace) if revealed_px_trace else None,
        "occ_min": float(min(occlusion_trace)) if occlusion_trace else None,
        "occ_max": float(max(occlusion_trace)) if occlusion_trace else None,
        "occ_mean": float(np.mean(occlusion_trace)) if occlusion_trace else None,
        "n_steps_above_threshold": sum(1 for s in occlusion_trace if s > OCCLUSION_TRIGGER_THRESHOLD),
        "frac_steps_above_threshold": (sum(1 for s in occlusion_trace if s > OCCLUSION_TRIGGER_THRESHOLD) / len(occlusion_trace)) if occlusion_trace else None,
        "object_dropped": object_dropped,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-ids", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument("--results-dir", type=str, default="libero_occ_benchmark_results")
    args = parser.parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    cfg = GenerateConfig(
        pretrained_checkpoint=CHECKPOINT,
        use_l1_regression=True, use_diffusion=False, use_film=False,
        num_images_in_input=2, use_proprio=True, load_in_8bit=False, load_in_4bit=False,
        center_crop=True, num_open_loop_steps=8, task_suite_name="libero_10", seed=7,
    )
    set_seed_everywhere(cfg.seed)
    model = get_model(cfg)
    processor = get_processor(cfg)
    check_unnorm_key(cfg, model)
    proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)
    action_head = get_action_head(cfg, model.llm_dim)
    resize_size = get_image_resize_size(cfg)
    max_steps = TASK_MAX_STEPS["libero_10"]

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict["libero_10"]()

    summary = {}
    for task_id in args.task_ids:
        task = task_suite.get_task(task_id)
        task_description = task.language
        init_states = task_suite.get_task_init_states(task_id)
        n = min(args.n_episodes, len(init_states))
        print(f"\n[bench] === task_id={task_id} '{task_description}' n={n} ===")

        env = get_libero_env_seg(task, resolution=resize_size)
        env.reset()
        env.set_init_state(init_states[0])
        # Settle first -- resolving segmentation IDs immediately after set_init_state can hit a
        # transient frame where a candidate object isn't rendered/visible yet from the wrist
        # camera at all (real bug hit on task_id=4's first candidate, "plate_1" -- 0 pixels
        # before AND after hiding it, so no count ever "drops", and find_segmentation_ids
        # legitimately returns empty). A few dummy steps gives a more representative frame.
        for _ in range(cfg.num_steps_wait):
            env.step(get_libero_dummy_action(cfg.model_family))
        sim = env.env.sim
        robot_geom_ids_ = geom_ids_by_substring(sim, ("robot", "panda", "gripper", "mount"))
        assert robot_geom_ids_, "no robot geoms matched"
        robot_seg_ids = find_segmentation_ids(env, sim, robot_geom_ids_)
        assert robot_seg_ids, "could not resolve robot segmentation id(s)"

        # Try EACH obj_of_interest candidate in order (a task can list several -- e.g. task_id=4
        # lists ['plate_1', 'plate_2', 'white_yellow_mug_1', 'porcelain_mug_1']) and use the first
        # one that actually resolves to real geoms AND a real, currently-visible segmentation id --
        # rather than assuming index 0 is always usable (real bug: index 0 for task_id=4 is a
        # plate that isn't wrist-camera-visible at episode start).
        obj_body_name, target_geom_ids, target_seg_ids = None, None, None
        for candidate in env.obj_of_interest:
            cand_geom_ids = geom_ids_by_substring(sim, (candidate.lower(),))
            if not cand_geom_ids:
                print(f"[bench] task{task_id} candidate '{candidate}': no geoms matched, skipping")
                continue
            cand_seg_ids = find_segmentation_ids(env, sim, cand_geom_ids)
            if not cand_seg_ids:
                print(f"[bench] task{task_id} candidate '{candidate}': geoms found but not currently "
                      f"visible in wrist camera (empty segmentation id set), skipping")
                continue
            obj_body_name, target_geom_ids, target_seg_ids = candidate, cand_geom_ids, cand_seg_ids
            break
        if target_seg_ids is None:
            print(f"[bench] task{task_id} SKIPPED -- none of {env.obj_of_interest} resolved to a "
                  f"visible, hideable target. Moving to next task.")
            continue
        print(f"[bench] task{task_id} target='{obj_body_name}' robot_seg_ids={robot_seg_ids} target_seg_ids={target_seg_ids}")

        results = []
        for i in range(n):
            res = run_episode(cfg, env, task_description, model, processor, action_head, proprio_projector,
                               init_states[i], robot_geom_ids_, target_seg_ids, obj_body_name, max_steps)
            res["episode_idx"] = i
            results.append(res)
            occ_max_s = f"{res['occ_max']:.3f}" if res["occ_max"] is not None else "NA"
            occ_mean_s = f"{res['occ_mean']:.3f}" if res["occ_mean"] is not None else "NA"
            frac_s = f"{res['frac_steps_above_threshold']:.2f}" if res["frac_steps_above_threshold"] is not None else "NA"
            print(f"[bench] task{task_id} ep{i} success={res['success']} done_step={res['done_step']} "
                  f"mean_revealed_px={res['mean_revealed_target_px']:.1f} occ_max={occ_max_s} occ_mean={occ_mean_s} "
                  f"frac_above_thresh={frac_s} dropped={res['object_dropped']}")
            with open(os.path.join(args.results_dir, f"task{task_id}.json"), "w") as f:
                json.dump({"task_id": task_id, "task_description": task_description,
                           "target_body": obj_body_name, "results": results}, f, indent=2)

        succ = sum(1 for r in results if r["success"])
        timeouts = sum(1 for r in results if r["timeout"])
        dropped = sum(1 for r in results if r["object_dropped"])
        occ_max_succ = [r["occ_max"] for r in results if r["success"] and r["occ_max"] is not None]
        occ_max_fail = [r["occ_max"] for r in results if not r["success"] and r["occ_max"] is not None]
        summary[task_id] = {
            "task_description": task_description, "target_body": obj_body_name, "n": n, "success": succ,
            "success_rate": succ / n, "timeouts": timeouts, "object_dropped": dropped,
            "mean_occ_max_success_eps": float(np.mean(occ_max_succ)) if occ_max_succ else None,
            "mean_occ_max_fail_eps": float(np.mean(occ_max_fail)) if occ_max_fail else None,
        }
        print(f"[bench] task{task_id} FINAL success={succ}/{n} ({100*succ/n:.1f}%) timeouts={timeouts} "
              f"dropped={dropped} mean_occ_max(success_eps)={summary[task_id]['mean_occ_max_success_eps']} "
              f"mean_occ_max(fail_eps)={summary[task_id]['mean_occ_max_fail_eps']}")
        with open(os.path.join(args.results_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    print("\n[bench] === ALL TASKS DONE ===")
    for tid, s in sorted(summary.items(), key=lambda kv: kv[1]["success_rate"]):
        print(f"  task{tid} '{s['task_description'][:50]}': {s['success']}/{s['n']} "
              f"({100*s['success_rate']:.1f}%) timeouts={s['timeouts']} dropped={s['object_dropped']}")


if __name__ == "__main__":
    main()
