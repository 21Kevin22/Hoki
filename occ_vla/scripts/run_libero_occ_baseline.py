"""
run_libero_occ_baseline.py

Step 2 of the LIBERO-Occ integration (occ_vla 2026-08-09): measures the
"None" (no correction) baseline severity of REAL, scene-induced occlusion
(actual 3D objects placed in the scene, confirmed via BDDL diff -- see
register_libero_occ_suites.py's docstring) on specific tasks, using the
stock (unmodified) OpenVLA-OFT checkpoint.

Deliberately does NOT reuse run_libero_eval_occlusion.py's run loop:
run_episode_occluded() unconditionally applies _apply_partial_patch (our own
synthetic gray-patch occlusion) on top of whatever the camera already
renders, whenever `occluded` is True -- for LIBERO-Occ's scenes, the real
occluder objects are ALREADY part of the rendered frame via scene geometry,
so running that script would double-occlude (synthetic patch stacked on top
of the real occluder) and confound the measurement. This script instead
drives the STOCK run_libero_eval.py's run_task() directly -- plain rollout,
zero occlusion-injection code path involved, since the occlusion here is
already baked into the env/BDDL, not something we inject per-frame.

Two runtime patches needed (see register_libero_occ_suites.py for the
benchmark-registration half); both done here since they're specific to
*evaluation*, not registration:
  1. validate_config()'s TaskSuite-enum assert doesn't know about our new
     "*_occluded" suite names.
  2. check_unnorm_key() looks up norm_stats under cfg.task_suite_name, but
     the checkpoint's norm_stats were only ever fit for the ORIGINAL suite
     name -- the occluded variant is the same task language/action
     distribution, just extra scene geometry, so this is a correct fallback,
     not a hack that changes what's being measured.
Both patches: if task_suite_name ends with "_occluded", temporarily swap to
the base suite name for that one call, restore after.

Usage (openvla-oft conda env):
  python scripts/run_libero_occ_baseline.py \
    --checkpoint checkpoints/openvla-7b-oft-libero10-vjepa \
    --task-suite libero_10_occluded --task-id 3 --num-trials 20 \
    --out-prefix libero_occ_none_baseline_moka_pots
(task-id 3 = KITCHEN_SCENE8_put_both_moka_pots_on_the_stove within
libero_10_occluded's own alphabetical indexing -- NOT the same index as the
original libero_10 suite's task8. Always verify via
get_benchmark("libero_10_occluded")().get_task_names(), never assume indices
carry over -- see register_libero_occ_suites.py's own printed task list.)
"""

import argparse
import json
import os
import sys

OCC_VLA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OFT_ROOT = os.path.join(OCC_VLA_ROOT, "thirdparty/openvla-oft")
SCRIPTS_DIR = os.path.join(OCC_VLA_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)
sys.path.insert(0, OFT_ROOT)
os.chdir(OFT_ROOT)
os.environ.setdefault("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero_oft"))

import register_libero_occ_suites  # noqa: E402  (registers the 4 occluded suites)

from libero.libero import benchmark  # noqa: E402

import experiments.robot.libero.run_libero_eval as rle  # noqa: E402
from experiments.robot.robot_utils import set_seed_everywhere  # noqa: E402

# Reuse the ORIGINAL suite's own calibrated max_steps for its occluded
# variant -- same task/physical solve, just extra static scene objects that
# don't block reachability (confirmed via BDDL diff: occluder objects are
# placed off the manipulation path, only in camera line-of-sight).
rle.TASK_MAX_STEPS.update({
    "libero_spatial_occluded": rle.TASK_MAX_STEPS[rle.TaskSuite.LIBERO_SPATIAL],
    "libero_object_occluded": rle.TASK_MAX_STEPS[rle.TaskSuite.LIBERO_OBJECT],
    "libero_goal_occluded": rle.TASK_MAX_STEPS[rle.TaskSuite.LIBERO_GOAL],
    "libero_10_occluded": rle.TASK_MAX_STEPS[rle.TaskSuite.LIBERO_10],
})

_orig_validate_config = rle.validate_config
_orig_check_unnorm_key = rle.check_unnorm_key


def _with_base_suite_name(fn):
    def wrapped(cfg, *args, **kwargs):
        if cfg.task_suite_name.endswith("_occluded"):
            real_name = cfg.task_suite_name
            cfg.task_suite_name = real_name[: -len("_occluded")]
            try:
                return fn(cfg, *args, **kwargs)
            finally:
                cfg.task_suite_name = real_name
        return fn(cfg, *args, **kwargs)
    return wrapped


rle.validate_config = _with_base_suite_name(_orig_validate_config)
rle.check_unnorm_key = _with_base_suite_name(_orig_check_unnorm_key)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task-suite", required=True, help="e.g. libero_10_occluded")
    parser.add_argument("--task-id", type=int, required=True,
                         help="index WITHIN the occluded suite's own task list -- verify with register_libero_occ_suites.py, don't assume it matches the original suite's numbering")
    parser.add_argument("--num-trials", type=int, default=20)
    parser.add_argument("--out-prefix", default="libero_occ_none_baseline")
    args = parser.parse_args()

    cfg = rle.GenerateConfig(
        pretrained_checkpoint=args.checkpoint,
        use_l1_regression=True, use_diffusion=False, use_film=False,
        num_images_in_input=2, use_proprio=True, load_in_8bit=False, load_in_4bit=False,
        center_crop=True, num_open_loop_steps=8, task_suite_name=args.task_suite,
        num_trials_per_task=args.num_trials, seed=7,
    )

    rle.validate_config(cfg)
    set_seed_everywhere(cfg.seed)
    model, action_head, proprio_projector, noisy_action_projector, processor = rle.initialize_model(cfg)
    resize_size = rle.get_image_resize_size(cfg)
    log_file, local_log_filepath, run_id = rle.setup_logging(cfg)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite]()
    task_name = task_suite.get_task_names()[args.task_id]
    print(f"Running {args.task_suite} task_id={args.task_id} ({task_name}), n={args.num_trials}")

    total_episodes, total_successes = rle.run_task(
        cfg, task_suite, args.task_id, model, resize_size, processor,
        action_head, proprio_projector, noisy_action_projector,
        0, 0, log_file,
    )
    rate = total_successes / total_episodes if total_episodes else 0.0
    print(f"Result: {total_successes}/{total_episodes} ({rate*100:.1f}%)")

    result = {
        "task_suite": args.task_suite, "task_id": args.task_id, "task_name": task_name,
        "checkpoint": args.checkpoint, "num_trials": args.num_trials,
        "successes": total_successes, "episodes": total_episodes, "rate": rate,
    }
    out_path = f"{args.out_prefix}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved {out_path}")

    if log_file:
        log_file.close()


if __name__ == "__main__":
    main()
