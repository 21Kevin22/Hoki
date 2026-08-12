"""
scan_wrist_occlusion_libero_occ.py

Fail-fast scope check requested by the user (2026-08-09): before investing in
a segmentation-based Semantic Blanking pipeline, scan ALL LIBERO-Occ tasks
(4 suites) to find which ones, if any, actually occlude the WRIST camera --
the only camera our vjepa correction module operates on. moka_pots
(libero_10_occluded task_id=3) was already checked by hand and showed ZERO
wrist occlusion across 160+ scanned steps (dummy no-ops + a scripted
downward reach) despite dramatic agentview occlusion -- this script checks
whether that's typical of the whole benchmark or specific to that one task.

No OpenVLA-OFT model loaded at all -- pure LIBERO/robosuite simulation +
element-level segmentation (`camera_segmentations="element"`, confirmed
supported by this robosuite version via demos/demo_segmentation.py), so this
is fast and doesn't need a GPU. Segmentation is used instead of the
alpha-toggle diff trick from test_natural_occlusion_generalization.py --
that trick was tried and found NOT to work in this robosuite/mujoco setup
(zeroing geom_rgba[...,3] did not change the render at all, confirmed via
a bit-identical before/after image) -- segmentation sidesteps this cleanly
and is also the actual mechanism the user's original Semantic Blanking
proposal called for.

Occluder object identification: for each occluded task, diffs its BDDL
against the ORIGINAL (non-occluded) suite's same-named file and extracts
newly-declared `(:objects ...)` entries (regex on "name - class" lines
present in the occluded file but absent from the original). This covers
the majority pattern seen across all 4 suites (a wholly new object added,
e.g. wooden_cabinet_1/short_cabinet_1/microwave_1/wine_rack_1/
white_storage_box_1/wooden_two_layer_shelf_1 -- confirmed via manual diff
inspection, 2026-08-09). Tasks where the occluded variant instead
repositions an ALREADY-existing object (no new declaration -- confirmed to
occur too, e.g. one libero_spatial task) are skipped with a warning rather
than silently mishandled, since that needs a different (region->object)
lookup this script doesn't attempt.

Scan motion: 10 dummy no-op steps (let physics settle) + up to
--scan-steps small-random steps biased toward reaching down/forward
(same profile already used for the moka_pots hand-check), checking wrist
segmentation every --check-every steps.

Usage: python scripts/scan_wrist_occlusion_libero_occ.py [--suites ...] [--scan-steps 150]
"""

import argparse
import glob
import json
import os
import re
import sys

OCC_VLA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OFT_ROOT = os.path.join(OCC_VLA_ROOT, "thirdparty/openvla-oft")
LIBERO_ROOT = os.path.join(OCC_VLA_ROOT, "thirdparty/LIBERO")
LIBERO_OCC_ROOT = os.path.join(OCC_VLA_ROOT, "third_party/Libero-Occ")
SCRIPTS_DIR = os.path.join(OCC_VLA_ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)
sys.path.insert(0, OFT_ROOT)
os.chdir(OFT_ROOT)
os.environ.setdefault("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero_oft"))

import numpy as np  # noqa: E402

import register_libero_occ_suites  # noqa: E402
from libero.libero import benchmark, get_libero_path  # noqa: E402
from experiments.robot.libero.run_libero_eval import get_libero_dummy_action  # noqa: E402

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]

OBJ_DECL_RE = re.compile(r"^\s*(\S+_\d+)\s*-\s*\S+\s*$", re.MULTILINE)


def find_new_objects(base_suite, task_name):
    orig_path = os.path.join(LIBERO_ROOT, "libero/libero/bddl_files", base_suite, f"{task_name}.bddl")
    occ_path = os.path.join(LIBERO_OCC_ROOT, "benchmark_assets/bddl_files", f"{base_suite}_occluded", f"{task_name}.bddl")
    if not (os.path.isfile(orig_path) and os.path.isfile(occ_path)):
        return None
    orig_objs = set(OBJ_DECL_RE.findall(open(orig_path).read()))
    occ_objs = set(OBJ_DECL_RE.findall(open(occ_path).read()))
    return sorted(occ_objs - orig_objs)


def scan_task(base_suite, task_id, new_objects, scan_steps, check_every):
    from libero.libero.envs import OffScreenRenderEnv

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[f"{base_suite}_occluded"]()
    task = task_suite.get_task(task_id)
    bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl_file, camera_heights=224, camera_widths=224,
                              camera_segmentations="element")
    env.seed(0)
    env.reset()
    init_states = task_suite.get_task_init_states(task_id)
    env.set_init_state(init_states[0])

    sim = env.env.sim
    all_names = [sim.model.geom_id2name(i) for i in range(sim.model.ngeom)]
    occ_geom_ids = set(
        i for i, n in enumerate(all_names)
        if n and any(n.startswith(obj + "_") or n == obj for obj in new_objects)
    )
    if not occ_geom_ids:
        env.close()
        return None, "no matching geoms found for " + ",".join(new_objects)

    dummy = get_libero_dummy_action("openvla")
    for _ in range(10):
        env.step(dummy)

    rng = np.random.default_rng(0)
    max_frac = 0.0
    max_step = -1
    for t in range(scan_steps):
        action = np.array([0.0, 0.0, -0.3, 0.0, 0.0, 0.0, -1.0]) + rng.normal(0, 0.05, size=7)
        action[6] = -1.0
        env.step(np.clip(action, -1, 1).tolist())
        if t % check_every == 0:
            obs = env.env._get_observations()
            seg = obs["robot0_eye_in_hand_segmentation_element"].squeeze(-1)
            frac = float(np.isin(seg, list(occ_geom_ids)).mean())
            if frac > max_frac:
                max_frac, max_step = frac, t

    env.close()
    return max_frac, max_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suites", nargs="+", default=SUITES)
    parser.add_argument("--scan-steps", type=int, default=150)
    parser.add_argument("--check-every", type=int, default=10)
    parser.add_argument("--out", default="wrist_occlusion_scan_results.json")
    args = parser.parse_args()

    results = []
    for base_suite in args.suites:
        occ_dir = os.path.join(LIBERO_OCC_ROOT, "benchmark_assets/bddl_files", f"{base_suite}_occluded")
        task_names = sorted(
            os.path.splitext(os.path.basename(p))[0]
            for p in glob.glob(os.path.join(occ_dir, "*.bddl"))
        )
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[f"{base_suite}_occluded"]()
        suite_task_names = task_suite.get_task_names()

        for task_name in task_names:
            task_id = suite_task_names.index(task_name)
            new_objects = find_new_objects(base_suite, task_name)
            if not new_objects:
                print(f"[skip] {base_suite}/{task_name}: no newly-declared occluder object found "
                      f"(likely a repositioned-existing-object case, not handled by this script)")
                results.append({"suite": base_suite, "task_id": task_id, "task_name": task_name,
                                 "status": "skipped_no_new_object"})
                continue
            print(f"[scan] {base_suite}_occluded task_id={task_id} ({task_name}), occluder={new_objects} ...", flush=True)
            try:
                max_frac, max_step = scan_task(base_suite, task_id, new_objects, args.scan_steps, args.check_every)
            except Exception as e:
                print(f"  ERROR: {e}")
                results.append({"suite": base_suite, "task_id": task_id, "task_name": task_name,
                                 "occluder_objects": new_objects, "status": f"error: {e}"})
                continue
            if max_frac is None:
                print(f"  {max_step}")
                results.append({"suite": base_suite, "task_id": task_id, "task_name": task_name,
                                 "occluder_objects": new_objects, "status": max_step})
                continue
            print(f"  max wrist-segmentation occlusion fraction: {max_frac*100:.2f}% (at scan step {max_step})")
            results.append({
                "suite": base_suite, "task_id": task_id, "task_name": task_name,
                "occluder_objects": new_objects, "status": "ok",
                "max_wrist_occlusion_frac": max_frac, "max_step": max_step,
            })
            with open(args.out, "w") as f:
                json.dump(results, f, indent=2)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Summary (sorted by max wrist occlusion) ===")
    ok = [r for r in results if r.get("status") == "ok"]
    ok.sort(key=lambda r: -r["max_wrist_occlusion_frac"])
    for r in ok:
        print(f"  {r['max_wrist_occlusion_frac']*100:6.2f}%  {r['suite']:15s} task{r['task_id']:2d}  {r['task_name']}")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
