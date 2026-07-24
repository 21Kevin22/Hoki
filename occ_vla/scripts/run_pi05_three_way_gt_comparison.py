"""Three-way comparison (user request, 2026-07-22): Baseline (occluded,
no injection) vs. Proposed (occluded + dust3r-recovered image injected)
vs. GT (unoccluded base_image, same technique as arm_removal_pairs --
robot geoms alpha-zeroed at the identical sim state, no injection).

Answers the question the plain Welch's t-test (baseline vs. proposed
alone) couldn't: does the recovered-image injection move pi0.5's action
*toward* what it would do with a genuinely clear view (GT), or just move
it somewhere else? Per user's proposed method: compare mean-action
cosine similarity and Euclidean distance to GT for both baseline and
proposed, and check whether proposed is significantly closer.

Same fixed (base_image per condition, wrist_image, state, prompt) each
call within a condition; only base_image (GT vs. occluded) or the
injected slot (proposed vs. baseline) differs between conditions. N=20
independent pi0.5 calls per condition (stochastic sampler, matching
project convention).

Requires pi05_worker running WITH occ_vla inputs enabled:
  PI05_WORKER_USE_OCC_VLA_INPUTS=1 scripts/run_with_gpu_cleanup.sh \
      python3 scripts/run_pi05_three_way_gt_comparison.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import stats

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent / "_workers"))

import rpc  # noqa: E402

PI05_RPC_DIR = str(_ROOT / ".rpc" / "pi05")
FRAME_DIR = _ROOT / "texture_ceiling_probe" / "pi05_injection_test"
N_CALLS = 20
OUT_PATH = FRAME_DIR / "three_way_comparison_results.json"


def call_pi05(base_image, wrist_image, state, prompt, subgoal_image=None):
    arrays = {"base_image": base_image, "wrist_image": wrist_image, "state": state}
    if subgoal_image is not None:
        arrays["subgoal_image"] = subgoal_image
    resp_arrays, _ = rpc.call(PI05_RPC_DIR, arrays, {"prompt": prompt})
    return resp_arrays["actions"]


def cosine_sim(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def main():
    meta = json.loads((FRAME_DIR / "meta.json").read_text())
    occluded_agentview = np.array(Image.open(FRAME_DIR / "occluded_agentview_224.png").convert("RGB"))
    gt_agentview = np.array(Image.open(FRAME_DIR / "gt_agentview_224.png").convert("RGB"))
    wrist_image = np.array(Image.open(FRAME_DIR / "occluded_wrist_224.png").convert("RGB"))
    recovered_image = np.array(Image.open(FRAME_DIR / "recovered_224.png").convert("RGB"))
    state = np.array(meta["state"], dtype=np.float32)
    prompt = meta["instruction"]

    print(f"prompt: {prompt}", flush=True)
    print(f"arm_s_occ at collection: {meta['arm_s_occ']:.4f}", flush=True)

    conditions = {
        "baseline": (occluded_agentview, None),
        "proposed": (occluded_agentview, recovered_image),
        "gt": (gt_agentview, None),
    }
    all_actions = {}
    for name, (base_img, subgoal) in conditions.items():
        actions = []
        for i in range(N_CALLS):
            a = call_pi05(base_img, wrist_image, state, prompt, subgoal_image=subgoal)
            actions.append(a[0])
            print(f"{name} call {i}: action[0]={a[0]}", flush=True)
        all_actions[name] = np.stack(actions)

    action_dim = all_actions["baseline"].shape[1]
    means = {name: acts.mean(axis=0) for name, acts in all_actions.items()}

    print("\n=== mean action vectors ===", flush=True)
    for name, m in means.items():
        print(f"{name}: {m}", flush=True)

    baseline_to_gt_cos = cosine_sim(means["baseline"], means["gt"])
    proposed_to_gt_cos = cosine_sim(means["proposed"], means["gt"])
    baseline_to_gt_dist = float(np.linalg.norm(means["baseline"] - means["gt"]))
    proposed_to_gt_dist = float(np.linalg.norm(means["proposed"] - means["gt"]))

    print("\n=== distance/similarity to GT ===", flush=True)
    print(f"baseline -> GT: cosine_sim={baseline_to_gt_cos:.6f}  euclidean_dist={baseline_to_gt_dist:.6f}", flush=True)
    print(f"proposed -> GT: cosine_sim={proposed_to_gt_cos:.6f}  euclidean_dist={proposed_to_gt_dist:.6f}", flush=True)
    moved_toward_gt = proposed_to_gt_dist < baseline_to_gt_dist
    print(f"\nproposed is {'CLOSER to' if moved_toward_gt else 'FARTHER from'} GT than baseline "
          f"(Delta euclidean = {baseline_to_gt_dist - proposed_to_gt_dist:+.6f})", flush=True)

    # per-dimension: is |proposed[d] - gt[d]| smaller than |baseline[d] - gt[d]|?
    print("\n=== per-dimension distance to GT (baseline vs proposed) ===", flush=True)
    per_dim = []
    for d in range(action_dim):
        b_err = float(abs(means["baseline"][d] - means["gt"][d]))
        p_err = float(abs(means["proposed"][d] - means["gt"][d]))
        t_stat, p_val = stats.ttest_ind(all_actions["baseline"][:, d], all_actions["proposed"][:, d], equal_var=False)
        per_dim.append({
            "dim": d, "baseline_err_to_gt": b_err, "proposed_err_to_gt": p_err,
            "improved": p_err < b_err, "baseline_vs_proposed_p": float(p_val),
        })
        print(f"dim {d}: |baseline-GT|={b_err:.4f}  |proposed-GT|={p_err:.4f}  "
              f"{'IMPROVED' if p_err < b_err else 'worse/same'}", flush=True)
    n_improved = sum(r["improved"] for r in per_dim)
    print(f"\n{n_improved}/{action_dim} dims moved closer to GT", flush=True)

    OUT_PATH.write_text(json.dumps({
        "n_calls": N_CALLS,
        "actions": {k: v.tolist() for k, v in all_actions.items()},
        "means": {k: v.tolist() for k, v in means.items()},
        "baseline_to_gt_cosine": baseline_to_gt_cos,
        "proposed_to_gt_cosine": proposed_to_gt_cos,
        "baseline_to_gt_euclidean": baseline_to_gt_dist,
        "proposed_to_gt_euclidean": proposed_to_gt_dist,
        "proposed_closer_to_gt": moved_toward_gt,
        "per_dim": per_dim,
        "n_dims_improved": n_improved,
    }, indent=2))
    print(f"\nsaved: {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
