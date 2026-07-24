"""Four-way pi0.5 injection comparison (user request, 2026-07-23):
Baseline (occluded, no injection) vs. SD-raw-injected (photorealistic-
leaning prompt, the "steampunk-style" one) vs. SD-styled-injected
(flat-shading LIBERO-style prompt) vs. GT (unoccluded, no injection).

Tests the user's hypothesis directly: does closing the style/domain gap
(test_sd_style_adapted.py's prompt steering) change how much the
injected content moves pi0.5's action toward GT, compared to the
raw/OOD-styled version? Same fixed (wrist_image, state, prompt) as the
existing pi05_injection_test/ frame; only base_image (occluded vs GT)
or the injected subgoal_image slot differs.

Requires pi05_worker running WITH occ_vla inputs enabled:
  PI05_WORKER_USE_OCC_VLA_INPUTS=1 scripts/run_with_gpu_cleanup.sh \
      python3 scripts/run_pi05_sd_style_comparison.py
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
OUT_PATH = FRAME_DIR / "sd_style_comparison_results.json"


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
    sd_raw = np.array(Image.open(FRAME_DIR / "raw_sd_224.png").convert("RGB"))
    sd_styled = np.array(Image.open(FRAME_DIR / "styled_smoothed_sd_224.png").convert("RGB"))
    state = np.array(meta["state"], dtype=np.float32)
    prompt = meta["instruction"]

    print(f"prompt: {prompt}", flush=True)

    conditions = {
        "baseline": (occluded_agentview, None),
        "sd_raw": (occluded_agentview, sd_raw),
        "sd_styled": (occluded_agentview, sd_styled),
        "gt": (gt_agentview, None),
    }
    all_actions = {}
    for name, (base_img, subgoal) in conditions.items():
        actions = []
        for i in range(N_CALLS):
            a = call_pi05(base_img, wrist_image, state, prompt, subgoal_image=subgoal)
            actions.append(a[0])
        all_actions[name] = np.stack(actions)
        print(f"{name}: done {N_CALLS} calls", flush=True)

    means = {name: acts.mean(axis=0) for name, acts in all_actions.items()}
    print("\n=== mean action vectors ===", flush=True)
    for name, m in means.items():
        print(f"{name}: {m}", flush=True)

    print("\n=== distance/similarity to GT ===", flush=True)
    dists = {}
    for name in ["baseline", "sd_raw", "sd_styled"]:
        cos = cosine_sim(means[name], means["gt"])
        dist = float(np.linalg.norm(means[name] - means["gt"]))
        dists[name] = dist
        print(f"{name} -> GT: cosine_sim={cos:.6f}  euclidean_dist={dist:.6f}", flush=True)

    print("\n=== ranking (closest to GT first) ===", flush=True)
    for name, d in sorted(dists.items(), key=lambda kv: kv[1]):
        print(f"  {name}: {d:.6f}", flush=True)

    action_dim = all_actions["baseline"].shape[1]
    per_dim = []
    for d in range(action_dim):
        row = {"dim": d}
        for name in ["baseline", "sd_raw", "sd_styled"]:
            row[f"{name}_err_to_gt"] = float(abs(means[name][d] - means["gt"][d]))
        per_dim.append(row)
        print(f"dim {d}: baseline={row['baseline_err_to_gt']:.4f}  sd_raw={row['sd_raw_err_to_gt']:.4f}  "
              f"sd_styled={row['sd_styled_err_to_gt']:.4f}", flush=True)

    OUT_PATH.write_text(json.dumps({
        "n_calls": N_CALLS,
        "actions": {k: v.tolist() for k, v in all_actions.items()},
        "means": {k: v.tolist() for k, v in means.items()},
        "dist_to_gt": dists,
        "per_dim": per_dim,
    }, indent=2))
    print(f"\nsaved: {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
