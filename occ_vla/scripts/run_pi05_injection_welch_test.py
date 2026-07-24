"""Welch's t-test comparing pi0.5's action distribution with vs. without
the dust3r-recovered image injected into OccVlaLiberoInputs'
`right_wrist_0_rgb` slot (user request, 2026-07-22) -- same methodology
already validated for the MMaDA subgoal slot ("OccVlaLiberoInputs wiring
... verified to actually influence pi0.5's output, Welch's t-test, N=20
calls, 3/7 action dims differ at p<0.0001 comparing subgoal_image=None
vs. a real duplicate image").

Fixed real occluded observation (base_image/wrist_image/state from
collect_occlusion_moment_with_state.py, bowl_top_drawer step 44,
arm_s_occ=0.339) and fixed recovered image (dust3r true-pose recovery,
generate_recovery_for_injection_test.py, S_occ=0.000, visually clean --
the validated bowl_top_drawer success case, not the mug_in_microwave
one with unresolved artifacts). N=20 independent pi0.5 calls per
condition (pi0.5's flow-matching sampler is stochastic run to run even
on identical input, per this project's own established finding) --
tests whether the INJECTED CONDITION alone changes the action
distribution, not a real rollout.

This does NOT test whether the change is *toward* the correct
(unoccluded ground-truth) action -- only whether pi0.5 responds to the
injected content at all, and how. A real rollout success-rate
comparison would be the next step if this shows a real, non-trivial
effect.

Requires pi05_worker running WITH occ_vla inputs enabled:
  PI05_WORKER_USE_OCC_VLA_INPUTS=1 scripts/run_with_gpu_cleanup.sh \
      python3 scripts/run_pi05_injection_welch_test.py
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
OUT_PATH = FRAME_DIR / "welch_test_results.json"


def call_pi05(base_image, wrist_image, state, prompt, subgoal_image=None):
    arrays = {"base_image": base_image, "wrist_image": wrist_image, "state": state}
    if subgoal_image is not None:
        arrays["subgoal_image"] = subgoal_image
    resp_arrays, _ = rpc.call(PI05_RPC_DIR, arrays, {"prompt": prompt})
    return resp_arrays["actions"]


def main():
    meta = json.loads((FRAME_DIR / "meta.json").read_text())
    base_image = np.array(Image.open(FRAME_DIR / "occluded_agentview_224.png").convert("RGB"))
    wrist_image = np.array(Image.open(FRAME_DIR / "occluded_wrist_224.png").convert("RGB"))
    recovered_image = np.array(Image.open(FRAME_DIR / "recovered_224.png").convert("RGB"))
    state = np.array(meta["state"], dtype=np.float32)
    prompt = meta["instruction"]

    print(f"prompt: {prompt}", flush=True)
    print(f"state: {state}", flush=True)
    print(f"arm_s_occ at collection: {meta['arm_s_occ']:.4f}", flush=True)

    baseline_actions, proposed_actions = [], []
    for i in range(N_CALLS):
        a = call_pi05(base_image, wrist_image, state, prompt, subgoal_image=None)
        baseline_actions.append(a[0])  # first action of the chunk
        print(f"baseline call {i}: action[0]={a[0]}", flush=True)
    for i in range(N_CALLS):
        a = call_pi05(base_image, wrist_image, state, prompt, subgoal_image=recovered_image)
        proposed_actions.append(a[0])
        print(f"proposed call {i}: action[0]={a[0]}", flush=True)

    baseline_actions = np.stack(baseline_actions)  # (N, action_dim)
    proposed_actions = np.stack(proposed_actions)
    action_dim = baseline_actions.shape[1]

    results = []
    for d in range(action_dim):
        t_stat, p_val = stats.ttest_ind(baseline_actions[:, d], proposed_actions[:, d], equal_var=False)
        results.append({
            "dim": d,
            "baseline_mean": float(baseline_actions[:, d].mean()),
            "baseline_std": float(baseline_actions[:, d].std()),
            "proposed_mean": float(proposed_actions[:, d].mean()),
            "proposed_std": float(proposed_actions[:, d].std()),
            "t_stat": float(t_stat),
            "p_value": float(p_val),
            "significant_p001": bool(p_val < 0.001),
        })
        print(
            f"dim {d}: baseline={results[-1]['baseline_mean']:.4f}+-{results[-1]['baseline_std']:.4f}  "
            f"proposed={results[-1]['proposed_mean']:.4f}+-{results[-1]['proposed_std']:.4f}  "
            f"t={t_stat:.3f} p={p_val:.6f} {'***' if p_val < 0.001 else ''}",
            flush=True,
        )

    n_sig = sum(r["significant_p001"] for r in results)
    print(f"\n{n_sig}/{action_dim} action dims differ at p<0.001", flush=True)

    OUT_PATH.write_text(json.dumps({
        "n_calls": N_CALLS, "action_dim": action_dim, "n_significant_p001": n_sig,
        "baseline_actions": baseline_actions.tolist(), "proposed_actions": proposed_actions.tolist(),
        "per_dim": results,
    }, indent=2))
    print(f"saved: {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
