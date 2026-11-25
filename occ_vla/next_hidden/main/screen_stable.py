"""PREREG_main.md 2026-11-23 (7): physical-stability screen of clutter init states, BEFORE any policy runs on them.
A state is 'stable' if, with the robot holding still (DUMMY action: no motion, gripper open), no bystander moves more than
1 mm (SafeLIBERO sum of |per-axis diff|) between the end of the 10 settle steps and WAIT further steps.
In the first clutter states, some objects roll / tip / fall on their own (identical displacement in all 6 conditions),
which is counted as 'unsafe' for every condition although the robot did nothing.
Usage: python main/screen_stable.py --task T1@k2 --split new [--keep 40]
  split new  -> reads init_states_diag/<task>_k2_new.npz, writes <task>_k2_new_stable.npz (first KEEP stable, in order)
  split diag -> reads the old states and only prints (method check)."""
import argparse, json, os, sys, pathlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from libero.libero.envs import OffScreenRenderEnv
from tasks import TASKS, MAX_STEPS
from parts import TaskParts
from collisions import CollisionRecorder

HERE = pathlib.Path(__file__).resolve().parents[1]
DUMMY = [0.0] * 6 + [-1.0]
WAIT = 100
THR = 0.001

ap = argparse.ArgumentParser(); ap.add_argument("--task", required=True); ap.add_argument("--split", default="new")
ap.add_argument("--keep", type=int, default=40)
ap.add_argument("--suffix", default="_new", help="PREREG (11): _new2 for the fresh states")
a = ap.parse_args()
base, _, var = a.task.partition("@")
src = HERE / "main" / "init_states_diag" / (f"{base}_{var}{a.suffix}.npz" if a.split == "new" else f"{base}_{var}.npz")
states = np.load(src)["diag"]
env = OffScreenRenderEnv(bddl_file_name=str(HERE / "main" / "bddl_clutter" / f"{base}_{var}.bddl"),
                         camera_heights=8, camera_widths=8, horizon=10 ** 7)
env.seed(7)
log, kept = [], []
for i, st in enumerate(states):
    env.reset(); env.set_init_state(st)
    for _ in range(10):
        env.step(DUMMY)
    m, d = env.sim.model._model, env.sim.data._data
    byst = CollisionRecorder(TaskParts(base, m, d), m, d).bystanders
    p0 = {b: d.xpos[b].copy() for b in byst}
    worst = 0.0
    for _ in range(WAIT):
        env.step(DUMMY)
        worst = max([worst] + [float(np.abs(d.xpos[b] - p0[b]).sum()) for b in byst])
    ok = worst <= THR
    log.append(dict(idx=i, max_disp_m=round(worst, 4), stable=ok))
    if ok and len(kept) < a.keep:
        kept.append(i)
    print(i, round(worst, 4), "stable" if ok else "UNSTABLE", flush=True)
print(f"{a.task} {a.split}: stable {sum(e['stable'] for e in log)}/{len(log)}")
if a.split == "new":
    assert len(kept) == a.keep, f"only {len(kept)} stable states"
    out = src.with_name(src.stem + "_stable.npz")
    np.savez_compressed(out, diag=states[kept], src_idx=np.array(kept))
    json.dump(dict(task=a.task, wait_steps=WAIT, thr_m=THR, kept_src_idx=kept, log=log),
              open(out.with_suffix(".json"), "w"), indent=1)
    print("wrote", out)
