"""Week 1: generate our OWN init states for the candidate tasks (PREREG_main.md §1, §2).
LIBERO's placement sampler is not bit-reproducible from a seed (repeat calls with every RNG seeded differ by 2-17 mm),
so, like LIBERO's own .init files, the generated MuJoCo states themselves are the artifact (saved + sha256); the seed
of each is recorded for provenance only.
Overlap check: for each state, distance to every official init state of the same task = max over movable objects of
the position difference; a state closer than 1 mm to any official one is discarded.
Split: first 10 kept = dev, next 20 = eval, rest = spare.  Usage: python main/initstates.py --task T1 --n 40"""
import argparse, hashlib, json, os, random, sys
import numpy as np
import torch
_tl = torch.load
torch.load = lambda *a, **k: _tl(*a, **{**k, "weights_only": False})   # LIBERO's official .pt init files
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tasks import TASKS, bddl_path
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED_BASE = 900_000          # our seeds: 900000 + 1000 * task_number + i (LIBERO's official files are not seed-indexed)


def movable_pos(m, flat):
    """Positions of every non-robot free joint, from a flattened [time, qpos, qvel] state."""
    qpos = flat[1:1 + m.nq]
    return np.stack([qpos[m.jnt_qposadr[j]:m.jnt_qposadr[j] + 3] for j in range(m.njnt)
                     if m.jnt_type[j] == 0 and not m.joint(j).name.startswith(("robot", "gripper", "mount"))])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--task", required=True); ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()
    suite, idx, lang = TASKS[a.task]
    bddl, l2 = bddl_path(suite, idx); assert l2 == lang
    official = np.asarray(benchmark.get_benchmark_dict()[suite]().get_task_init_states(idx))
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=64, camera_widths=64)
    env.reset()
    m = env.sim.model._model
    off_pos = [movable_pos(m, s) for s in official]
    kept, log, i = [], [], 0
    while len(kept) < a.n and i < 5 * a.n:
        seed = SEED_BASE + 1000 * int(a.task[1:]) + i; i += 1
        env.seed(seed); np.random.seed(seed); random.seed(seed)
        env.reset()
        st = env.sim.get_state().flatten().copy()
        assert len(st) == official.shape[1], (len(st), official.shape)
        p = movable_pos(env.sim.model._model, st)
        dmin = min(float(np.abs(p - q).max()) for q in off_pos)
        ok = dmin >= 0.001
        log.append(dict(seed=seed, min_dist_to_official_m=round(dmin, 4), kept=ok))
        if ok:
            kept.append(st)
    kept = np.array(kept)
    out = os.path.join(HERE, "main", "init_states"); os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"{a.task}.npz")
    np.savez_compressed(path, dev=kept[:10], eval=kept[10:30], spare=kept[30:],
                        seeds=np.array([e["seed"] for e in log if e["kept"]]))
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    json.dump(dict(task=a.task, suite=suite, index=idx, language=lang, n_official=len(official), n_kept=len(kept),
                   n_tried=len(log), min_dist_to_official_m=dict(min=min(e["min_dist_to_official_m"] for e in log),
                   median=float(np.median([e["min_dist_to_official_m"] for e in log]))), sha256=sha, log=log),
              open(os.path.join(out, f"{a.task}.json"), "w"), indent=1)
    print(a.task, "kept", len(kept), "/ tried", len(log), "min dist to official (m):",
          min(e["min_dist_to_official_m"] for e in log), "sha256", sha[:12], flush=True)
    env.close()


if __name__ == "__main__":
    main()
