"""6-1: reload every saved init state through LIBERO's own env.set_init_state (exactly as a rollout would),
settle 10 zero-action steps (openpi's num_steps_wait), then 40 more, and re-run all checks.
Writes results/init/verify.json. Exit code 1 if any state fails."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_init_states import HERE, OBJ, make_env, zero_steps, check_state

env = make_env()
states = np.load(os.path.join(HERE, "results", "init", "init_states.npz"))["states"]
import mujoco
rows = []
# two passes: states loaded back-to-back in one env (as a rollout loop does), then each with a cleared solver
# warm-start; a marginally stable mug slides ~3 mm depending on this solver history
for i, s in [(i, s) for i, s in enumerate(states)] + [(-1 - i, s) for i, s in enumerate(states)]:
    env.reset()
    env.set_init_state(s)
    m, d = env.sim.model._model, env.sim.data._data
    if i < 0:
        d.qacc_warmstart[:] = 0; mujoco.mj_forward(m, d)
    qa = m.joint(f"{OBJ}_joint0").qposadr[0]
    zero_steps(env, 10)
    p0 = d.qpos[qa:qa + 3].copy()
    obs = zero_steps(env, 40)
    c = dict(state=i if i >= 0 else -1 - i, warmstart_cleared=i < 0, **check_state(env, obs, p0, d.qpos[qa:qa + 3].copy()))
    rows.append(c)
    print(json.dumps({k: c[k] for k in ["state", "closed", "inside", "goal_false", "pixels_agentview",
                                         "pixels_wrist", "drift_mm", "deepest_penetration_mm", "pass"]}), flush=True)
json.dump(dict(render="egl", n=len(rows), n_pass=sum(r["pass"] for r in rows), rows=rows),
          open(os.path.join(HERE, "results", "init", "verify.json"), "w"), indent=1)
print(f"pass {sum(r['pass'] for r in rows)} / {len(rows)}")
sys.exit(0 if all(r["pass"] for r in rows) else 1)
