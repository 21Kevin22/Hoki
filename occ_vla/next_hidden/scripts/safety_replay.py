"""(ii) Safety-layer false alarms at the narrow entrance: replay the expert's recorded states (exact, set_state),
give the safety layer the observation and the expert's command u at every step, record its corrected command
(NOT executed: the motion is the recording). Correction = ||u_hat[:3]-u[:3]|| > 0.05. 'Would collide' = the
recording has a robot contact with the microwave (except the handle bar) or a non-target object in t+1..t+5.
Classes: justified / over-conservative (correction, no collision) / miss (no correction, collision).
Versions = existing main-line modes, unchanged (PREREG_next.md (ii)). Writes results/safety_replay/<mode>.json.
Usage: python scripts/safety_replay.py --mode safety|named_A [--states 0-9]"""
import argparse, collections, json, os, sys
import numpy as np
SS = "/home/ubuntu/slocal1/third_party/stowsafe_kit_v4/stowsafe"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, SS); sys.path.insert(0, os.path.join(SS, "integration"))
from build_init_states import HERE, make_env
from stowsafe import StowSafe
from stowsafe.roles import SplitRolePoints
from run_oft_stowsafe import MODES, SPLIT_ROLES

CORR, HORIZON = 0.05, 5
INSTRUCTION = "take the yellow and white mug out of the microwave and put it on the plate"
NARROW = ("take:pre", "take:insert", "take:pinch", "take:lift", "take:slide_out")


def run(k, mode, env):
    z = np.load(os.path.join(HERE, "results", "expert", f"state{k}_traj.npz"))
    log = json.load(open(os.path.join(HERE, "results", "expert", f"state{k}.json")))["log"]
    states, actions, phases = z["states"], z["actions"], z["phases"]
    collide = np.array([(e["robot_microwave_contacts"] > 0) or (e["robot_other_contacts"] > 0) for e in log])
    env.sim.set_state_from_flattened(states[0]); env.sim.forward()
    obs = env.env._get_observations(force_update=True)
    cfg = MODES[mode]
    guard = StowSafe(env, obs, cfg, instruction=INSTRUCTION)
    m, d = env.sim.model._model, env.sim.data._data
    split = SplitRolePoints(m, tuple(env.env.obj_of_interest), SPLIT_ROLES[mode]) if mode in SPLIT_ROLES else None
    rows = []
    for t in range(1, len(states)):
        env.sim.set_state_from_flattened(states[t - 1]); env.sim.forward()
        obs = env.env._get_observations(force_update=True)
        if split is not None:
            guard.set_roles(*split(d))
        u = actions[t].astype(float)
        uh = guard.act(obs, u).action
        corr = float(np.linalg.norm(uh[:3] - u[:3]))
        will = bool(collide[t:t + HORIZON].any())
        rows.append(dict(t=t, phase=str(phases[t]), correction=round(corr, 3), corrected=corr > CORR, would_collide=will))
    return rows, guard.summary()


def classify(rows):
    c = collections.Counter()
    for r in rows:
        c[("justified" if r["would_collide"] else "over_conservative") if r["corrected"]
          else ("miss" if r["would_collide"] else "quiet")] += 1
    return dict(c)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", required=True); ap.add_argument("--states", default="0-9")
    a = ap.parse_args(); lo, hi = map(int, a.states.split("-"))
    out = os.path.join(HERE, "results", "safety_replay"); os.makedirs(out, exist_ok=True)
    env = make_env(res=256, depth=True); env.reset()
    per, allrows = [], []
    for k in range(lo, hi + 1):
        rows, summ = run(k, a.mode, env)
        by_phase = {p: classify([r for r in rows if r["phase"] == p]) for p in dict.fromkeys(r["phase"] for r in rows)}
        e = dict(state=k, total=classify(rows), narrow=classify([r for r in rows if r["phase"] in NARROW]),
                 by_phase=by_phase, unsolved_qp=summ.get("unsolved_qp"), map_ms=summ.get("map_ms"), qp_ms=summ.get("qp_ms"))
        per.append(e); allrows += [dict(state=k, **r) for r in rows]
        print(json.dumps({x: e[x] for x in ("state", "total", "narrow", "unsolved_qp")}), flush=True)
    tot = collections.Counter(); nar = collections.Counter()
    for e in per:
        tot.update(e["total"]); nar.update(e["narrow"])
    nc = nar["justified"] + nar["over_conservative"]
    verdict = dict(mode=a.mode, total=dict(tot), narrow=dict(nar),
                   narrow_over_conservative_share=round(nar["over_conservative"] / nc, 3) if nc else None,
                   narrow_misses=nar["miss"])
    json.dump(dict(verdict=verdict, per_state=per, rows=allrows), open(os.path.join(out, f"{a.mode}.json"), "w"), indent=1)
    print(json.dumps(verdict))


if __name__ == "__main__":
    main()
