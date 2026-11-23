"""Closed-loop episodes with a frozen policy (pi0.5 via openpi's websocket server) on our own init states.
Client logic = openpi/examples/libero/main.py (256 render, 180-deg rotation, resize_with_pad 224, 10 wait steps,
replan every 5, per-suite max steps). Every step: collision recorder (PREREG_main.md §4) + per-step log for the failure
taxonomy. Filters (C1..C8) plug in through `make_filter` (week 1: only C0 = none).
Outputs per episode: main/runs/<cond>/<task>_<split><idx>.json (+ .npz per-step arrays, + .mp4 for failures).
Usage: python main/run_closed_loop.py --cond C0 --task T1 --split dev --idx 0-9 [--port 8000] [--tag rep1]"""
import argparse, json, math, os, pathlib, sys, time, collections
import numpy as np
import torch
_tl = torch.load
torch.load = lambda *a, **k: _tl(*a, **{**k, "weights_only": False})
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/ubuntu/slocal1/third_party/stowsafe_kit_v5/stowsafe")   # v5 (== v4 with options off)
import imageio
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _wcp
from libero.libero.envs import OffScreenRenderEnv
from tasks import TASKS, MAX_STEPS, bddl_path
from parts import TaskParts
from collisions import CollisionRecorder
from filters import make_filter

HERE = pathlib.Path(__file__).resolve().parents[1]
DUMMY = [0.0] * 6 + [-1.0]


def quat2axisangle(q):
    q = q.copy(); q[3] = np.clip(q[3], -1, 1)
    den = np.sqrt(1.0 - q[3] * q[3])
    return np.zeros(3) if math.isclose(den, 0.0) else (q[:3] * 2.0 * math.acos(q[3])) / den


BRANCH_STEPS = 5                     # PREREG §4: branch counterfactual horizon


def branch_would_collide(twin, task, state, raw_actions):
    """Copy the sim state into the twin env, execute the UNCORRECTED commands for up to 5 steps, report an L1 contact."""
    twin.env.timestep = 0; twin.env.done = False          # the twin never resets: keep it below its horizon
    twin.sim.set_state_from_flattened(state); twin.sim.forward()
    m, d = twin.sim.model._model, twin.sim.data._data
    rec = CollisionRecorder(TaskParts(task, m, d), m, d)
    for k, u in enumerate(raw_actions[:BRANCH_STEPS]):
        twin.step(np.asarray(u, float).tolist()); rec.step(k)
        if rec.steps[-1]["l1"]:
            return True
    return False


def run_episode(env, client, task, state, cond, out_stem, replan=5, params=None, twin=None):
    suite, _, lang = TASKS[task]
    env.reset()
    obs = env.set_init_state(state)
    m, d = env.sim.model._model, env.sim.data._data
    parts = TaskParts(task, m, d)
    rec = CollisionRecorder(parts, m, d)
    filt = None
    branches = collections.Counter()
    plan, frames, log = collections.deque(), [], collections.defaultdict(list)
    t, done, n_inf, inf_s, t0 = 0, False, 0, 0.0, time.time()
    max_t = MAX_STEPS[suite] + 10
    while t < max_t:
        if t < 10:
            obs, _, done, _ = env.step(DUMMY); rec.step(t); t += 1
            if t == 10:
                rec.rebase()                 # displacement from the settled pose (objects drop ~7 cm at reset)
                filt = make_filter(cond, env, obs, task, params, lang)   # built on the settled scene (main-line order)
            continue
        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]), 224, 224))
        wimg = image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]), 224, 224))
        frames.append(img)
        if not plan:
            el = {"observation/image": img, "observation/wrist_image": wimg,
                  "observation/state": np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])),
                  "prompt": lang}
            ti = time.time(); chunk = client.infer(el)["actions"]; inf_s += time.time() - ti; n_inf += 1
            plan.extend(chunk[:replan])
        u = np.asarray(plan.popleft(), float)
        info = {}
        if filt is None:
            a = u
        else:
            a, info = filt.act(obs, u)
            if info["corrected"] and twin is not None:      # justified vs over-conservative, in the closed loop
                hit = branch_would_collide(twin, task, env.sim.get_state().flatten().copy(), [u] + list(plan))
                branches["justified" if hit else "over_conservative"] += 1
        log["u"].append(u); log["a"].append(a); log["eef"].append(obs["robot0_eef_pos"].copy())
        log["corrected"].append(bool(info.get("corrected", False)))
        log["grip_q"].append(obs["robot0_gripper_qpos"].copy())
        obs, _, done, _ = env.step(a.tolist())
        rec.step(t)
        if filt is not None and filt.last_replan:           # StowSafe asked to re-query the policy (main-line rule)
            plan.clear()
        t += 1
        if done:
            break
    wall = time.time() - t0
    col = rec.summary()
    corr = np.array(log["corrected"], bool)
    l1 = np.array([s["l1"] for s in rec.steps[10:]], bool)[:len(corr)]
    miss = int(sum(l1[k] and not corr[max(0, k - BRANCH_STEPS):k + 1].any() for k in range(len(l1)))) if filt else None
    res = dict(task=task, cond=cond, params=params or {}, success=bool(done), steps=t, max_steps=max_t, wall_s=round(wall, 1),
               corrected_steps=int(corr.sum()), branch=dict(branches), miss_steps=miss,
               stalled=stalled(np.array(log["u"]), np.array(log["eef"])),
               filter=None if filt is None else {k: v for k, v in filt.summary().items()
                                                  if k in ("map_ms", "qp_ms", "unsolved_qp", "replans", "mean_correction",
                                                           "attach_events")},
               blockers=None if filt is None else [list(map(str, b)) for b in filt.guard.stats.blockers[:30]],
               n_infer=n_inf, mean_infer_s=round(inf_s / max(n_inf, 1), 3), collision=col,
               cf_success=bool(done) and not col["l1"], render=os.environ.get("MUJOCO_GL"),
               held_steps=sum(bool(s["held"]) for s in rec.steps))
    np.savez_compressed(f"{out_stem}.npz", u=np.array(log["u"]), a=np.array(log["a"]), eef=np.array(log["eef"]),
                        grip_q=np.array(log["grip_q"]), l1=np.array([s["l1"] for s in rec.steps]),
                        fmax=np.array([s["fmax"] for s in rec.steps]),
                        held=np.array([bool(s["held"]) for s in rec.steps]))
    if not res["cf_success"]:
        imageio.mimwrite(f"{out_stem}.mp4", frames, fps=10)
    json.dump(dict(**res, l1_pairs_first=[s["pairs"] for s in rec.steps if s["l1"]][:20]), open(f"{out_stem}.json", "w"), indent=1)
    return res


def stalled(u, eef, window=40, ratio=0.2):
    """PREREG §4 failure type 'stalled': for 40 consecutive steps the hand moves < 20 % of what was commanded."""
    if len(u) < window + 1:
        return False
    cmd = np.linalg.norm(u[1:, :3], axis=1) * 0.0127
    moved = np.linalg.norm(np.diff(eef, axis=0), axis=1)
    bad = (cmd > 1e-3) & (moved < ratio * cmd)
    run = 0
    for b in bad:
        run = run + 1 if b else 0
        if run >= window:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", default="C0"); ap.add_argument("--task", required=True)
    ap.add_argument("--split", default="dev"); ap.add_argument("--idx", default="0-9")
    ap.add_argument("--port", type=int, default=8000); ap.add_argument("--tag", default="")
    ap.add_argument("--params", default="{}", help='tuned values as JSON, e.g. {"delta_m": 0.02}')
    ap.add_argument("--no-branch", action="store_true", help="skip the branch counterfactual (faster)")
    a = ap.parse_args()
    lo, hi = map(int, a.idx.split("-"))
    states = np.load(HERE / "main" / ("init_states_diag" if a.split == "diag" else "init_states") / f"{a.task}.npz")[a.split]
    bddl, lang = bddl_path(*TASKS[a.task][:2])
    # depth on for EVERY condition (StowSafe needs it; the RGB the policy sees is unchanged) -> same env for all
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256, camera_depths=True,
                             camera_names=["agentview", "robot0_eye_in_hand"],
                             horizon=MAX_STEPS[TASKS[a.task][0]] + 50)    # never truncated before our own limit
    env.seed(7)
    params = json.loads(a.params)
    no_branch = a.no_branch or os.environ.get("NO_BRANCH") == "1"      # tuning runs skip the counterfactual
    twin = None if (a.cond in ("C0", "C0p") or no_branch) else OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=8, camera_widths=8, horizon=10**7)
    if twin is not None:
        twin.seed(7); twin.reset()
    client = _wcp.WebsocketClientPolicy("127.0.0.1", a.port)
    out = HERE / "main" / "runs" / (a.cond + (f"_{a.tag}" if a.tag else "")); out.mkdir(parents=True, exist_ok=True)
    for i in range(lo, hi + 1):
        if (HERE / "results" / "STOP_NEW_EPISODES").exists():
            print("STOP_NEW_EPISODES present; stopping", flush=True); break
        stem = out / f"{a.task}_{a.split}{i:02d}"
        if pathlib.Path(f"{stem}.json").exists():
            continue                                     # resume: never re-run a finished episode
        p_i = dict(params, state_stem=f"{a.task}_{a.split}{i:02d}") if a.cond in ("C7", "C9", "C10", "C7v9") else params
        r = run_episode(env, client, a.task, states[i], a.cond, str(stem), params=p_i, twin=twin)
        (HERE / "results" / "last_episode_done").touch()
        print(json.dumps({k: r[k] for k in ("task", "cond", "success", "cf_success", "steps", "wall_s")} |
                         {"l1": r["collision"]["l1"], "moved_1mm": r["collision"]["moved_1mm"]}), flush=True)
    env.close()


if __name__ == "__main__":
    main()
