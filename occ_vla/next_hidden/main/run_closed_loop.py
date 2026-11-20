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
sys.path.insert(0, "/home/ubuntu/slocal1/third_party/stowsafe_kit_v4/stowsafe")
import imageio
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _wcp
from libero.libero.envs import OffScreenRenderEnv
from tasks import TASKS, MAX_STEPS, bddl_path
from parts import TaskParts
from collisions import CollisionRecorder

HERE = pathlib.Path(__file__).resolve().parents[1]
DUMMY = [0.0] * 6 + [-1.0]


def quat2axisangle(q):
    q = q.copy(); q[3] = np.clip(q[3], -1, 1)
    den = np.sqrt(1.0 - q[3] * q[3])
    return np.zeros(3) if math.isclose(den, 0.0) else (q[:3] * 2.0 * math.acos(q[3])) / den


def make_filter(cond, env, obs, task):
    if cond in ("C0", "C0p"):
        return None
    raise NotImplementedError(f"condition {cond} is implemented later in week 1")


def run_episode(env, client, task, state, cond, out_stem, replan=5):
    suite, _, lang = TASKS[task]
    env.reset()
    obs = env.set_init_state(state)
    m, d = env.sim.model._model, env.sim.data._data
    parts = TaskParts(task, m, d)
    rec = CollisionRecorder(parts, m, d)
    filt = make_filter(cond, env, obs, task)
    plan, frames, log = collections.deque(), [], collections.defaultdict(list)
    t, done, n_inf, inf_s, t0 = 0, False, 0, 0.0, time.time()
    max_t = MAX_STEPS[suite] + 10
    while t < max_t:
        if t < 10:
            obs, _, done, _ = env.step(DUMMY); rec.step(t); t += 1; continue
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
        a = u if filt is None else filt(obs, u)
        log["u"].append(u); log["a"].append(a); log["eef"].append(obs["robot0_eef_pos"].copy())
        log["grip_q"].append(obs["robot0_gripper_qpos"].copy())
        obs, _, done, _ = env.step(a.tolist())
        rec.step(t)
        t += 1
        if done:
            break
    wall = time.time() - t0
    col = rec.summary()
    res = dict(task=task, cond=cond, success=bool(done), steps=t, max_steps=max_t, wall_s=round(wall, 1),
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", default="C0"); ap.add_argument("--task", required=True)
    ap.add_argument("--split", default="dev"); ap.add_argument("--idx", default="0-9")
    ap.add_argument("--port", type=int, default=8000); ap.add_argument("--tag", default="")
    a = ap.parse_args()
    lo, hi = map(int, a.idx.split("-"))
    states = np.load(HERE / "main" / "init_states" / f"{a.task}.npz")[a.split]
    bddl, lang = bddl_path(*TASKS[a.task][:2])
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    env.seed(7)
    client = _wcp.WebsocketClientPolicy("127.0.0.1", a.port)
    out = HERE / "main" / "runs" / (a.cond + (f"_{a.tag}" if a.tag else "")); out.mkdir(parents=True, exist_ok=True)
    for i in range(lo, hi + 1):
        if (HERE / "results" / "STOP_NEW_EPISODES").exists():
            print("STOP_NEW_EPISODES present; stopping", flush=True); break
        stem = out / f"{a.task}_{a.split}{i:02d}"
        if pathlib.Path(f"{stem}.json").exists():
            continue                                     # resume: never re-run a finished episode
        r = run_episode(env, client, a.task, states[i], a.cond, str(stem))
        (HERE / "results" / "last_episode_done").touch()
        print(json.dumps({k: r[k] for k in ("task", "cond", "success", "cf_success", "steps", "wall_s")} |
                         {"l1": r["collision"]["l1"], "moved_1mm": r["collision"]["moved_1mm"]}), flush=True)
    env.close()


if __name__ == "__main__":
    main()
