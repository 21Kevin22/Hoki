"""6-4: run pi0.5 (pi05_libero) on official LIBERO tasks from its training suites, ONE episode at a time.
Client logic is a copy of openpi/examples/libero/main.py (256 render, 180-deg rotation, resize_with_pad 224,
num_steps_wait 10, replan_steps 5, per-suite max_steps, seed 7). Adds: per-episode wall time, per-episode JSON,
results/last_episode_done for the monitor, and refuses to start an episode if results/STOP_NEW_EPISODES exists.
The policy server runs separately (scripts/serve_pi05.sh) on its own GPU.
Usage: python scripts/pi05_official_smoke.py --suite libero_spatial --task-id 0 --episode 0"""
import argparse, collections, json, math, os, pathlib, time
import imageio
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _wcp

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DUMMY = [0.0] * 6 + [-1.0]
MAX_STEPS = dict(libero_spatial=220, libero_object=280, libero_goal=300, libero_10=520, libero_90=400)


def quat2axisangle(quat):
    quat = quat.copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial"); ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--episode", type=int, default=0); ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=7); ap.add_argument("--replan-steps", type=int, default=5)
    args = ap.parse_args()
    out = pathlib.Path(HERE, "results", "pi05_smoke"); out.mkdir(parents=True, exist_ok=True)
    if pathlib.Path(HERE, "results", "STOP_NEW_EPISODES").exists():
        raise SystemExit("STOP_NEW_EPISODES present (monitor saw >95% memory); not starting")
    np.random.seed(args.seed)
    suite = benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(args.task_id)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    env.seed(args.seed)
    client = _wcp.WebsocketClientPolicy("127.0.0.1", args.port)
    env.reset()
    obs = env.set_init_state(suite.get_task_init_states(args.task_id)[args.episode])
    plan, t, done, frames, n_infer, infer_s = collections.deque(), 0, False, [], 0, 0.0
    t0 = time.time()
    while t < MAX_STEPS[args.suite] + 10:
        if t < 10:
            obs, _, done, _ = env.step(DUMMY); t += 1; continue
        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]), 224, 224))
        wimg = image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]), 224, 224))
        frames.append(img)
        if not plan:
            el = {"observation/image": img, "observation/wrist_image": wimg,
                  "observation/state": np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])),
                  "prompt": str(task.language)}
            ti = time.time(); chunk = client.infer(el)["actions"]; infer_s += time.time() - ti; n_infer += 1
            plan.extend(chunk[: args.replan_steps])
        obs, _, done, _ = env.step(plan.popleft().tolist())
        if done:
            break
        t += 1
    wall = time.time() - t0
    name = f"{args.suite}_task{args.task_id}_ep{args.episode}"
    imageio.mimwrite(out / f"{name}_{'success' if done else 'failure'}.mp4", [np.asarray(x) for x in frames], fps=10)
    rec = dict(suite=args.suite, task_id=args.task_id, task=task.language, episode=args.episode, success=bool(done),
               steps=t, max_steps=MAX_STEPS[args.suite], wall_s=round(wall, 1), n_infer=n_infer,
               mean_infer_s=round(infer_s / max(n_infer, 1), 3), seed=args.seed, render=os.environ.get("MUJOCO_GL"))
    json.dump(rec, open(out / f"{name}.json", "w"), indent=1)
    pathlib.Path(HERE, "results", "last_episode_done").touch()
    print(json.dumps(rec), flush=True)
    env.close()


if __name__ == "__main__":
    main()
