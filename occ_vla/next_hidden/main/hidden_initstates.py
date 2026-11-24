"""PREREG_main.md 2026-11-24 (8) step 1: our own init states with ONE small object (butter_9) completely hidden at t0.
After LIBERO's reset (our seed), butter_9 is moved to ANCHOR + OFFSET (+ xy jitter), physics settles with the robot
still (DUMMY action), and a state is kept only if
  hidden_seg  0 butter pixels in BOTH cameras (element segmentation, 256x256 = what the policy pipeline renders)
  hidden_rgb  rendering the same state with butter_9 moved far away changes 0 RGB pixels in both cameras by > 8/255
              (the microwave glass is opaque in segmentation but see-through in RGB)
  stable      over the next 100 still steps no object (butter included) moves > 1 mm (SafeLIBERO sum of |axis diff|)
  no_overlap  no contact deeper than 1 mm
  closed      (drawer tasks) the container joint is where the original task starts it
Segmentation is used only here (not by any method). Usage:
  python main/hidden_initstates.py --task H1 --n 20 [--probe]   -> main/init_states_hidden/H1.npz (dev 10 / eval 40 / spare; + .json, t0 images)"""
import argparse, json, os, random, sys, hashlib
import numpy as np
import mujoco
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from libero.libero.envs import OffScreenRenderEnv

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAMS = ("agentview", "robot0_eye_in_hand")
DUMMY = [0.0] * 6 + [-1.0]
RES = 256
SEED_BASE = 980_000
OBJ = "butter_9"
# anchor: ("site", name) or ("body", name); offset in world frame (m); jitter: +- xy (m)
SPEC = {
    "H1": dict(anchor=("site", "wooden_cabinet_1_top_region"), offset=(0.0, -0.05, 0.0), jitter=0.01),
    "H5": dict(anchor=("site", "wooden_cabinet_1_middle_region"), offset=(0.0, -0.05, 0.0), jitter=0.01),
    "H2": dict(anchor=("site", "white_cabinet_1_bottom_region"), offset=(0.0, 0.05, 0.0), jitter=0.01),
    "H3": dict(anchor=("site", "microwave_1_heating_region"), offset=(0.0, 0.05, 0.0), jitter=0.01),
    "H4": dict(anchor=("body", "milk_9_main"), offset=(-0.045, 0.0, 0.0), jitter=0.005),
}


def anchor_pos(d, a):
    return (d.site(a[1]).xpos if a[0] == "site" else d.body(a[1]).xpos).copy()


def free_adr(m, name):
    j = [j for j in range(m.njnt) if m.joint(j).name.startswith(name) and m.jnt_type[j] == 0][0]
    return m.jnt_qposadr[j], m.jnt_dofadr[j]


def obj_pixels(obs, m, cam):
    ids = [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.startswith(OBJ)]
    return int(np.isin(obs[f"{cam}_segmentation_element"][..., 0], ids).sum())


def rgb_diff(env, m, d, qa):
    imgs = [env.sim.render(camera_name=c, width=RES, height=RES).astype(int) for c in CAMS]
    keep = d.qpos[qa:qa + 3].copy()
    d.qpos[qa:qa + 3] = (5.0, 5.0, -5.0); mujoco.mj_forward(m, d)
    away = [env.sim.render(camera_name=c, width=RES, height=RES).astype(int) for c in CAMS]
    d.qpos[qa:qa + 3] = keep; mujoco.mj_forward(m, d)
    return [int((np.abs(a - b).max(-1) > 8).sum()) for a, b in zip(imgs, away)]


def build(task, offset, seed, env, save_img=None):
    sp = SPEC[task]
    env.seed(seed); np.random.seed(seed); random.seed(seed)
    env.reset()
    m, d = env.sim.model._model, env.sim.data._data                 # handles AFTER reset (robosuite rebuilds)
    qa, va = free_adr(m, OBJ)
    for _ in range(10):
        env.step(DUMMY)
    rng = np.random.default_rng(seed)
    p = anchor_pos(d, sp["anchor"]) + np.array(offset) + np.array([*rng.uniform(-sp["jitter"], sp["jitter"], 2), 0.0])
    d.qpos[qa:qa + 3] = p; d.qvel[va:va + 6] = 0.0; mujoco.mj_forward(m, d)
    for _ in range(40):
        env.step(DUMMY)
    movable = [b for b in range(1, m.nbody) if m.body_rootid[b] == b and not m.body(b).name.startswith(("robot", "mount", "gripper"))
               and any(m.jnt_type[j] == 0 for j in range(m.body_jntadr[b], m.body_jntadr[b] + m.body_jntnum[b]))]
    st = env.sim.get_state().flatten().copy()
    p0 = {b: d.xpos[b].copy() for b in movable}
    obs = None
    worst, wname = 0.0, ""
    for _ in range(100):
        obs, *_ = env.step(DUMMY)
        for b in movable:
            v = float(np.abs(d.xpos[b] - p0[b]).sum())
            if v > worst:
                worst, wname = v, m.body(b).name
    # visibility is judged on the SAVED state (= t0 of the episode)
    env.sim.set_state_from_flattened(st); env.sim.forward()
    obs = env.env._get_observations(force_update=True)
    pen = float(-min([d.contact[i].dist for i in range(d.ncon)] + [0.0]))
    c = dict(seed=seed, butter_pos=d.qpos[qa:qa + 3].round(4).tolist(),
             pix_seg=[obj_pixels(obs, m, cm) for cm in CAMS], pix_rgb=rgb_diff(env, m, d, qa),
             still_max_disp_mm=round(worst * 1000, 2), worst_body=wname, deepest_pen_mm=round(pen * 1000, 2))
    c["hidden"] = c["pix_seg"] == [0, 0] and c["pix_rgb"] == [0, 0]
    c["stable"] = worst <= 0.001
    c["no_overlap"] = pen < 0.001
    c["kept"] = c["hidden"] and c["stable"] and c["no_overlap"]
    if save_img:
        for cm in CAMS:
            Image.fromarray(obs[f"{cm}_image"][::-1, ::-1]).save(f"{save_img}_{cm}.png")
    return c, st


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--task", required=True); ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--probe", default="", help="semicolon list of offsets 'dx,dy,dz' to try once each (no output file)")
    a = ap.parse_args()
    env = OffScreenRenderEnv(bddl_file_name=os.path.join(HERE, "main", "bddl_hidden", f"{a.task}.bddl"),
                             camera_heights=RES, camera_widths=RES, camera_segmentations="element", horizon=10 ** 7)
    tnum = int(a.task[1:])
    if a.probe:
        for k, o in enumerate(a.probe.split(";")):
            off = tuple(map(float, o.split(",")))
            c, _ = build(a.task, off, SEED_BASE + 1000 * tnum + 999, env,
                         save_img=f"{os.environ.get('PROBE_DIR', '/tmp')}/{a.task}_probe{k}")
            print(off, json.dumps(c), flush=True)
        return
    out = os.path.join(HERE, "main", "init_states_hidden"); os.makedirs(os.path.join(out, "t0"), exist_ok=True)
    kept, log, i = [], [], 0
    while len(kept) < a.n and i < 3 * a.n:
        seed = SEED_BASE + 1000 * tnum + i; i += 1
        c, st = build(a.task, SPEC[a.task]["offset"], seed, env,
                      save_img=os.path.join(out, "t0", f"{a.task}_seed{seed}"))
        log.append(c); print(json.dumps(c), flush=True)
        if c["kept"]:
            kept.append(st)
    path = os.path.join(out, f"{a.task}.npz")
    K = np.array(kept)                                   # split: first 10 dev, next 40 eval, rest spare (in seed order)
    np.savez_compressed(path, dev=K[:10], eval=K[10:50], spare=K[50:],
                        seeds=np.array([c["seed"] for c in log if c["kept"]]))
    json.dump(dict(task=a.task, spec=SPEC[a.task], res=RES, n_kept=len(kept), n_tried=len(log),
                   sha256=hashlib.sha256(open(path, "rb").read()).hexdigest(), log=log),
              open(path.replace(".npz", ".json"), "w"), indent=1)
    print(a.task, "kept", len(kept), "/", len(log), flush=True)


if __name__ == "__main__":
    main()
