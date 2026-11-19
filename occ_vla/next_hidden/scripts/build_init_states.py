"""6-1: build init states with the mug inside the CLOSED microwave (KITCHEN_SCENE6, plate added).

LIBERO's sampler cannot put an object inside a closed microwave, so reset states are post-processed:
reset (door open, mug on table) -> teleport the mug to the microwave's heating-region centre (+ xy jitter,
dropped from just above the region floor) -> settle with the door open -> set the door joint to closed ->
settle with zero actions. A state is kept only if every check passes:

  closed      LIBERO's Close(microwave_1) predicate (door qpos > -0.005 rad)
  inside      LIBERO's In(white_yellow_mug_1, microwave_1_heating_region)
  hidden      0 mug pixels in the third-person and wrist cameras (element segmentation)
  stable      mug moves < 2 mm over the last 40 settle steps
  no_overlap  no contact penetrates deeper than 1 mm
  rgb_hidden  0 mug pixels in RGB at 256x256 in both cameras (mug vs mug-transparent render, diff > 10):
              the door glass (alpha 0.25) is opaque in segmentation but not in RGB
  door_shut   door < 0.3 deg open and the mug does not touch the door (else it can pop the door ajar)
  goal_false  the task goal (On mug plate) is false at start

Render: EGL (same backend for every stage). Pin a GPU for rendering only, e.g. CUDA_VISIBLE_DEVICES=2.
Writes results/init/{init_states.npz, init_checks.json, images/}.
"""
import argparse, json, os, time
import numpy as np
import mujoco
from PIL import Image
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.envs.predicates import eval_predicate_fn

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BDDL = os.path.join(HERE, "bddl",
                    "KITCHEN_SCENE6_take_the_yellow_and_white_mug_out_of_the_microwave_and_put_it_on_the_plate.bddl")
CAMS = ["agentview", "robot0_eye_in_hand"]
OBJ, MW, REGION, PLATE = "white_yellow_mug_1", "microwave_1", "microwave_1_heating_region", "plate_1"
DOOR_JOINT = "microwave_1_microjoint"
DOOR_CLOSED = -0.001   # range [-2.094, 0]; LIBERO Close = qpos > -0.005


def object_pixels(env, obs, cam, obj):
    m = env.sim.model._model
    ids = [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.startswith(obj)]
    return int(np.isin(obs[f"{cam}_segmentation_element"][..., 0], ids).sum())


def deepest_penetration_mm(sim):
    d = sim.data._data
    return float(-min([d.contact[i].dist for i in range(d.ncon)] + [0.0]) * 1000)


def zero_steps(env, n):
    obs = None
    for _ in range(n):
        obs, *_ = env.step(np.zeros(7))
    return obs


OPAQUE_MATERIALS = ("microwave_1_micro_window",)   # door glass: alpha 0.25 in LIBERO's microwave.xml


class OpaqueGlassEnv(OffScreenRenderEnv):
    """The task's env: LIBERO's OffScreenRenderEnv with the microwave door glass made opaque (alpha 1).

    Why (PREREG_next.md, 2026-11-19): with alpha 0.25 the mug inside the closed microwave is visible in RGB through
    the door (wrist camera while the arm moves; 1-18 px in the third-person camera at some init states), which
    breaks the premise "the object is inside a closed container and cannot be seen". Rendering only: the glass is a
    visual mesh (contype 0); the door's collision box is untouched. robosuite rebuilds the MuJoCo model on every
    reset(), so the patch is re-applied after each reset and the reset observation is re-rendered.
    """

    def _patch(self):
        m = self.env.sim.model._model
        for name in OPAQUE_MATERIALS:
            m.mat_rgba[m.material(name).id, 3] = 1.0

    def reset(self):
        super().reset()
        self._patch()
        return self.env._get_observations(force_update=True)


def make_env(res=128, depth=False):
    return OpaqueGlassEnv(bddl_file_name=BDDL, camera_heights=res, camera_widths=res,
                          camera_segmentations="element", camera_names=CAMS, camera_depths=depth)


def check_state(env, obs, p0, p1):
    """Return the check dict for the current state (shared with verify_init_states.py)."""
    d = env.sim.data._data
    m = env.sim.model._model
    st = env.env.object_states_dict
    c = dict(
        door_qpos=round(float(d.qpos[m.joint(DOOR_JOINT).qposadr[0]]), 4),
        closed=bool(eval_predicate_fn("close", st[MW])),
        inside=bool(eval_predicate_fn("in", st[OBJ], st[REGION])),
        goal_false=not bool(eval_predicate_fn("on", st[OBJ], st[PLATE])),
        pixels_agentview=object_pixels(env, obs, "agentview", OBJ),
        pixels_wrist=object_pixels(env, obs, "robot0_eye_in_hand", OBJ),
        drift_mm=round(float(np.linalg.norm(p1 - p0) * 1000), 2),
        deepest_penetration_mm=round(deepest_penetration_mm(env.sim), 2),
        object_pos=p1.round(4).tolist(),
    )
    c["hidden"] = c["pixels_agentview"] == 0 and c["pixels_wrist"] == 0
    c["stable"] = c["drift_mm"] < 2.0
    c["no_overlap"] = c["deepest_penetration_mm"] < 1.0
    door_b = m.body("microwave_1_microdoorroot").id
    mug_b = {i for i in range(m.nbody) if m.body(i).name.startswith(OBJ)}
    c["mug_touches_door"] = any({m.geom_bodyid[d.contact[i].geom1], m.geom_bodyid[d.contact[i].geom2]} & mug_b
                                and door_b in {m.geom_bodyid[d.contact[i].geom1], m.geom_bodyid[d.contact[i].geom2]}
                                for i in range(d.ncon))
    c["door_deg"] = round(float(np.degrees(-c["door_qpos"])), 3)
    c["door_shut"] = c["door_deg"] < 0.3 and not c["mug_touches_door"]
    c["pass"] = all(c[k] for k in ["closed", "inside", "goal_false", "hidden", "stable", "no_overlap", "door_shut"])
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max-tries", type=int, default=40)
    ap.add_argument("--x-offset", type=float, default=-0.011, help="from the interior x-centre [m]")
    ap.add_argument("--y-offset", type=float, default=0.0, help="from the interior y-centre (+ = deeper) [m]")
    ap.add_argument("--x-jitter", type=float, default=0.008)
    ap.add_argument("--y-jitter", type=float, default=0.006)
    ap.add_argument("--reloads", type=int, default=3, help="reload round-trips that must all pass (first one with a cleared solver warm-start)")
    args = ap.parse_args()
    out = os.path.join(HERE, "results", "init")
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    env = make_env()
    from rgb_visibility import make_rgb_env, rgb_mug_pixels
    rgb_env = make_rgb_env()
    kept, checks = [], []
    for seed in range(args.max_tries):
        if len(kept) >= args.n:
            break
        t0 = time.time()
        rng = np.random.default_rng(seed)
        env.seed(seed)
        env.reset()
        m, d = env.sim.model._model, env.sim.data._data      # take handles AFTER reset
        jq, jv = m.joint(DOOR_JOINT).qposadr[0], m.joint(DOOR_JOINT).dofadr[0]
        free = m.joint(f"{OBJ}_joint0")
        qa, va = free.qposadr[0], free.dofadr[0]
        zero_steps(env, 10)
        # Interior from the microwave's own collision boxes: floor g7 (top face), side walls g3/g4, back g5.
        g = lambda n: m.geom(n).id
        floor_top = d.geom_xpos[g("microwave_1_g7")][2] + m.geom_size[g("microwave_1_g7")][2]
        x_c = 0.5 * ((d.geom_xpos[g("microwave_1_g3")][0] + m.geom_size[g("microwave_1_g3")][0])
                     + (d.geom_xpos[g("microwave_1_g4")][0] - m.geom_size[g("microwave_1_g4")][0]))
        # Placement window (see README): too far forward the mug handle (facing the door) touches the door and pops
        # it ~2 deg ajar; too deep / too far right the ~20 cm Panda hand hits the side wall when reaching the handle.
        y_c = d.geom_xpos[g("microwave_1_g6")][1] + args.y_offset
        x_c = x_c + args.x_offset
        mug_geoms = [i for i in range(m.ngeom) if m.body(m.geom_bodyid[i]).name.startswith(OBJ)]
        bottom_offset = d.qpos[qa + 2] - min(d.geom_xpos[i][2] - m.geom_rbound[i] * 0 - m.geom_size[i][2]
                                             for i in mug_geoms)     # upright mug: body z - lowest box face
        d.qpos[qa:qa + 3] = [x_c + rng.uniform(-args.x_jitter, args.x_jitter),
                             y_c + rng.uniform(-args.y_jitter, args.y_jitter), floor_top + bottom_offset + 0.003]
        d.qvel[va:va + 6] = 0.0
        mujoco.mj_forward(m, d)
        zero_steps(env, 40)
        d.qpos[jq] = DOOR_CLOSED
        d.qvel[jv] = 0.0
        mujoco.mj_forward(m, d)
        zero_steps(env, 160)                                   # long settle: 40 left ~3.5 mm residual drift
        p0 = d.qpos[qa:qa + 3].copy()
        obs = zero_steps(env, 40)
        p1 = d.qpos[qa:qa + 3].copy()
        c = dict(seed=seed, **check_state(env, obs, p0, p1))
        state = env.sim.get_state().flatten().copy()
        if c["pass"]:                     # reload exactly as a rollout would; reloads are not bit-identical
            c["reload_drift_mm"] = []
            for rs in range(args.reloads):
                env.seed(1000 + rs); env.reset(); env.set_init_state(state)
                m, d = env.sim.model._model, env.sim.data._data
                if rs == 0:   # solver warm-start left from a previous episode changes whether a marginal mug slides
                    d.qacc_warmstart[:] = 0; mujoco.mj_forward(m, d)
                zero_steps(env, 10); q0 = d.qpos[qa:qa + 3].copy(); obs = zero_steps(env, 40)
                rr = check_state(env, obs, q0, d.qpos[qa:qa + 3].copy())
                c["reload_drift_mm"].append(rr["drift_mm"])
                c["pass"] = c["pass"] and rr["pass"]
        if c["pass"]:
            c.update(rgb_mug_pixels(rgb_env, state))
            c["rgb_hidden"] = c["rgb_mug_pixels_agentview"] == 0 and c["rgb_mug_pixels_robot0_eye_in_hand"] == 0
            c["pass"] = c["rgb_hidden"]
        c["seconds"] = round(time.time() - t0, 1)
        checks.append(c)
        print(json.dumps(c), flush=True)
        if c["pass"]:
            kept.append(state)
            k = len(kept) - 1
            for cam in CAMS:
                Image.fromarray(obs[f"{cam}_image"][::-1, ::-1]).save(os.path.join(out, "images", f"state{k:02d}_{cam}.png"))
    np.savez_compressed(os.path.join(out, "init_states.npz"), states=np.array(kept))
    json.dump(dict(bddl=os.path.basename(BDDL), kept=len(kept), tried=len(checks), render="egl", checks=checks),
              open(os.path.join(out, "init_checks.json"), "w"), indent=1)
    print(f"kept {len(kept)} / tried {len(checks)}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
