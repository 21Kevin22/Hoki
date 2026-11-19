"""6-2 gate: scripted expert with privileged state (true door hinge / handle / mug / plate positions).
Checks that the task is physically solvable (open -> take out -> place) and measures what a policy faces.

Phases
  1 open   : pinch the door's vertical handle bar with the gripper pointing +y (fingers spread along x),
             then trace an arc about the hinge axis to --door-target-deg, rotating the gripper with the door.
  2 take   : the mug body (~9 cm) is wider than the Panda's 8 cm opening and there is < 2 cm above its rim
             under the ceiling, so pinch the mug HANDLE (faces the door) horizontally and slide the mug out.
  3 place  : carry over the plate (mug centre, not the grip point, over the plate centre) and lower/release.

Measured per step: door angle, mug pixels in each camera, robot contacts with the microwave other than the
handle bar and with other objects. Door opening: max angle, and the free entrance width = right inner wall x
minus the door's x-extent near the front plane. Same two quantities at the original libero_10 task's init
door angles (probe_scene6.json) for comparison.
Output: results/expert/state<k>.json, state<k>_frames.png.
"""
import argparse, json, os, sys
import numpy as np
import mujoco
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_init_states import HERE, OBJ, MW, PLATE, DOOR_JOINT, CAMS, make_env
from libero.libero.envs.predicates import eval_predicate_fn

M_PER_UNIT, RAD_PER_UNIT = 0.0127, 0.11   # OSC_POSE scale measured in StowSafe results/scale.json
OPEN, CLOSE = -1.0, 1.0
BAR, DOOR_BOX, RIGHT_WALL, LEFT_WALL, FLOOR = "microwave_1_g14", "microwave_1_g17", "microwave_1_g4", "microwave_1_g3", "microwave_1_g7"


def rotvec(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1); ang = np.arccos(c)
    if ang < 1e-6:
        return np.zeros(3)
    return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(ang)) * ang


def rot(axis, th):
    c, s = np.cos(th), np.sin(th)
    return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
            "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]


class Runner:
    def __init__(self, env, save_every):
        self.env, self.save_every = env, save_every
        self.log, self.frames, self.k = [], [], 0
        self.traj_states, self.traj_actions = [], []     # for the policy-free replays in (i)/(ii)
        m = self.m = env.sim.model._model
        self.d = env.sim.data._data
        self.robot_b = {i for i in range(m.nbody) if m.body(i).name.startswith(("robot0", "gripper0"))}
        self.mw_b = {i for i in range(m.nbody) if m.body(i).name.startswith(MW)}
        self.mug_b = {i for i in range(m.nbody) if m.body(i).name.startswith(OBJ)}
        self.other_b = {i for i in range(1, m.nbody) if i not in self.robot_b | self.mw_b | self.mug_b
                        and not m.body(m.body_rootid[i]).name.startswith(("robot0", "mount0", "gripper0", "world"))
                        and "table" not in m.body(m.body_rootid[i]).name}
        self.handle_g = {m.geom(n).id for n in ["microwave_1_g14", "microwave_1_g15", "microwave_1_g16"]}
        self.mug_g = [g for g in range(m.ngeom) if m.geom_bodyid[g] in self.mug_b]
        self.floor_g = m.geom(FLOOR).id
        j = m.joint(DOOR_JOINT); self.jq = j.qposadr[0]; self.jid = j.id
        self.grasped_mug = False
        self.debug_contacts = bool(os.environ.get("EXPERT_DEBUG"))
        self._seen = set()

    def phase_hit(self, phase):
        if phase in self._seen:
            return False
        mw, *_ = self.contacts()
        if mw:
            self._seen.add(phase)
        return bool(mw)

    def print_contacts(self, phase):
        m, d = self.m, self.d
        for i in range(d.ncon):
            c = d.contact[i]; b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if (b1 in self.robot_b and b2 in self.mw_b) or (b2 in self.robot_b and b1 in self.mw_b):
                print("DEBUG", phase, "eef", self.eef().round(3), m.geom(c.geom1).name, "|", m.geom(c.geom2).name,
                      round(c.dist * 1000, 1), flush=True)

    def door(self):
        return float(self.d.qpos[self.jq])

    def contacts(self):
        mw, other, mug = 0, 0, 0
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            b1, b2 = self.m.geom_bodyid[c.geom1], self.m.geom_bodyid[c.geom2]
            for rb, ob, og in ((b1, b2, c.geom2), (b2, b1, c.geom1)):
                if rb in self.robot_b:
                    if ob in self.mw_b and og not in self.handle_g:
                        mw += 1
                    elif ob in self.other_b:
                        other += 1
                        if self.debug_contacts and ("o", phase_name := getattr(self, "_cur", "")) not in self._seen:
                            self._seen.add(("o", phase_name))
                            print("DEBUG-OTHER", phase_name, self.m.geom(og).name, self.m.body(self.m.body_rootid[ob]).name, flush=True)
                    elif ob in self.mug_b:
                        mug += 1
            # mug (while carried) against the microwave = the object hitting the container
        # mug hitting the container: mug vs any microwave geom EXCEPT the floor box it rests / slides on (support)
        mug_mw = 0
        for i in range(self.d.ncon):
            g1, g2 = self.d.contact[i].geom1, self.d.contact[i].geom2
            bs = {self.m.geom_bodyid[g1], self.m.geom_bodyid[g2]}
            if bs & self.mug_b and bs & self.mw_b and self.floor_g not in (g1, g2):
                mug_mw += 1
        return mw, other, mug, mug_mw

    def step(self, a, phase):
        self._cur = phase
        obs, rew, done, info = self.env.step(a)
        self.traj_states.append(self.env.sim.get_state().flatten().copy())
        self.traj_actions.append(np.asarray(a, dtype=np.float32).copy())
        px = {c: int(np.isin(obs[f"{c}_segmentation_element"][..., 0], self.mug_g).sum()) for c in CAMS}
        mw, oth, mug, mug_mw = self.contacts()
        self.log.append(dict(k=self.k, phase=phase, door_deg=round(float(np.degrees(-self.door())), 1),
                             px_agent=px["agentview"], px_wrist=px["robot0_eye_in_hand"],
                             robot_microwave_contacts=mw, robot_other_contacts=oth, robot_mug_contacts=mug,
                             mug_microwave_contacts=mug_mw, done=bool(done)))
        if self.debug_contacts and self.phase_hit(phase):
            self.print_contacts(phase)
        if not hasattr(self, "_seen_through") and self.log[-1]["door_deg"] < 0.3 and (px["agentview"] or px["robot0_eye_in_hand"]):
            self._seen_through = np.concatenate([obs[f"{c}_image"][::-1, ::-1] for c in CAMS], 1)
        if self.k % self.save_every == 0:
            self.frames.append(np.concatenate([obs[f"{c}_image"][::-1, ::-1] for c in CAMS], 1))
        self.k += 1; self.obs = obs; self.done = done
        return obs

    def eef(self):
        return self.obs["robot0_eef_pos"].copy()

    def grip_R(self):
        return self.d.site("gripper0_grip_site").xmat.reshape(3, 3).copy()

    def goto(self, target, grip, phase, R, tol=0.005, max_steps=120, gain=1.0, rot_tol=0.05):
        for _ in range(max_steps):
            err = target - self.eef(); r = rotvec(R @ self.grip_R().T)
            if np.linalg.norm(err) < tol and np.linalg.norm(r) < rot_tol:
                return True
            a = np.zeros(7); a[:3] = np.clip(gain * err / M_PER_UNIT, -1, 1)
            a[3:6] = np.clip(0.5 * r / RAD_PER_UNIT, -1, 1); a[6] = grip
            self.step(a, phase)
        return False

    def hold(self, grip, n, phase):
        for _ in range(n):
            a = np.zeros(7); a[6] = grip; self.step(a, phase)


def entrance_width(m, d, door_angle_rad=None):
    """Free x-width for a straight-in approach to the opening, not blocked by the door (current door pose)."""
    g = lambda n: m.geom(n).id
    right_inner = d.geom_xpos[g(RIGHT_WALL)][0] - m.geom_size[g(RIGHT_WALL)][0]
    left_inner = d.geom_xpos[g(LEFT_WALL)][0] + m.geom_size[g(LEFT_WALL)][0]
    gd = g(DOOR_BOX); R = d.geom_xmat[gd].reshape(3, 3); s = m.geom_size[gd]
    corners = [d.geom_xpos[gd] + R @ (np.array([i, j, 0]) * s) for i in (-1, 1) for j in (-1, 1)]
    # a partly open door stands in FRONT of the opening (it swings toward the robot), so every door corner
    # counts: the straight-in approach is blocked up to the door's largest x
    xs = [c[0] for c in corners]
    blocked_to = max([left_inner] + xs)
    return round(float(max(0.0, right_inner - blocked_to)), 4), round(float(right_inner - left_inner), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", type=int, default=0)
    ap.add_argument("--door-target-deg", type=float, default=100.0)
    ap.add_argument("--save-every", type=int, default=15)
    args = ap.parse_args()
    out = os.path.join(HERE, "results", "expert"); os.makedirs(out, exist_ok=True)
    states = np.load(os.path.join(HERE, "results", "init", "init_states.npz"))["states"]
    env = make_env(); env.seed(0); env.reset(); env.set_init_state(states[args.state])
    r = Runner(env, args.save_every); m, d = r.m, r.d
    r.hold(OPEN, 10, "settle")                     # openpi's num_steps_wait
    ph = {}
    R_down = r.grip_R()
    # gripper z -> +y (toward the door); then roll 90 deg about world y so the fingers spread along x
    # (pointing +y with only the pitch, the fingers spread vertically and one pad hits the door face)
    R_fwd = rot("y", np.pi / 2) @ rot("x", np.pi / 2) @ R_down
    ph["R_fwd_z"] = R_fwd[:, 2].round(2).tolist()
    ph["finger_axis"] = R_fwd[:, 1].round(2).tolist()

    # ---- 1. open the door
    hinge = d.xanchor[r.jid].copy()
    bar0 = d.geom_xpos[m.geom(BAR).id].copy()
    ph["pre_bar"] = r.goto(bar0 + [0, -0.10, 0], OPEN, "open:pre", R_fwd, max_steps=200)
    ph["at_bar"] = r.goto(bar0 + [0, -0.005, 0], OPEN, "open:insert", R_fwd, max_steps=80)
    r.hold(CLOSE, 15, "open:pinch")
    for th in np.radians(np.linspace(3, args.door_target_deg, 40)):
        Rz = rot("z", -th)
        tgt = hinge + Rz @ (bar0 + [0, -0.005, 0] - hinge)
        r.goto(tgt, CLOSE, "open:arc", Rz @ R_fwd, tol=0.01, max_steps=8, gain=0.8, rot_tol=0.1)
    r.hold(CLOSE, 5, "open:arc")
    ph["door_deg_after_pull"] = round(float(np.degrees(-r.door())), 1)
    r.hold(OPEN, 10, "open:release")
    Rz = rot("z", r.door())                          # current door rotation (door qpos is negative)
    r.goto(r.eef() + Rz @ np.array([0, -0.08, 0]), OPEN, "open:retreat", r.grip_R(), max_steps=60)
    # the gripper is on the door's OUTER side; go around the door's free edge before heading to the mug
    edge = d.geom_xpos[m.geom(BAR).id].copy()
    r.goto(np.array([r.eef()[0], edge[1] - 0.12, r.eef()[2]]), OPEN, "open:around_door", R_fwd, max_steps=150)
    ph["door_deg_max"] = round(max(e["door_deg"] for e in r.log), 1)
    ph["door_deg_final"] = round(float(np.degrees(-r.door())), 1)
    ph["door_open_predicate"] = bool(eval_predicate_fn("open", env.env.object_states_dict[MW]))
    ph["entrance_width_m"], ph["opening_full_width_m"] = entrance_width(m, d)

    # ---- 2. take the mug out by its handle
    hg = [g for g in r.mug_g if m.geom(g).name in (f"{OBJ}_g1", f"{OBJ}_g2", f"{OBJ}_g3")]
    handle = np.mean([d.geom_xpos[g] for g in hg], axis=0)
    mug_c = d.body(f"{OBJ}_main").xpos.copy()
    ph["handle_world"] = handle.round(3).tolist(); ph["mug_world"] = mug_c.round(3).tolist()
    # cross high: at handle height the arm sweeps the other (porcelain) mug on the table
    r.goto(r.eef() + [0, 0, 0.12], OPEN, "take:up", R_fwd, max_steps=100)
    r.goto(np.array([handle[0], r.eef()[1], r.eef()[2]]), OPEN, "take:cross", R_fwd, max_steps=200)
    r.goto(np.array([handle[0], handle[1] - 0.10, r.eef()[2]]), OPEN, "take:cross", R_fwd, max_steps=200)
    ph["pre_handle"] = r.goto(handle + [0, -0.10, 0], OPEN, "take:pre", R_fwd, max_steps=200)
    ph["at_handle"] = r.goto(handle + [0, 0.005, 0], OPEN, "take:insert", R_fwd, max_steps=80, gain=0.6)
    r.hold(CLOSE, 15, "take:pinch")
    r.goto(r.eef() + [0, 0, 0.01], CLOSE, "take:lift", R_fwd, max_steps=30, gain=0.5)
    r.goto(r.eef() + [0, -0.16, 0], CLOSE, "take:slide_out", R_fwd, max_steps=150, gain=0.5)
    m_now = d.body(f"{OBJ}_main").xpos.copy()
    ph["mug_moved_m"] = round(float(np.linalg.norm(m_now - mug_c)), 3)
    ph["mug_out_of_region"] = not bool(eval_predicate_fn("in", env.env.object_states_dict[OBJ],
                                                         env.env.object_states_dict["microwave_1_heating_region"]))

    # ---- 3. place on the plate (mug centre over plate centre)
    plate = d.body(f"{PLATE}_main").xpos.copy()
    off = r.eef() - d.body(f"{OBJ}_main").xpos       # grip point relative to mug centre
    above = plate + off + [0, 0, 0.14]           # carry high: lower carries brushed the porcelain mug
    r.goto(np.array([r.eef()[0], r.eef()[1], above[2]]), CLOSE, "place:raise", R_fwd, max_steps=80, gain=0.6)
    r.goto(above, CLOSE, "place:carry", R_fwd, max_steps=200, gain=0.6)
    # (a closed-loop xy alignment on the true mug centre here was tried and made placement worse: 7/10 -> 5/10)
    r.goto(plate + off + [0, 0, 0.012], CLOSE, "place:lower", R_fwd, max_steps=160, gain=0.4)
    r.hold(OPEN, 15, "place:release")
    r.goto(r.eef() + [0, -0.08, 0.04], OPEN, "place:retreat", R_fwd, max_steps=80)
    r.hold(OPEN, 20, "place:settle")
    st = env.env.object_states_dict
    success = bool(eval_predicate_fn("on", st[OBJ], st[PLATE]))
    mug_f, plate_f = d.body(f"{OBJ}_main").xpos.copy(), d.body(f"{PLATE}_main").xpos.copy()
    ph["final_mug_plate_xy_m"] = round(float(np.linalg.norm(mug_f[:2] - plate_f[:2])), 4)   # On needs < 0.03
    ph["final_mug_above_plate_m"] = round(float(mug_f[2] - plate_f[2]), 4)
    ph["final_mug_touches_plate"] = bool(st[PLATE].check_contact(st[OBJ]))
    # entrance width at the original libero_10 task's init door angles (76 / 97.5 / 118 deg) for comparison
    q_save = d.qpos[r.jq]
    ph["entrance_width_at_deg"] = {}
    for deg in (30, 45, 60, 76.0, 97.5, 118.0):
        d.qpos[r.jq] = -np.radians(deg); mujoco.mj_kinematics(m, d)
        ph["entrance_width_at_deg"][str(deg)] = entrance_width(m, d)[0]
    d.qpos[r.jq] = q_save; mujoco.mj_kinematics(m, d)

    L = r.log
    first = lambda key: next((e["k"] for e in L if e[key] > 0), None)
    summary = dict(state=args.state, steps=len(L), success=success, env_done=bool(r.done), phases=ph,
                   first_visible_step=dict(agentview=first("px_agent"), wrist=first("px_wrist")),
                   robot_microwave_contact_steps={p: sum(e["robot_microwave_contacts"] > 0 for e in L if e["phase"] == p)
                                                  for p in dict.fromkeys(e["phase"] for e in L)},
                   mug_microwave_contact_steps_while_taking=sum(e["mug_microwave_contacts"] > 0 for e in L
                                                                if e["phase"].startswith("take:slide")),
                   robot_other_object_contact_steps=sum(e["robot_other_contacts"] > 0 for e in L), render="egl")
    json.dump(dict(summary=summary, log=L), open(os.path.join(out, f"state{args.state}.json"), "w"), indent=1)
    # MuJoCo state after every step (replays are exact by set_state; re-executing actions is not, see RESULTS)
    np.savez_compressed(os.path.join(out, f"state{args.state}_traj.npz"), states=np.array(r.traj_states),
                        actions=np.array(r.traj_actions), phases=np.array([e["phase"] for e in L]),
                        door_deg=np.array([e["door_deg"] for e in L]))
    if hasattr(r, "_seen_through"):
        Image.fromarray(r._seen_through).save(os.path.join(out, f"state{args.state}_visible_door_closed.png"))
    sel = r.frames[:: max(1, len(r.frames) // 8)][:8]
    Image.fromarray(np.concatenate(sel, 0)).save(os.path.join(out, f"state{args.state}_frames.png"))
    print(json.dumps(summary), flush=True)
    env.close()


if __name__ == "__main__":
    main()
