"""Failure-cause study (PREREG_main.md 2026-11-23 (2)): are occlusion and collision causes of long-horizon failures?
Same closed loop as run_closed_loop.py (frozen pi0.5, no safety layer); conditions are interventions on the scene:
  B1 / B2 : unchanged (B2 = A/A repeat)
  Occ     : obstacles made INVISIBLE (removed from both camera images), physics unchanged
  Col     : obstacles made NON-COLLIDING with the robot / targets / destinations / container (still on the table,
            still colliding with the table and each other), appearance unchanged
  Both    : both
Obstacle = every body that is not the robot, the table/floor/room, a target, a destination or the task container.
Per step (from step 10) the recorder logs, for each target / destination / the container, the share of its surface points
inside each camera's view whose camera ray first hits another body (occlusion ratio, by occluder type), the share of
points actually visible, L1/L2 contacts and bystander / container displacement -> json["collision"]["attrib"].
Usage: python main/diag_attrib.py --cond Occ --task T8 --split diag --idx 0-19 --tag attrib"""
import sys, os
import numpy as np
import mujoco
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_closed_loop as rcl
import filters
from collisions import CollisionRecorder
from stowsafe.spheres import body_surface_points

CONDS = {"B1": None, "B2": None, "Occ": "occ", "Col": "col", "Both": "both"}
CAMS = ("agentview", "robot0_eye_in_hand")
RES = 256                       # policy renders at 256 (openpi main.py)
NPTS = 200
OBST_BIT = 2


def obstacle_bodies(m, parts):
    rec = CollisionRecorder.__new__(CollisionRecorder)
    root = lambda b: m.body(m.body_rootid[b]).name
    out = []
    for b in range(1, m.nbody):
        r, n = root(b), m.body(b).name
        if n.startswith(("robot0", "gripper0", "mount0")) or r.startswith(("robot0", "mount0")):
            continue
        if "table" in r or "floor" in r or "wall" in r:
            continue
        o = r.rsplit("_main", 1)[0]
        if o in parts.target_roots | parts.dest_roots or r.startswith(parts.container_root):
            continue
        out.append(b)
    return out


def apply_intervention(env, task, cond):
    kind = CONDS[cond]
    if kind is None:
        return []
    m = env.sim.model._model
    from parts import TaskParts
    parts = TaskParts(task, m, env.sim.data._data)
    obs_b = set(obstacle_bodies(m, parts))
    gs = [g for g in range(m.ngeom) if m.geom_bodyid[g] in obs_b]
    if kind in ("occ", "both"):
        for g in gs:
            m.geom_rgba[g, 3] = 0.0
            m.geom_matid[g] = -1
    if kind in ("col", "both"):
        root = lambda b: m.body(m.body_rootid[b]).name
        for g in gs:
            if m.geom_contype[g] or m.geom_conaffinity[g]:
                m.geom_contype[g] = OBST_BIT; m.geom_conaffinity[g] = OBST_BIT
        for g in range(m.ngeom):                              # table / floor / room keep obstacles from falling
            b = m.geom_bodyid[g]
            if (b == 0 or "table" in root(b) or "floor" in root(b)) and (m.geom_contype[g] or m.geom_conaffinity[g]):
                m.geom_contype[g] |= OBST_BIT; m.geom_conaffinity[g] |= OBST_BIT
    mujoco.mj_forward(m, env.sim.data._data)
    return sorted({m.body(m.body_rootid[b]).name for b in obs_b})


class AttribRecorder(CollisionRecorder):
    def __init__(self, parts, model, data):
        super().__init__(parts, model, data)
        m = model
        rng = np.random.default_rng(0)
        self.objs = []                                       # (label, kind, [(body, local pts)])
        names = {m.body(b).name: b for b in range(m.nbody)}
        for kind, roots in (("target", sorted(parts.target_roots)), ("dest", sorted(parts.dest_roots))):
            for r in roots:
                b = names.get(r + "_main", names.get(r))
                if b is not None:
                    self.objs.append((r, kind, [(b, body_surface_points(m, b, include_visual=True, spacing=0.01))]))
        cb = [b for b in range(m.nbody) if self.root_name[b].startswith(parts.container_root)]
        self.objs.append((parts.container_root, "container",
                          [(b, body_surface_points(m, b, include_visual=True, spacing=0.01)) for b in cb]))
        for i, (lab, kind, items) in enumerate(self.objs):     # subsample NPTS points per object, fixed
            sizes = [len(P) for _, P in items]
            tot = sum(sizes)
            if tot == 0:
                continue
            pick = set(rng.choice(tot, min(NPTS, tot), replace=False).tolist())
            new, off = [], 0
            for b, P in items:
                idx = [k - off for k in range(off, off + len(P)) if k in pick]
                new.append((b, P[idx])); off += len(P)
            self.objs[i] = (lab, kind, new)
        self.obj_bodies = [{b for b, _ in items} for _, _, items in self.objs]
        self.cam_ids = [m.camera(c).id for c in CAMS]
        self.log = {"occ_ag": [], "occby_ag": [], "vis_ag": [], "vis_wr": [], "occ_wr": [], "l2": [], "byst": [],
                    "cont": [], "held": []}
        self.gid = np.zeros(1, np.int32)

    def _hit_type(self, b, own):
        if b in own:
            return "self"
        if self.is_robot[b]:
            return "robot"
        if self.is_table[b]:
            return "table"
        r = self.obj_root(b)
        if r in self.p.target_roots or r in self.p.dest_roots:
            return "task"
        if self.root_name[b].startswith(self.p.container_root):
            return "container"
        return "obstacle"

    def _cam(self, k, P, own):
        m, d = self.m, self.d
        c = self.cam_ids[k]
        cp, cm = d.cam_xpos[c], d.cam_xmat[c].reshape(3, 3)
        f = (RES / 2) / np.tan(np.deg2rad(m.cam_fovy[c]) / 2)
        n_in = n_vis = 0
        occ = {}
        for p in P:
            pc = cm.T @ (p - cp)
            if pc[2] >= -1e-3:
                continue
            u = RES / 2 + f * pc[0] / -pc[2]; v = RES / 2 - f * pc[1] / -pc[2]
            if not (0 <= u < RES and 0 <= v < RES):
                continue
            n_in += 1
            vec = p - cp; L = float(np.linalg.norm(vec))
            dist = mujoco.mj_ray(m, d, cp, vec / L, None, 1, -1, self.gid)
            if dist < 0 or dist >= L - 0.005:
                n_vis += 1
                continue
            t = self._hit_type(int(m.geom_bodyid[self.gid[0]]), own)
            if t == "self":
                if dist >= L - 0.02:
                    n_vis += 1
                continue
            occ[t] = occ.get(t, 0) + 1
        return n_in, n_vis, occ

    def step(self, t):
        super().step(t)
        if t < 10:
            return
        d = self.d
        oa, ob, va, vw, ow = [], [], [], [], []
        for (lab, kind, items), own in zip(self.objs, self.obj_bodies):
            P = np.concatenate([Q @ d.xmat[b].reshape(3, 3).T + d.xpos[b] for b, Q in items if len(Q)]) \
                if any(len(Q) for _, Q in items) else np.zeros((0, 3))
            n = max(len(P), 1)
            ni, nv, occ = self._cam(0, P, own)
            oa.append(round(sum(occ.values()) / ni, 3) if ni else 0.0)
            ob.append({k: round(v / ni, 3) for k, v in occ.items()} if ni else {})
            va.append(round(nv / n, 3))
            ni2, nv2, occ2 = self._cam(1, P, own)
            vw.append(round(nv2 / n, 3)); ow.append(round(sum(occ2.values()) / ni2, 3) if ni2 else 0.0)
        s = self.steps[-1]
        self.log["occ_ag"].append(oa); self.log["occby_ag"].append(ob); self.log["vis_ag"].append(va)
        self.log["vis_wr"].append(vw); self.log["occ_wr"].append(ow)
        self.log["l2"].append(bool(s["l1"] and s["fmax"] > 5.0))
        self.log["byst"].append(round(max([float(np.linalg.norm(d.xpos[b] - self.start[b])) for b in self.bystanders] + [0.0]), 4))
        self.log["cont"].append(round(max(float(np.linalg.norm(d.xpos[self.container_body] - self.container_start))
                                          if self.container_body is not None else 0.0, s["joint_move"]), 4))
        self.log["held"].append(s["held"])

    def summary(self):
        S = super().summary()
        S["attrib"] = dict(objects=[(lab, kind) for lab, kind, _ in self.objs], **self.log)
        return S


rcl.CollisionRecorder = AttribRecorder
_make_filter = rcl.make_filter
rcl.make_filter = lambda cond, *a, **k: None if cond in CONDS else _make_filter(cond, *a, **k)
_run_episode = rcl.run_episode


def run_episode(env, client, task, state, cond, out_stem, **kw):
    orig = env.set_init_state
    info = {}

    def sis(s):
        o = orig(s)
        info["obstacles"] = apply_intervention(env, task, cond)
        return o
    env.set_init_state = sis
    try:
        r = _run_episode(env, client, task, state, cond, out_stem, **kw)
    finally:
        del env.set_init_state
    import json
    j = json.load(open(f"{out_stem}.json")); j["intervention"] = dict(cond=cond, obstacles=info.get("obstacles"))
    json.dump(j, open(f"{out_stem}.json", "w"), indent=1)
    return r


rcl.run_episode = run_episode

if __name__ == "__main__":
    rcl.main()
