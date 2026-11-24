"""PREREG_main.md 2026-11-24 (8): closed loop for the hidden-box tasks (T3@hid, T1@hid, T4@hid, *@vis).
Same loop, policy and filters as run_closed_loop.py; only the collision recorder is extended:
  * the hidden box (butter_9) is measured in the task container's frame (the drawer body; the microwave body), so a
    drawer that carries the box does not count; it stays a bystander for the SafeLIBERO 1 mm rule (CAR / SSR);
  * first step (>= 10) at which the box has >= 1 pixel in the 128x128 segmentation of either camera (record only;
    no method reads it); first step it moved > 1 mm (container frame);
  * cause of each box movement (> 0.2 mm jump): what touched the box during that control step's physics substeps
    (robot direct / held object / another object = chain / only the container or table = shake), as diag_cqb_cause.py.
Usage: python main/run_hidden.py --cond PROP --task T3@hid --split dev --idx 0-9 --tag g1 --params '{"delta_m":0.02}'"""
import sys, os
import numpy as np
import mujoco
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_closed_loop as rcl
from collisions import CollisionRecorder
from robosuite.utils.binding_utils import MjSim

BOX = "butter_9"
RES = 128
CAMS = ("agentview", "robot0_eye_in_hand")
JUMP = 0.0002
STATE = {"rec": None}


class HiddenRecorder(CollisionRecorder):
    def __init__(self, parts, model, data):
        super().__init__(parts, model, data)
        m = model
        self.box = [b for b in self.bystanders if self.root_name[b].startswith(BOX)]
        assert self.box, "no hidden box"
        root, level = rcl_parts_container(parts.task)
        self.ref = m.body(f"{root}_cabinet_{level}").id if level else m.body(f"{root}_main").id \
            if any(m.body(i).name == f"{root}_main" for i in range(m.nbody)) else m.body(root).id
        self.box_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] in set(self.box)]
        self.r = None
        self.first_seen = None
        self.first_move = None
        self.events = []
        self.sub = set()
        self._rel_start()
        STATE["rec"] = self

    def _rel(self, b):
        d = self.d
        R = d.xmat[self.ref].reshape(3, 3)
        return R.T @ (d.xpos[b] - d.xpos[self.ref])

    def _rel_start(self):
        self.rstart = {b: self._rel(b) for b in self.box}
        self.rmax = {b: 0.0 for b in self.box}
        self.rmax_l2 = {b: 0.0 for b in self.box}

    def rebase(self):
        super().rebase()
        self._rel_start()
        self.first_seen = self.first_move = None
        self.events = []

    def _kind(self, b, held):
        if self.is_robot[b]:
            return "robot"
        if self.is_table[b]:
            return "table"
        r = self.obj_root(b)
        if r in held:
            return "held"
        if self.root_name[b].startswith(self.p.container_root):
            return "container"
        return "other_object"

    def substep(self):
        m, d = self.m, self.d
        held = self._held()
        bx = set(self.box)
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            for x, y in ((b1, b2), (b2, b1)):
                if x in bx and y not in bx:
                    self.sub.add(self._kind(y, held))

    def _seen(self):
        if self.r is None:
            self.r = mujoco.Renderer(self.m, RES, RES)
        ids = set(self.box_geoms)
        for c in CAMS:
            self.r.update_scene(self.d, camera=c)
            self.r.enable_segmentation_rendering(); s = self.r.render(); self.r.disable_segmentation_rendering()
            gid = s[..., 0][s[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)]
            if np.isin(gid, list(ids)).any():
                return True
        return False

    def step(self, t):
        super().step(t)
        for b in self.box:                       # replace the world-frame displacement by the container-frame one
            v = self._rel(b) - self.rstart[b]
            l1, l2 = float(np.abs(v).sum()), float(np.linalg.norm(v))
            if t >= 10 and l1 - self.rmax[b] > JUMP:
                self.events.append(dict(t=t, disp=round(l1, 4), jump=round(l1 - self.rmax[b], 4),
                                        touch=sorted(self.sub), seen_before=self.first_seen is not None,
                                        held=sorted(self._held())))
            self.rmax[b] = max(self.rmax[b], l1); self.rmax_l2[b] = max(self.rmax_l2[b], l2)
            self.max_disp_l1[b] = self.rmax[b]; self.max_disp[b] = self.rmax_l2[b]
            if t >= 10 and self.first_move is None and self.rmax[b] > 0.001:
                self.first_move = t
        if t >= 10 and self.first_seen is None and self._seen():
            self.first_seen = t
        self.sub = set()

    def summary(self):
        S = super().summary()
        mv = max(self.rmax.values())
        S["hidden"] = dict(box_max_disp_rel_l1_m=round(mv, 4), box_moved_1mm=mv > 0.001,
                           first_seen_step=self.first_seen, first_move_step=self.first_move,
                           moved_before_seen=(self.first_move is not None and
                                              (self.first_seen is None or self.first_move < self.first_seen)),
                           events=self.events[:200])
        return S


def rcl_parts_container(task):
    from parts import TASK_CONTAINER
    return TASK_CONTAINER[task]


_orig_step = MjSim.step


def _step(self, *a, **k):
    r = _orig_step(self, *a, **k)
    rec = STATE["rec"]
    if rec is not None and rec.d is self.data._data:
        rec.substep()
    return r


MjSim.step = _step
rcl.CollisionRecorder = HiddenRecorder

if __name__ == "__main__":
    rcl.main()
