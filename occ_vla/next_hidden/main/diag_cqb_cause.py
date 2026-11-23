"""CQB failure-cause diagnostic (PREREG_main.md 2026-11-23 (7)): why did bystanders still move under CQB?
Same closed loop as run_closed_loop.py (same args); only the recorder is extended. At every step where some bystander's
SafeLIBERO displacement (sum of |per-axis diff| from the settled pose) increases by > 0.2 mm, the recorder logs what
touches that bystander (by type: arm / hand / finger / held = the object in the gripper / target (not held) / dest /
container / bystander / table), whether that bystander had an active CBF fence, whether something was held, and the
QP slack of that step. Result -> json["collision"]["cause"]. Not a condition: the policy, filter and metric are CQB's.
Usage: python main/diag_cqb_cause.py --cond CQB --task T2@k2 --split diag --idx 0-19 --tag cause --params '{...}'"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_closed_loop as rcl
from collisions import CollisionRecorder

JUMP = 0.0002
STATE = {"filt": None, "sub": {}}      # sub: bystander root -> kinds touching it at ANY physics substep


class CauseRecorder(CollisionRecorder):
    def __init__(self, parts, model, data):
        super().__init__(parts, model, data)
        self.prev = {b: 0.0 for b in self.bystanders}
        self.events = []
        self.root_id = {b: model.body_rootid[b] for b in range(model.nbody)}
        STATE["rec"] = self

    def rebase(self):
        super().rebase()
        self.prev = {b: 0.0 for b in self.bystanders}

    def _kind(self, b, held):
        if self.is_robot[b]:
            return "finger" if self.is_finger[b] else ("hand" if self.is_hand[b] else "arm")
        if self.is_table[b]:
            return "table"
        r = self.obj_root(b)
        if r in held:
            return "held"
        if r in self.p.target_roots:
            return "target"
        if r in self.p.dest_roots:
            return "dest"
        if self.root_name[b].startswith(self.p.container_root):
            return "container"
        return "bystander"

    def step(self, t):
        super().step(t)
        m, d = self.m, self.d
        held = self._held()
        f = STATE["filt"]
        dm = getattr(f, "dyn_map", None)
        for b in self.bystanders:
            v = float(np.abs(d.xpos[b] - self.start[b]).sum())
            if v - self.prev[b] > JUMP:
                rb = self.root_id[b]
                touch = set(STATE["sub"].get(rb, ()))            # contacts during the control step's substeps
                for i in range(d.ncon):
                    c = d.contact[i]
                    b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
                    for x, y in ((b1, b2), (b2, b1)):
                        if self.root_id[x] == rb and self.root_id[y] != rb:
                            touch.add(self._kind(y, held))
                fenced = None if dm is None else bool(dm.until.get(rb, -1) >= dm.t)
                sl = f.guard.stats.slack[-1] if (f is not None and f.guard.stats.slack) else None
                self.events.append(dict(t=t, body=m.body(b).name, disp=round(v, 4), jump=round(v - self.prev[b], 4),
                                        touch=sorted(touch), fenced=fenced, held=sorted(held),
                                        slack=None if sl is None else round(float(sl), 4)))
            self.prev[b] = max(self.prev[b], v)
        STATE["sub"] = {}

    def summary(self):
        S = super().summary()
        S["cause"] = self.events[:400]
        return S


rcl.CollisionRecorder = CauseRecorder
_mk = rcl.make_filter


def make_filter(*a, **k):
    STATE["filt"] = _mk(*a, **k)
    env = a[1]
    rec = STATE["rec"]
    sim = env.sim
    orig = sim.step

    def step(*x, **y):                  # robosuite runs several physics substeps per control step: log every contact
        r = orig(*x, **y)
        m, d = rec.m, rec.d
        held = rec._held()
        roots = {rec.root_id[b] for b in rec.bystanders}
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            for xx, yy in ((b1, b2), (b2, b1)):
                rx, ry = rec.root_id[xx], rec.root_id[yy]
                if rx in roots and ry != rx:
                    STATE["sub"].setdefault(rx, set()).add(rec._kind(yy, held))
        return r
    sim.step = step
    return STATE["filt"]


rcl.make_filter = make_filter

if __name__ == "__main__":
    rcl.main()
