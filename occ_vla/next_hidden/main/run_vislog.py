"""Visibility logging (PREREG_main.md 2026-11-25 (12), Kevin: "その記録を入れて進めて"). Measurement only: the policy,
filters and metric are those of run_closed_loop.py / run_hidden.py. Extra per-step record, from step 10 on:
  * pixels: how many pixels of each tracked object are seen in the 128x128 geom segmentation of each camera;
  * coverage: the share of the object's surface voxels (true surface points, 1 cm spacing, in the map's 2 cm grid)
    that the safety layer's camera map holds as occupied. For C0 (no filter) a VIS filter runs in SHADOW mode only to
    build the same camera map (its action is never executed). Filters whose map is replaced by true geometry
    (CQB / CQB2 / CQBR / ORACLE / C8*) report coverage as None.
Tracked objects: the hidden box (butter_9) if the scene has one, else every bystander root.
Result -> json["collision"]["vislog"]: per object the per-step lists and a summary (first step seen, steps seen,
max pixels, coverage when it first moved > 1 mm, max coverage).
Usage: python main/run_vislog.py --cond C0 --task T1@hid --split dev --idx 0-9 --tag vis12"""
import sys, os
import numpy as np
import mujoco
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_closed_loop as rcl
import run_hidden as RH
from collisions import CollisionRecorder
from stowsafe.spheres import body_surface_points

RES = 128
CAMS = ("agentview", "robot0_eye_in_hand")
ORACLE_MAP = ("CQB", "CQB2", "CQBR", "ORACLE", "C8", "C8v9")
S = {"env": None, "filt": None, "shadow": None, "cond": None}


class VisMixin:
    def _vis_init(self):
        m = self.m
        roots = {}
        box = [b for b in self.bystanders if self.root_name[b].startswith("butter_9")]
        for b in (box if box else self.bystanders):
            roots.setdefault(self.root_name[b], []).append(b)
        self.trk = {}
        for name, bodies in roots.items():
            geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] in set(bodies)]
            pts = [(b, body_surface_points(m, b, include_visual=False, spacing=0.01)) for b in bodies]
            self.trk[name] = dict(bodies=bodies, geoms=np.array(geoms), pts=[p for p in pts if len(p[1])],
                                  t=[], px=[], cov=[], disp=[])
        self.r = mujoco.Renderer(m, RES, RES)
        self.vstart = {n: {b: self.d.xpos[b].copy() for b in v["bodies"]} for n, v in self.trk.items()}

    def _map(self):
        f = S["shadow"] or S["filt"]
        if f is None or S["cond"] in ORACLE_MAP:
            return None
        return f.guard.perception.map

    def _vis_step(self, t):
        if t < 10:
            return
        if t == 10:
            self.vstart = {n: {b: self.d.xpos[b].copy() for b in v["bodies"]} for n, v in self.trk.items()}
        sh = S["shadow"]
        if sh is not None and S["env"] is not None:                     # C0: build the same camera map, act ignored
            obs = S["env"].env._get_observations(force_update=True)
            sh.act(obs, np.zeros(7))
        segs = []
        for c in CAMS:
            self.r.update_scene(self.d, camera=c)
            self.r.enable_segmentation_rendering(); s = self.r.render(); self.r.disable_segmentation_rendering()
            segs.append(np.where(s[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), s[..., 0], -1))
        mp = self._map()
        occ = mp.occupied if mp is not None else None
        for n, v in self.trk.items():
            v["t"].append(t)
            v["px"].append([int(np.isin(sg, v["geoms"]).sum()) for sg in segs])
            if occ is not None and v["pts"]:
                P = np.concatenate([Q @ self.d.xmat[b].reshape(3, 3).T + self.d.xpos[b] for b, Q in v["pts"]])
                P = P[mp.grid.contains(P)]
                idx = np.unique(mp.grid.flat_index(P))
                v["cov"].append(round(float(occ.reshape(-1)[idx].mean()), 3) if len(idx) else None)
            else:
                v["cov"].append(None)
            v["disp"].append(round(max(float(np.abs(self.d.xpos[b] - self.vstart[n][b]).sum()) for b in v["bodies"]), 4))

    def _vis_summary(self, S_):
        out = {}
        hidden = S_.get("hidden")
        for n, v in self.trk.items():
            seen = [i for i, p in enumerate(v["px"]) if sum(p) > 0]
            disp = v["disp"]
            if hidden is not None and n.startswith("butter_9"):          # container-frame rule of the metric
                fm = hidden.get("first_move_step")
            else:
                k = next((i for i, x in enumerate(disp) if x > 0.001), None)
                fm = v["t"][k] if k is not None else None
            ci = v["t"].index(fm) if fm in v["t"] else None
            covs = [c for c in v["cov"] if c is not None]
            out[n] = dict(first_seen=v["t"][seen[0]] if seen else None, steps_seen=len(seen),
                          max_px=max((max(p) for p in v["px"]), default=0), first_move=fm,
                          px_at_move=v["px"][ci] if ci is not None else None,
                          cov_at_move=v["cov"][ci] if ci is not None else None,
                          max_cov=max(covs) if covs else None,
                          t=v["t"], px=v["px"], cov=v["cov"])
        return out


class VisHidden(VisMixin, RH.HiddenRecorder):
    def __init__(self, *a):
        super().__init__(*a); self._vis_init()

    def step(self, t):
        super().step(t); self._vis_step(t)

    def summary(self):
        s = super().summary(); s["vislog"] = self._vis_summary(s); return s


class VisPlain(VisMixin, CollisionRecorder):
    def __init__(self, *a):
        super().__init__(*a); self._vis_init()

    def step(self, t):
        super().step(t); self._vis_step(t)

    def summary(self):
        s = super().summary(); s["vislog"] = self._vis_summary(s); return s


def factory(parts, m, d):
    has_box = any(m.body(b).name.startswith("butter_9") for b in range(m.nbody))
    return VisHidden(parts, m, d) if has_box else VisPlain(parts, m, d)


_mk = rcl.make_filter


def make_filter(cond, env, obs, task, params=None, instruction=""):
    f = _mk(cond, env, obs, task, params, instruction)
    S.update(env=env, filt=f, cond=cond, shadow=None)
    if f is None and cond in ("C0", "C0p", "LOOK"):
        S["shadow"] = _mk("VIS", env, obs, task, {"margin_m": 0.0, "gamma": 0.5}, instruction)
    return f


_branch = rcl.branch_would_collide


def branch_would_collide(*a, **k):                    # twin recorders stay plain (no rendering, no shadow map)
    rcl.CollisionRecorder = CollisionRecorder
    try:
        return _branch(*a, **k)
    finally:
        rcl.CollisionRecorder = factory


rcl.CollisionRecorder = factory
rcl.make_filter = make_filter
rcl.branch_would_collide = branch_would_collide

if __name__ == "__main__":
    rcl.main()
