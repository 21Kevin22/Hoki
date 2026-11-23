"""Occlusion diagnostic (PREREG_main.md 2026-11-23): were the surfaces the robot hit ever visible to a camera?
Same closed loop as run_closed_loop.py (same args); only the collision recorder is extended. Every step from step 10 on,
the agentview and wrist cameras are rendered at 128x128 (geom segmentation + depth) and all body poses are kept.
For the FIRST L1 contact of each (robot side, contacted geom) pair, the contact point is fixed in the contacted body's
frame and projected into both cameras at every kept step: 'visible' at a step = some pixel of the 3x3 window around the
projection shows the same geom AND its depth is within 1.5 cm of the point's depth.
Category: (a) visible at the contact step, (b) visible before but not at the contact step, (c) never visible.
Result -> json["collision"]["vis"]. Usage: python main/diag_visibility.py --cond C0 --task T1 --split dev --idx 0-9 --tag vis"""
import sys, os
import numpy as np
import mujoco
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_closed_loop as rcl
from collisions import CollisionRecorder

RES, WIN, TOL = 128, 1, 0.015
CAMS = ("agentview", "robot0_eye_in_hand")


class VisRecorder(CollisionRecorder):
    def __init__(self, parts, model, data):
        super().__init__(parts, model, data)
        self.r = mujoco.Renderer(model, RES, RES)
        self.cam_ids = [model.camera(c).id for c in CAMS]
        self.hist = []            # per kept step: (t, seg[2], depth[2], cam_pos[2], cam_mat[2], xpos, xmat)
        self.first = {}           # (side, geom) -> dict(t, body, local, bystander)
        self.fovy = [float(model.cam_fovy[c]) for c in self.cam_ids]

    def _render(self, t):
        m, d = self.m, self.d
        seg, dep = [], []
        for c in CAMS:
            self.r.update_scene(d, camera=c)
            self.r.enable_segmentation_rendering(); s = self.r.render().copy(); self.r.disable_segmentation_rendering()
            gid = np.where(s[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), s[..., 0], -1).astype(np.int32)
            self.r.enable_depth_rendering(); z = self.r.render().copy(); self.r.disable_depth_rendering()
            seg.append(gid); dep.append(z.astype(np.float32))
        self.hist.append((t, seg, dep, [d.cam_xpos[c].copy() for c in self.cam_ids],
                          [d.cam_xmat[c].reshape(3, 3).copy() for c in self.cam_ids], d.xpos.copy(), d.xmat.copy()))

    def _visible(self, h, g, b, local):
        t, seg, dep, cp, cm, xpos, xmat = h
        p = xmat[b].reshape(3, 3) @ local + xpos[b]
        for k in range(2):
            pc = cm[k].T @ (p - cp[k])
            if pc[2] >= -1e-3:
                continue
            f = (RES / 2) / np.tan(np.deg2rad(self.fovy[k]) / 2)
            u = RES / 2 + f * pc[0] / -pc[2]; v = RES / 2 - f * pc[1] / -pc[2]
            ui, vi = int(np.floor(u)), int(np.floor(v))
            if not (0 <= ui < RES and 0 <= vi < RES):
                continue
            ys, xs = slice(max(0, vi - WIN), vi + WIN + 1), slice(max(0, ui - WIN), ui + WIN + 1)
            ok = (seg[k][ys, xs] == g) & (np.abs(dep[k][ys, xs] - (-pc[2])) <= TOL)
            if ok.any():
                return True
        return False

    def step(self, t):
        super().step(t)
        if t < 10:
            return
        self._render(t)
        m, d = self.m, self.d
        held = self._held()
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            b1, b2 = m.geom_bodyid[g1], m.geom_bodyid[g2]
            for (ga, ba), (go, bo) in (((g1, b1), (g2, b2)), ((g2, b2), (g1, b1))):
                if self.is_robot[ba] and not self.is_robot[bo]:
                    side = "finger" if self.is_finger[ba] else ("hand" if self.is_hand[ba] else "arm")
                elif self.obj_root(ba) in held and not self.is_robot[bo] and self.obj_root(bo) != self.obj_root(ba):
                    side = "held"
                else:
                    continue
                if not self._allowed(side, go, bo, held) and (side, int(go)) not in self.first:
                    local = d.xmat[bo].reshape(3, 3).T @ (np.asarray(c.pos) - d.xpos[bo])
                    self.first[(side, int(go))] = dict(t=t, body=int(bo), local=local, bystander=bool(bo in self.bystanders),
                                                       name=m.geom(go).name or m.body(bo).name, n_hist=len(self.hist))
                break

    def _proj(self, h, k, p):
        t, seg, dep, cp, cm, xpos, xmat = h
        pc = cm[k].T @ (p - cp[k])
        f = (RES / 2) / np.tan(np.deg2rad(self.fovy[k]) / 2)
        return RES / 2 + f * pc[0] / -pc[2], RES / 2 - f * pc[1] / -pc[2], -pc[2], f

    def sanity(self):
        """Method check on the first kept step, agentview: (1) the target body's centre projects within 3 px of a
        pixel segmented as that body (independent of the depth test: checks the image orientation / projection);
        (2) round trip: target pixels unprojected with the rendered depth are 'visible' by _visible."""
        m, h = self.m, self.hist[0]
        t, seg, dep, cp, cm, xpos, xmat = h
        out = {}
        for b in range(m.nbody):
            n = m.body(b).name
            if not (n.endswith("_main") and n[:-5] in self.p.target_roots):
                continue
            gs = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b]
            u, v, z, f = self._proj(h, 0, xpos[b])
            ui, vi = int(u), int(v)
            win = seg[0][max(0, vi - 3):vi + 4, max(0, ui - 3):ui + 4]
            centre_ok = bool(np.isin(win, gs).any())
            ys, xs = np.nonzero(np.isin(seg[0], gs))
            rt = []
            for j in np.linspace(0, len(ys) - 1, min(20, len(ys))).astype(int) if len(ys) else []:
                y, x = ys[j], xs[j]; zz = float(dep[0][y, x]); g = int(seg[0][y, x])
                pc = np.array([(x + 0.5 - RES / 2) * zz / f, -(y + 0.5 - RES / 2) * zz / f, -zz])
                pw = cp[0] + cm[0] @ pc
                bb = m.geom_bodyid[g]
                rt.append(self._visible(h, g, bb, xmat[bb].reshape(3, 3).T @ (pw - xpos[bb])))
            out[n] = dict(centre_ok=centre_ok, roundtrip=float(np.mean(rt)) if rt else None, n_pix=int(len(ys)))
        return out

    def summary(self):
        S = super().summary()
        ev = []
        for (side, g), e in sorted(self.first.items(), key=lambda x: x[1]["t"]):
            H = self.hist[:e["n_hist"]]
            now = self._visible(H[-1], g, e["body"], e["local"]) if H else False
            before = any(self._visible(h, g, e["body"], e["local"]) for h in H[:-1])
            cat = "a_visible_now" if now else ("b_seen_before" if before else "c_never_seen")
            ev.append(dict(side=side, geom=e["name"], t=e["t"], bystander=e["bystander"], cat=cat))
        S["vis"] = dict(events=ev, sanity=self.sanity() if self.hist else {})
        getattr(self.r, "close", lambda: None)()
        self.hist = []
        return S


rcl.CollisionRecorder = VisRecorder

if __name__ == "__main__":
    rcl.main()
