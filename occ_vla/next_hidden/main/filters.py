"""Safety-layer conditions of PREREG_main.md §3, built on StowSafe v5 (third_party/stowsafe_kit_v5, recorded in this repo
only as a diff: main/stowsafe_v5.patch). One interface: make_filter(cond, env, obs, task, params) -> Filter or None,
Filter.act(obs, u) -> (a, info).

  C1  all obstacles + fixed margins                      = StowSafe MODES["safety"]
  C2  observed obstacles only, task objects touchable     = MODES["named_A"] (+ SplitRolePoints "whole")
  C3  hand-only CBF + drawer part affordances             = MODES["hand_aff_fresh"] (+ AffordancePoints, refresh boxes)
  C4  PROPOSED: container part roles (GT, main/parts.py) + visibility-dependent margin (Delta)
  C5  C4 without the visibility margin (Delta = 0)
  C10 C9 + speed limits: permitted surfaces and unseen shadow are slow (capped approach), not free / kept out
  C9  C5 + affordance refinements (inner surfaces for the held object, no permission leak, phase) -- PREREG 2026-11-22
  C6  C4 without part roles (the whole task container is touchable like a target)
  C8  C4 on the TRUE geometry (oracle map: every non-robot collision surface occupied, nothing unknown)
  C6v9 / C7v9 / C8v9 / C9_5Hz: C6 / C7 / C8 / C4-5Hz rebuilt on the proposal C9 (PREREG 2026-11-22 (4))
  (C7 = C4 with roles estimated by Qwen2.5-VL-3B: main/roles_qwen.py, added separately)
C1-C3 are the main-line modes unchanged except for the tuned values (§7); with the v5 options off v5 == v4 (53 tests)."""
import dataclasses as dc
import os
import sys

import mujoco
import numpy as np

SS = "/home/ubuntu/slocal1/third_party/stowsafe_kit_v5/stowsafe"
for p in (SS, os.path.join(SS, "integration")):
    if p not in sys.path:
        sys.path.insert(0, p)
from stowsafe import StowSafe                                                          # noqa: E402
from stowsafe.config import StowSafeConfig, SafetyConfig, MapConfig, AttachConfig      # noqa: E402
from stowsafe.roles import (SplitRolePoints, AffordancePoints, articulated_sweep_boxes,  # noqa: E402
                            named_drawer, _box_surface)
from stowsafe.spheres import body_surface_points                                       # noqa: E402
from run_oft_stowsafe import MODES                                                      # noqa: E402

from parts import TaskParts                                                             # noqa: E402

INTERIOR_SITE = {"T1": "microwave_1_heating_region", "T2": "white_cabinet_1_bottom_region",
                 "T3": "wooden_cabinet_1_top_region", "T5": "desk_caddy_1_back_contain_region",
                 "T6": "wooden_cabinet_1_top_region", "T4": "wooden_cabinet_1_middle_region",
                 "T7": "basket_1_contain_region"}
CORRECTED = 0.05                     # PREREG §4: a step counts as corrected if ||a[:3] - u[:3]|| > 0.05

# C4 family: named obstacles, whole body, held object attached (as MODES["named_gt"]), the task container's sweep
# reset to unknown every step (refresh_articulated), the arm and the held object in separate fields (v5).
_C4 = dc.replace(MODES["named_gt"], map=dc.replace(MODES["named_gt"].map, refresh_articulated=True),
                 safety=dc.replace(MODES["named_gt"].safety, separate_held_field=True))


def base_config(cond, p):
    """Config of a condition. p: dict of tuned values (margin_m for C1-C3, delta_m and margin_m for C4-family)."""
    if cond == "C1":
        c = MODES["safety"]
    elif cond == "C2":
        c = MODES["named_A"]
    elif cond == "C3":
        c = MODES["hand_aff_fresh"]
    elif cond in ("C4", "C6", "C7", "C8", "C4_5Hz"):
        c = dc.replace(_C4, safety=dc.replace(_C4.safety, visibility_margin_m=p.get("delta_m", 0.02)))
    elif cond == "PROP":             # PREREG 2026-11-24 (8): C9 + Delta on unseen shadow for EVERY part (no interior exemption)
        c = dc.replace(_C4, safety=dc.replace(_C4.safety, visibility_margin_m=p.get("delta_m", 0.02)))
    elif cond in ("C5", "C9", "C6v9", "C7v9", "C8v9", "C9_5Hz", "CQB", "CQB2", "VIS", "ORACLE", "OBJ", "OBJLOOK"):
        c = _C4
    elif cond == "C10":              # PREREG 2026-11-22 (2): C9 with permitted surfaces and unseen shadow speed-limited
        c = dc.replace(_C4, safety=dc.replace(_C4.safety, slow_speed_m=float(p.get("slow_speed_m", 0.001)),
                                              slow_range_m=float(p.get("slow_range_m", 0.03))))
    else:
        raise ValueError(cond)
    # PREREG §7 (2026-11-20): gamma = 0.5 for every filter unless tuned (alpha = 10 at 20 Hz in AEGIS / multi-link)
    c = dc.replace(c, safety=dc.replace(c.safety, gamma=float(p.get("gamma", 0.5))))
    if "margin_m" in p:              # one margin scale tuned per filter (§7); arm keeps its +1 cm offset if it had one
        s = c.safety
        off = s.margin_arm_m - s.margin_palm_m
        c = dc.replace(c, safety=dc.replace(s, margin_arm_m=p["margin_m"] + off, margin_palm_m=p["margin_m"],
                                            margin_finger_m=p["margin_m"], margin_held_m=p["margin_m"]))
    if cond in ("CQB", "CQB2"):      # fences hold bystanders only (never the target): closing fingers stay constrained
        c = dc.replace(c, safety=dc.replace(c.safety, allowed=dc.replace(c.safety.allowed, fingers_free_when_closing=False)))
    if cond in ("C4_5Hz", "C9_5Hz"):
        c = dc.replace(c, map=dc.replace(c.map, update_every_steps=4))
    return c


def geom_points(m, g, spacing=0.01):
    """Points on a geom's bounding box (geom frame aabb) in its BODY frame."""
    a = m.geom_aabb[g]
    P = _box_surface(np.maximum(a[3:], 1e-3), spacing) + a[:3]
    q = np.zeros(9); mujoco.mju_quat2Mat(q, m.geom_quat[g])
    return P @ q.reshape(3, 3).T + m.geom_pos[g]


class PartPoints:
    """GT part roles of PREREG §3 as world points every step: (finger, palm, held) permissions for set_affordance."""

    def __init__(self, parts: TaskParts, m, whole_container=False):
        self.m = m
        sel = lambda kinds: [g for g, k in parts.part.items() if k in kinds]
        hand_g = list(parts.part) if whole_container else sel(("handle", "panel"))
        held_g = list(parts.part) if whole_container else sel(("handle", "panel", "floor"))
        self.hand = [(m.geom_bodyid[g], geom_points(m, g)) for g in hand_g]
        self.held = [(m.geom_bodyid[g], geom_points(m, g)) for g in held_g]
        objs = [b for b in range(m.nbody) if m.body(b).name.rsplit("_main", 1)[0] in parts.target_roots | parts.dest_roots
                and m.body(b).name.endswith("_main")]
        tgt = [b for b in objs if m.body(b).name.rsplit("_main", 1)[0] in parts.target_roots]
        self.targets = [(b, body_surface_points(m, b, include_visual=True, spacing=0.01)) for b in tgt]
        self.dests = [(b, body_surface_points(m, b, include_visual=True, spacing=0.01)) for b in objs if b not in tgt]

    @staticmethod
    def _world(items, d):
        L = [P @ d.xmat[b].reshape(3, 3).T + d.xpos[b] for b, P in items if len(P)]
        return np.concatenate(L) if L else None

    def __call__(self, d):
        hand = self._world(self.targets + self.hand, d)
        held = self._world(self.targets + self.held + self.dests, d)
        return hand, hand, held


class PartPointsV2(PartPoints):
    """C9 (PREREG 2026-11-22): C5's GT part roles with three affordance refinements, nothing else changed.
    (1) inner surfaces: the held object may touch every task-container surface inside the interior box (+2 cm),
        not only the floor (putting an object in touches the inner walls; 50% of C0's L1 contacts are of this kind).
    (2) no leak: the 1-voxel growth of a permission never covers a voxel of a surface that part may not touch
        (stowsafe perception.permission_voxels, v5).
    (3) phase: while an object is held (StowSafe's attachment), the hand loses the handle/panel permissions and
        gains the inner floor (doors and drawers are opened before grasping and closed after release here)."""

    PAD = 0.02

    def __init__(self, parts: TaskParts, m, guard, interior_site, off=()):
        super().__init__(parts, m)
        self.off = set(off)                           # ablation: subset of {"inner", "forbid", "phase"} switched off
        self.guard, self.site = guard, interior_site
        kind = lambda ks: [(m.geom_bodyid[g], geom_points(m, g)) for g, k in parts.part.items() if k in ks]
        self.cont = kind(("handle", "panel", "floor", "wall"))
        self.floor = kind(("floor",))
        self.handle_panel = kind(("handle", "panel"))
        self.walls = kind(("wall",))

    def _inside(self, P, d):
        if P is None or self.site is None:
            return None
        lo, hi = site_box(self.m, d, self.site)
        return P[np.all((P >= np.asarray(lo) - self.PAD) & (P <= np.asarray(hi) + self.PAD), axis=1)]

    def __call__(self, d):
        W = lambda items: self._world(items, d)
        cat = lambda *A: (lambda L: np.concatenate(L) if L else None)([a for a in A if a is not None and len(a)])
        inner = None if "inner" in self.off else self._inside(W(self.cont), d)
        held = cat(W(self.targets), W(self.dests), W(self.floor), inner)
        if self.guard.held.attached and "phase" not in self.off:
            hand = cat(W(self.targets), W(self.dests), W(self.floor))
        else:
            hand = cat(W(self.targets), W(self.handle_panel))
        allc = W(self.cont)

        def forbid(allowed):                          # container points this part may not touch
            if allc is None:
                return None
            if allowed is None or not len(allowed):
                return allc
            from scipy.spatial import cKDTree
            dist, _ = cKDTree(allowed).query(allc, distance_upper_bound=0.005)
            return allc[~np.isfinite(dist)]
        fh, fd = (None, None) if "forbid" in self.off else (forbid(hand), forbid(held))
        return (hand, hand, held), (fh, fh, fd)


def flip_roles(parts: TaskParts, p, stem):
    """PREREG 2026-11-22 (2) role-error test: each task-container geom's role is flipped with probability p
    (touchable handle/panel/floor -> wall, wall -> panel), seeded by the init state. Only the FILTER's copy of the
    parts changes; the collision metric builds its own TaskParts (run_closed_loop.py)."""
    import zlib
    rng = np.random.default_rng(zlib.crc32(stem.encode()))
    for g in sorted(parts.part):
        if rng.random() < p:
            parts.part[g] = "panel" if parts.part[g] == "wall" else "wall"
    parts.flipped = p


def site_box(m, d, name):
    s = m.site(name).id
    R, c, h = d.site_xmat[s].reshape(3, 3), d.site_xpos[s], m.site_size[s]
    corners = np.array([c + R @ (np.array([i, j, k]) * h) for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)])
    return corners.min(0), corners.max(0)


class QwenRoles:
    """C7 (PREREG change log 2026-11-20): container parts estimated by Qwen2.5-VL-3B on the t0 image, lifted to 3D with
    depth + calibration only. Fixed at t0 (do not follow a moving drawer / door). Target objects: true shapes (as C2-C4)."""

    def __init__(self, stem, parts: TaskParts, m):
        import json
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        R = json.load(open(os.path.join(here, "main", "qwen", "qwen_roles.json")))["results"].get(stem, {})
        self.ok = R.get("parsed") is not None
        base = PartPoints(parts, m, whole_container=not self.ok)        # targets / dests (and C6 fallback)
        self.base = base
        if not self.ok:
            return
        z = np.load(os.path.join(here, "main", "qwen", "t0", f"{stem}.npz"))
        D, K, T = z["depth"], z["K"], z["T"]

        def lift(rows, cols):
            zz = D[rows, cols]
            ok = np.isfinite(zz) & (zz < 2.5)
            rows, cols, zz = rows[ok], cols[ok], zz[ok]
            P = np.stack([(cols - K[0, 2]) / K[0, 0] * zz, (rows - K[1, 2]) / K[1, 1] * zz, zz, np.ones_like(zz)], -1)
            return (P @ T.T)[:, :3], zz

        def box_pixels(b):
            x1, y1, x2, y2 = [int(round(v)) for v in b]
            x1, x2 = sorted((max(0, x1), min(255, x2))); y1, y2 = sorted((max(0, y1), min(255, y2)))
            rr, cc = np.mgrid[y1:y2 + 1, x1:x2 + 1]
            return rr.ravel(), cc.ravel()

        j = R["parsed"]
        C, _ = lift(*box_pixels(j["container_box"]))
        F, fz = lift(*box_pixels(j["front_box"]))
        front = F[np.abs(fz - np.percentile(fz, 10)) <= 0.02] if len(F) else np.zeros((0, 3))
        handle = np.zeros((0, 3))
        if j.get("handle_point") is not None:
            x, y = [int(round(v)) for v in j["handle_point"]]
            rr, cc = np.mgrid[max(0, y - 7):min(256, y + 8), max(0, x - 7):min(256, x + 8)]
            H, _ = lift(rr.ravel(), cc.ravel())
            Hc, _ = lift(np.array([min(255, max(0, y))]), np.array([min(255, max(0, x))]))
            if len(H) and len(Hc):
                handle = H[np.linalg.norm(H - Hc[0], axis=1) <= 0.03]
        fwd = T[:3, 2]                                       # camera viewing direction (OpenCV +z) in the world
        self.hand_pts = np.concatenate([handle, front])
        self.held_pts = np.concatenate([handle, front, C])
        ext = np.concatenate([C, C + 0.15 * fwd]) if len(C) else np.zeros((0, 3))
        self.interior = (ext.min(0), ext.max(0)) if len(ext) else None
        self.meta = dict(n_handle=len(handle), n_front=len(front), n_container=len(C))

    def __call__(self, d):
        if not self.ok:
            return self.base(d)
        tg = self.base._world(self.base.targets, d)
        de = self.base._world(self.base.dests, d)
        cat = lambda *L: np.concatenate([x for x in L if x is not None and len(x)])
        hand = cat(tg, self.hand_pts)
        return hand, hand, cat(tg, de, self.held_pts)


class QwenRolesV2:
    """C7v9 (PREREG 2026-11-22 (4)): C7's estimated parts (QwenRoles, fixed at t0) with C9's three refinements.
    No floor/wall labels are estimated, so: inner = the lifted container_box points (all inside the estimated interior);
    floor = those within 2 cm of their 5th-percentile height; container = the same points. Not parsed -> C6v9."""

    def __init__(self, stem, parts: TaskParts, m, guard):
        self.q = QwenRoles(stem, parts, m)
        self.ok, self.guard = self.q.ok, guard
        if self.ok:
            C = self.q.held_pts
            self.cont = C
            self.floor = C[C[:, 2] <= np.percentile(C[:, 2], 5) + 0.02] if len(C) else C
            self.interior = self.q.interior
            self.meta = self.q.meta

    def __call__(self, d):
        b = self.q.base
        tg, de = b._world(b.targets, d), b._world(b.dests, d)
        cat = lambda *A: (lambda L: np.concatenate(L) if L else None)([a for a in A if a is not None and len(a)])
        if not self.ok:
            h, _, hd = self.q(d)
            return (h, h, hd), (None, None, None)
        held = cat(tg, de, self.cont)
        hand = cat(tg, de, self.floor) if self.guard.held.attached else cat(tg, self.q.hand_pts)

        def forbid(allowed):
            if allowed is None or not len(allowed):
                return self.cont
            from scipy.spatial import cKDTree
            dist, _ = cKDTree(allowed).query(self.cont, distance_upper_bound=0.005)
            return self.cont[~np.isfinite(dist)]
        fh = forbid(hand)
        return (hand, hand, held), (fh, fh, forbid(held))


class OracleMap:
    """C8: the map is replaced by the true geometry every update (all non-robot, non-table collision surfaces
    occupied, everything else free -> nothing unknown, no depth noise)."""

    def __init__(self, guard, m):
        self.g, self.m = guard, m
        skip = ("robot0", "gripper0", "mount0")
        self.items = []
        for b in range(1, m.nbody):
            root = m.body(m.body_rootid[b]).name
            if root.startswith(skip) or "table" in root or "floor" in root:
                continue
            P = body_surface_points(m, b, include_visual=False, spacing=0.01)
            if len(P):
                self.items.append((b, P))

    def install(self):
        per = self.g.perception
        orig_named, orig_vis = per._named_fields, per._visibility_fields

        def update(obs, rays_per_side=None, held=None, held_clear_pad=0.0):
            d = self.g.data
            mp = per.map
            P = np.concatenate([Q @ d.xmat[b].reshape(3, 3).T + d.xpos[b] for b, Q in self.items])
            P = P[mp.grid.contains(P)]
            lo = np.full(mp.grid.shape, mp.cfg.log_odds_min, np.float32)
            idx = np.unique(mp.grid.flat_index(P))
            lo.reshape(-1)[idx] = mp.cfg.log_odds_max
            mp.log_odds[...] = lo
            zmax = np.full(mp.grid.shape, -np.inf, np.float32)
            np.maximum.at(zmax.reshape(-1), mp.grid.flat_index(P), P[:, 2].astype(np.float32))
            mp.hit_zmax[...] = zmax
            robot = per.self_filter.in_world(d)
            orig_named(held, held_clear_pad, robot)
            orig_vis(held, held_clear_pad, robot)          # nothing unknown -> no visibility fields
        per.update = update


class DynamicOracleMap(OracleMap):
    """CQB (PREREG 2026-11-23 (6)): true geometry of ONLY the bystanders the consequence check predicted to move
    (fences), kept for HOLD steps after the last such prediction. Everything else is free -> no fence."""
    HOLD = 20

    def __init__(self, guard, m, bystanders):
        self.g, self.m = guard, m
        roots = {m.body_rootid[b] for b in bystanders}
        self.items_by_root = {}
        for b in range(1, m.nbody):
            if m.body_rootid[b] in roots:
                P = body_surface_points(m, b, include_visual=False, spacing=0.01)
                if len(P):
                    self.items_by_root.setdefault(m.body_rootid[b], []).append((b, P))
        self.until = {}
        self.t = 0
        self.items = []

    def set_predicted(self, moved_bodies):
        self.t += 1
        for b in moved_bodies:
            self.until[self.m.body_rootid[b]] = self.t + self.HOLD
        active = [r for r, u in self.until.items() if u >= self.t]
        self.items = [it for r in active for it in self.items_by_root.get(r, [])]
        return len(active)

    def install(self):
        per = self.g.perception
        orig_named, orig_vis = per._named_fields, per._visibility_fields

        def update(obs, rays_per_side=None, held=None, held_clear_pad=0.0):
            d = self.g.data
            mp = per.map
            lo = np.full(mp.grid.shape, mp.cfg.log_odds_min, np.float32)
            zmax = np.full(mp.grid.shape, -np.inf, np.float32)
            if self.items:
                P = np.concatenate([Q @ d.xmat[b].reshape(3, 3).T + d.xpos[b] for b, Q in self.items])
                P = P[mp.grid.contains(P)]
                if len(P):
                    idx = np.unique(mp.grid.flat_index(P))
                    lo.reshape(-1)[idx] = mp.cfg.log_odds_max
                    np.maximum.at(zmax.reshape(-1), mp.grid.flat_index(P), P[:, 2].astype(np.float32))
            mp.log_odds[...] = lo
            mp.hit_zmax[...] = zmax
            robot = per.self_filter.in_world(d)
            # true-geometry fences need none of the camera-map heuristics: do not drop the bottom table layers of an
            # object, nor voxels under the robot's / held object's spheres (pre-run check, PREREG 2026-11-23 (6))
            none = np.zeros(mp.grid.shape, bool)
            orig_inside, orig_table = per._inside, mp.table_surface
            per._inside = lambda *a, **k: none
            mp.table_surface = lambda *a, **k: none
            try:
                orig_named(held, held_clear_pad, robot)
                orig_vis(held, held_clear_pad, robot)
            finally:
                per._inside, mp.table_surface = orig_inside, orig_table
        per.update = update


class ForeignObjects:
    """OBJ (PREREG 2026-11-25 (10)): object-level permissions. Depth points inside the task container's interior box
    (+2 cm) that are > 1.0 cm from every container surface point, > 1.5 cm from the target / destination surfaces and
    outside the robot's and the held object's spheres are 'foreign'. Voxels with >= 3 foreign points at a step are
    remembered for the episode; those voxels grown by 1 voxel are removed from every part's permission (and from its
    1-voxel growth). Inputs: depth (2 cameras) + the same container / target geometry the C9 roles already use."""
    PAD, D_CONT, D_TGT, MIN_PTS, STRIDE = 0.02, 0.010, 0.015, 3, 4

    def __init__(self, parts, m, guard, interior_site):
        from scipy.spatial import cKDTree
        self.m, self.g, self.site, self.KD = m, guard, interior_site, cKDTree
        kind = lambda ks: [(m.geom_bodyid[g], geom_points(m, g)) for g, k in parts.part.items() if k in ks]
        self.cont = kind(("handle", "panel", "floor", "wall"))
        roots = set(parts.target_roots) | set(parts.dest_roots)
        self.tgt = [(b, body_surface_points(m, b, include_visual=False, spacing=0.01)) for b in range(1, m.nbody)
                    if m.body(m.body_rootid[b]).name in {r for r in roots} or
                    m.body(m.body_rootid[b]).name.replace("_main", "") in roots]
        self.vox = set()
        self.n_steps_with_new = 0

    @staticmethod
    def _world(items, d):
        L = [Q @ d.xmat[b].reshape(3, 3).T + d.xpos[b] for b, Q in items if len(Q)]
        return np.concatenate(L) if L else np.zeros((0, 3))

    def _points(self, obs, d):
        from stowsafe.libero_env import camera_views, back_project
        lo, hi = site_box(self.m, d, self.site)
        lo, hi = np.asarray(lo) - self.PAD, np.asarray(hi) + self.PAD
        out = []
        for v in camera_views(self.g.perception.env, obs):
            H, W = v.depth_m.shape
            r, c = np.meshgrid(np.arange(0, H, self.STRIDE), np.arange(0, W, self.STRIDE), indexing="ij")
            P = back_project(v, r.ravel(), c.ravel())
            out.append(P[np.all((P >= lo) & (P <= hi), axis=1)])
        return np.concatenate(out)

    def update(self, obs, d):
        P = self._points(obs, d)
        if not len(P):
            return
        C = self._world(self.cont, d)
        if len(C):
            dc, _ = self.KD(C).query(P, distance_upper_bound=self.D_CONT)
            P = P[~np.isfinite(dc)]
        T = self._world(self.tgt, d)
        if len(P) and len(T):
            dt, _ = self.KD(T).query(P, distance_upper_bound=self.D_TGT)
            P = P[~np.isfinite(dt)]
        if len(P):
            per = self.g.perception
            S = [per.self_filter.in_world(d)]
            h = self.g.held.in_world(d) if self.g.held.attached else None
            if h is not None:
                S.append(h)
            keep = np.ones(len(P), bool)
            for s in S:
                dd = np.linalg.norm(P[:, None] - s.centres[None], axis=2) - s.radii[None]
                keep &= dd.min(1) > 0
            P = P[keep]
        if len(P):
            gr = self.g.perception.map.grid
            P = P[gr.contains(P)]
            idx, cnt = np.unique(gr.flat_index(P), return_counts=True)
            new = set(idx[cnt >= self.MIN_PTS].tolist()) - self.vox
            if new:
                self.vox |= new; self.n_steps_with_new += 1

    def mask(self):
        from scipy.ndimage import binary_dilation
        gr = self.g.perception.map.grid
        M = np.zeros(gr.shape, bool)
        if self.vox:
            M.reshape(-1)[list(self.vox)] = True
            M = binary_dilation(M, iterations=1)
        return M

    def apply(self, obs, d, allow, forbid):
        self.update(obs, d)
        if not self.vox:
            return allow, forbid
        gr = self.g.perception.map.grid
        M = self.mask()
        Fc = gr.centres[M.reshape(-1)]

        def cut(P):
            if P is None or not len(P):
                return P
            ok = gr.contains(P)
            bad = np.zeros(len(P), bool)
            bad[ok] = M.reshape(-1)[gr.flat_index(P[ok])]
            return P[~bad]
        allow = tuple(cut(P) for P in allow)
        forbid = tuple(Fc if F is None else np.concatenate([F, Fc]) for F in forbid)
        return allow, forbid


class HiddenBoxOracle:
    """ORACLE (PREREG 2026-11-24 (8)): the camera map as in VIS, plus the TRUE surface of the hidden box (butter_9)
    written as occupied after every depth integration, before the obstacle fields are built (privileged)."""
    NAME = "butter_9"

    def __init__(self, guard, m):
        self.g = guard
        self.items = []
        for b in range(1, m.nbody):
            if m.body(m.body_rootid[b]).name.startswith(self.NAME):
                P = body_surface_points(m, b, include_visual=False, spacing=0.01)
                if len(P):
                    self.items.append((b, P))
        assert self.items, "no hidden box in this scene"

    def install(self):
        per = self.g.perception
        orig_named = per._named_fields

        def named(held, held_clear_pad, robot):
            d, mp = self.g.data, per.map
            P = np.concatenate([Q @ d.xmat[b].reshape(3, 3).T + d.xpos[b] for b, Q in self.items])
            P = P[mp.grid.contains(P)]
            if len(P):
                idx = mp.grid.flat_index(P)
                mp.log_odds.reshape(-1)[np.unique(idx)] = mp.cfg.log_odds_max
                np.maximum.at(mp.hit_zmax.reshape(-1), idx, P[:, 2].astype(np.float32))
            return orig_named(held, held_clear_pad, robot)
        per._named_fields = named


class OracleHeld:
    """C8: the held object is also ground truth. Held = both fingers touch a target (same rule as the metric,
    collisions.py); its spheres are fitted to the TRUE surface with StowSafe's own fit (map-based attach is off)."""

    def __init__(self, guard, parts, m, d, push=False):
        from collisions import CollisionRecorder
        self.push = push
        from stowsafe.spheres import fit_spheres_slab
        self.g, self.m, self.d, self.fit = guard, m, d, fit_spheres_slab
        self.rec = CollisionRecorder(parts, m, d)
        self.pts = {}
        for b in range(m.nbody):
            n = m.body(b).name
            if n.endswith("_main") and n[:-5] in parts.target_roots:
                P = body_surface_points(m, b, include_visual=False, spacing=0.01)
                self.pts[n[:-5]] = (b, P[np.random.default_rng(0).choice(len(P), min(len(P), 300), replace=False)])
        guard.held.update = lambda *a, **k: None          # map-based attachment off for the oracle
        self.current = None

    def _touched(self):
        """Targets in contact with any gripper body (hand or finger)."""
        m, d, rec = self.m, self.d, self.rec
        out = set()
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            for a, o in ((b1, b2), (b2, b1)):
                if rec.is_hand[a] and rec.obj_root(o) in self.pts:
                    out.add(rec.obj_root(o))
        return out

    def step(self, t):
        held = self.rec._held()
        h = self.g.held
        name = next(iter(sorted(held)), None)
        if self.push and name is None:                     # pushed, not grasped: refit every step (moves vs the palm)
            name = next(iter(sorted(self._touched())), None)
            refit = name is not None
        else:
            refit = False
        if name is None and h.attached:
            h.local_centres = h.radii = None; h.events.append((t, "detach", 0)); self.current = None
        elif name is not None and (name != self.current or refit) and name in self.pts:
            b, P = self.pts[name]
            W = P @ self.d.xmat[b].reshape(3, 3).T + self.d.xpos[b]
            R, p = self.d.xmat[h.palm].reshape(3, 3), self.d.xpos[h.palm]
            local = (W - p) @ R
            h.local_centres, h.radii = self.fit(local, h.cfg.max_spheres, pad=h.cfg.pad_m)
            if name != self.current:
                h.events.append((t, "attach_oracle", len(local)))
            self.current = name


class Filter:
    def __init__(self, cond, env, obs, task, params, instruction):
        self.cond = cond
        self.last_replan = False
        m, d = env.sim.model._model, env.sim.data._data
        self.m, self.d, self.task = m, d, task
        self.cfg = base_config(cond, params)
        self.guard = StowSafe(env, obs, self.cfg, instruction=instruction)
        ooi = tuple(env.env.obj_of_interest)
        self.split = self.aff = self.parts_pts = None
        # PROP (PREREG (8)): no interior box -> the hand also keeps Delta from unseen space inside the container
        self.interior = INTERIOR_SITE[task] if cond not in ("C1", "C2", "C3", "PROP") else None
        if cond == "C2":
            self.split = SplitRolePoints(m, ooi, "whole")
        elif cond == "C3":
            self.aff = AffordancePoints(m, ooi, instruction)
            nd = named_drawer(m, ooi, instruction)
            self.guard.set_refresh_boxes(articulated_sweep_boxes(m, d, env.env.sim, ooi, only={nd} if nd else None))
        else:                                              # C4 family (C7 overrides parts_pts with estimated roles)
            parts = TaskParts(task, m, d)
            if cond in ("C9", "C10") and params.get("role_noise", 0) > 0:
                flip_roles(parts, params["role_noise"], params["state_stem"])
            if cond in ("CQB", "CQB2"):                    # fences only on predicted-harm bystanders, no roles
                self.parts_pts = None
            elif cond in ("C9", "C10", "C8v9", "C9_5Hz", "VIS", "PROP", "ORACLE", "OBJ", "OBJLOOK"):
                self.parts_pts = PartPointsV2(parts, m, self.guard, INTERIOR_SITE[task],
                                              off=params.get("c9_off", ()))
            elif cond == "C6v9":
                self.parts_pts = PartPoints(parts, m, whole_container=True)
            elif cond == "C7v9":
                self.parts_pts = QwenRolesV2(params["state_stem"], parts, m, self.guard)
                if not self.parts_pts.ok:                  # PREREG: not parsed -> no part roles (= C6v9)
                    self.parts_pts = PartPoints(parts, m, whole_container=True)
                elif self.parts_pts.interior is not None:
                    self.interior_fixed = self.parts_pts.interior
            elif cond != "C7":
                self.parts_pts = PartPoints(parts, m, whole_container=(cond == "C6"))
            else:
                self.parts_pts = QwenRoles(params["state_stem"], parts, m)
                if self.parts_pts.ok and self.parts_pts.interior is not None:
                    self.interior_fixed = self.parts_pts.interior
            if parts.joint is not None:                    # the task container's sweep -> unknown every update
                jb = m.body(m.jnt_bodyid[parts.joint]).name
                self.guard.set_refresh_boxes(articulated_sweep_boxes(m, d, env.env.sim, (parts.container_root,),
                                                                     only={jb}))
            if cond in ("OBJ", "OBJLOOK"):                 # PREREG (10): permissions never cover a foreign object
                self.foreign = ForeignObjects(parts, m, self.guard, INTERIOR_SITE[task])
            if cond == "ORACLE":                           # PREREG (8): the hidden box's TRUE surface is always occupied
                HiddenBoxOracle(self.guard, m).install()
            if cond in ("C8", "C8v9"):
                OracleMap(self.guard, m).install()
                self.oracle_held = OracleHeld(self.guard, parts, m, d)
            if cond in ("CQB", "CQB2"):
                from collisions import CollisionRecorder
                self.dyn_map = DynamicOracleMap(self.guard, m, CollisionRecorder(parts, m, d).bystanders)
                self.dyn_map.install()
                # CQB2 (PREREG 2026-11-23 (7)): a target touched by the hand / fingers (pushed, not only grasped)
                # is treated as part of the robot, so the fences also keep it away from predicted-harm bystanders
                self.oracle_held = OracleHeld(self.guard, parts, m, d, push=(cond == "CQB2"))

    def act(self, obs, u):
        if self.split is not None:
            self.guard.set_roles(*self.split(self.d))
        if self.aff is not None:
            self.guard.set_affordance(*self.aff(self.d))
        if self.parts_pts is not None:
            if isinstance(self.parts_pts, (PartPointsV2, QwenRolesV2)):
                allow, forbid = self.parts_pts(self.d)
                if getattr(self, "foreign", None) is not None:
                    allow, forbid = self.foreign.apply(obs, self.d, allow, forbid)
                self.guard.set_affordance(*allow, forbid=forbid)
            else:
                self.guard.set_affordance(*self.parts_pts(self.d))
        if getattr(self, "interior_fixed", None) is not None:            # C7: estimated, fixed at t0
            self.guard.set_interior_boxes([self.interior_fixed])
        elif self.interior is not None and self.cond not in ("C7", "C7v9"):
            self.guard.set_interior_boxes([site_box(self.m, self.d, self.interior)])
        if getattr(self, "oracle_held", None) is not None:
            self._t = getattr(self, "_t", 0) + 1
            self.oracle_held.step(self._t)
        step = self.guard.act(obs, np.asarray(u, float))
        self.last_replan = bool(step.request_replan)
        a = np.asarray(step.action, float)
        r = step.safety
        return a, dict(corrected=bool(np.linalg.norm(a[:3] - np.asarray(u)[:3]) > CORRECTED),
                       correction=float(np.linalg.norm(a[:3] - np.asarray(u)[:3])),
                       min_clearance=None if r is None else float(r.min_clearance),
                       solved=None if r is None else bool(r.solved))

    def summary(self):
        return self.guard.summary()


def make_filter(cond, env, obs, task, params=None, instruction=""):
    if cond in ("C0", "C0p", "LOOK"):
        return None
    return Filter(cond, env, obs, task, params or {}, instruction)
