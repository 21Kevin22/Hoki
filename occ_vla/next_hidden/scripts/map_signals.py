"""(i) Map-signal reliability: replay the expert's recorded MuJoCo states (exact, by set_state) and derive, from
depth only (+ calibrated cameras, the robot's own pixels removed, the container's box B given ONCE at t0),
door angle / opened fraction / entrance width / 'new blob inside' signals; compare with ground truth.
Definitions and pass criteria: PREREG_next.md (2026-11-19, (i)). Writes results/map_signals/state<k>.json + summary.json.
Usage: python scripts/map_signals.py [--states 0-9]"""
import argparse, json, os, sys
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/ubuntu/slocal1/third_party/stowsafe_kit_v4/stowsafe")
from build_init_states import HERE, OBJ, MW, make_env
from scripted_expert import entrance_width, DOOR_BOX
from stowsafe.libero_env import camera_views, back_project

OPEN_DEG, OPEN_HOLD = 74.5, 3
SWING_FRONT, SWING_BACK, SWING_X, SWING_ZLO = 0.30, 0.01, 0.05, 0.04
MIN_DOOR_PTS, MIN_DOOR_SPAN = 30, 0.05
REVEAL_M = 0.03
IN_FRONT, FLOOR_LO, FLOOR_HI, WALL_IN, MIN_BLOB, MIN_INSIDE = 0.02, 0.03, 0.13, 0.02, 20, 50


def body_ids(m, prefixes):
    return {i for i in range(m.nbody) if m.body(i).name.startswith(prefixes)}


def geom_ids(m, bodies):
    return np.array([g for g in range(m.ngeom) if m.geom_bodyid[g] in bodies])


EDGE_PX = 2   # segmentation and depth disagree by 1-2 px at object edges (edge pixels carry the depth behind)


def cloud(env, obs, exclude_geoms, only_geoms=None):
    """World points of both cameras (OpenCV orientation, see stowsafe.libero_env), minus excluded geoms.
    The excluded (robot) mask is grown and the kept-only mask shrunk by EDGE_PX to drop mixed edge pixels."""
    P, views = [], camera_views(env, obs)
    for v in views:
        seg = obs[f"{v.name}_segmentation_element"][..., 0][::-1]
        ok = np.isfinite(v.depth_m) & (v.depth_m < 2.5) & ~binary_dilation(np.isin(seg, exclude_geoms), iterations=EDGE_PX)
        if only_geoms is not None:
            ok &= binary_erosion(np.isin(seg, only_geoms), iterations=EDGE_PX)
        r, c = np.nonzero(ok)
        P.append(back_project(v, r, c))
    return (np.concatenate(P) if P else np.zeros((0, 3))), views


def door_angle(P, B):
    lo, hi = B
    yf = lo[1]
    sel = ((P[:, 1] > yf - SWING_FRONT) & (P[:, 1] < yf - SWING_BACK) & (P[:, 0] > lo[0] - SWING_X) &
           (P[:, 0] < hi[0] + SWING_X) & (P[:, 2] > lo[2] + SWING_ZLO) & (P[:, 2] < hi[2]))
    D = P[sel]
    if len(D) < MIN_DOOR_PTS:
        return 0.0, D, len(D)
    xy = D[:, :2]; mu = xy.mean(0)
    _, _, vt = np.linalg.svd(xy - mu, full_matrices=False)
    d = vt[0]; s = (xy - mu) @ d
    if s.max() - s.min() < MIN_DOOR_SPAN:
        return 0.0, D, len(D)
    a, b = mu + s.min() * d, mu + s.max() * d
    hinge, far = (a, b) if a[1] > b[1] else (b, a)                 # hinge end = closer to the front plane
    v = far - hinge
    side = np.sign(0.5 * (lo[0] + hi[0]) - hinge[0]) or 1.0
    return float(np.degrees(np.arctan2(-v[1], side * v[0]))), D, len(D)


def true_door_max_x(m, d):
    """x of the door's outermost point (collision box corners), for a like-for-like width ground truth."""
    g = m.geom(DOOR_BOX).id; R = d.geom_xmat[g].reshape(3, 3); s_ = m.geom_size[g]
    return max((d.geom_xpos[g] + R @ (np.array([i, j, 0]) * s_))[0] for i in (-1, 1) for j in (-1, 1))


def blob(P, B):
    lo, hi = B
    I = P[(P[:, 0] > lo[0]) & (P[:, 0] < hi[0]) & (P[:, 1] > lo[1] + IN_FRONT) & (P[:, 1] < hi[1]) &
          (P[:, 2] > lo[2]) & (P[:, 2] < hi[2])]
    if len(I) < MIN_INSIDE:
        return False, len(I), 0
    floor = np.percentile(I[:, 2], 5)
    xl, xr = np.percentile(I[:, 0], 2), np.percentile(I[:, 0], 98)
    yb = np.percentile(I[:, 1], 98)
    S = I[(I[:, 2] > floor + FLOOR_LO) & (I[:, 2] < floor + FLOOR_HI) & (I[:, 0] > xl + WALL_IN) &
          (I[:, 0] < xr - WALL_IN) & (I[:, 1] < yb - WALL_IN)]
    return len(S) >= MIN_BLOB, len(I), len(S)


def run(k, env):
    z = np.load(os.path.join(HERE, "results", "expert", f"state{k}_traj.npz"))
    states, door_gt = z["states"], z["door_deg"]
    init = np.load(os.path.join(HERE, "results", "init", "init_states.npz"))["states"][k]
    env.reset(); env.set_init_state(init)                         # t0 = the state before the expert's first step
    m, d = env.sim.model._model, env.sim.data._data
    robot_g = geom_ids(m, body_ids(m, ("robot0", "gripper0", "mount0")))
    mw_g = geom_ids(m, body_ids(m, (MW,)))
    mug_g = geom_ids(m, body_ids(m, (OBJ,)))
    obs = env.env._get_observations(force_update=True)
    Pm, views0 = cloud(env, obs, robot_g, only_geoms=mw_g)        # container box B, given once
    B = (Pm.min(0), Pm.max(0)); lo, hi = B
    v0 = views0[0]
    r0, c0 = np.nonzero(np.isfinite(v0.depth_m))
    X0 = back_project(v0, r0, c0)
    inB = np.all((X0 >= lo) & (X0 <= hi), 1)
    base = (r0[inB], c0[inB], v0.depth_m[r0[inB], c0[inB]])
    rows = []
    for t, s in enumerate(states):
        env.sim.set_state_from_flattened(s); env.sim.forward()
        obs = env.env._get_observations(force_update=True)
        P, views = cloud(env, obs, robot_g)
        th, D, n_door = door_angle(P, B)
        # width, same formula for estimate and ground truth (box B's right edge minus the door's largest x);
        # None when the door is open but too few door points are visible (not 0)
        w_hat = float(max(0.0, hi[0] - max(lo[0], D[:, 0].max()))) if th > 0 else (0.0 if door_gt[t] <= 5 else None)
        w_gt = float(max(0.0, hi[0] - max(lo[0], true_door_max_x(m, d))))
        v = views[0]
        seg = obs["agentview_segmentation_element"][..., 0][::-1]
        rr, cc, d0 = base
        keep = ~binary_dilation(np.isin(seg, robot_g), iterations=EDGE_PX)[rr, cc]
        r_open = float(((v.depth_m[rr, cc] - d0)[keep] > REVEAL_M).mean()) if keep.any() else 0.0
        has_blob, n_in, n_blob = blob(P, B)
        mug_px = sum(int(np.isin(obs[f"{c}_segmentation_element"][..., 0], mug_g).sum())
                     for c in ("agentview", "robot0_eye_in_hand"))
        rows.append(dict(t=t, door_gt=float(door_gt[t]), door_hat=round(th, 1), n_door=n_door, r_open=round(r_open, 3),
                         width_gt=round(w_gt, 4), width_inner_opening=entrance_width(m, d)[0],
                         width_hat=None if w_hat is None else round(w_hat, 4), blob=has_blob, n_inside=n_in,
                         n_blob=n_blob, mug_px=mug_px))
    return rows, [B[0].round(3).tolist(), B[1].round(3).tolist()]


def first(rows, f, hold=1):
    for i in range(len(rows) - hold + 1):
        if all(f(rows[j]) for j in range(i, i + hold)):
            return rows[i]["t"]
    return None


def evaluate(rows):
    t_open = first(rows, lambda r: r["door_gt"] > OPEN_DEG)
    t_open_hat = first(rows, lambda r: r["door_hat"] > OPEN_DEG, OPEN_HOLD)
    vis = [r for r in rows if r["n_door"] >= MIN_DOOR_PTS and r["door_hat"] > 0]
    ang_err = [abs(r["door_hat"] - r["door_gt"]) for r in vis]
    wrows = [r for r in rows if r["door_gt"] > 5]
    w_err = [r["width_hat"] - r["width_gt"] for r in wrows if r["width_hat"] is not None]
    w_missing = sum(r["width_hat"] is None for r in wrows)
    t_vis = first(rows, lambda r: r["mug_px"] > 0)
    t_blob = first(rows, lambda r: r["blob"])
    return dict(t_open_gt=t_open, t_open_hat=t_open_hat,
                open_dt=None if t_open is None or t_open_hat is None else t_open_hat - t_open,
                angle_mae_deg=round(float(np.mean(ang_err)), 1) if ang_err else None, angle_n=len(ang_err),
                width_mae_m=round(float(np.mean(np.abs(w_err))), 4) if w_err else None,
                width_bias_m=round(float(np.mean(w_err)), 4) if w_err else None,
                width_missing_steps=w_missing, width_open_steps=len(wrows),
                t_mug_visible_gt=t_vis, t_blob=t_blob,
                blob_dt=None if t_vis is None or t_blob is None else t_blob - t_vis,
                blob_false_alarm=t_blob is not None and (t_vis is None or t_blob < t_vis))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--states", default="0-9")
    a, b = map(int, ap.parse_args().states.split("-"))
    out = os.path.join(HERE, "results", "map_signals"); os.makedirs(out, exist_ok=True)   # (re-run overwrites)
    env = make_env(res=256, depth=True); env.reset()
    summ = []
    for k in range(a, b + 1):
        rows, B = run(k, env)
        e = dict(state=k, box_B=B, **evaluate(rows))
        json.dump(dict(eval=e, rows=rows), open(os.path.join(out, f"state{k}.json"), "w"), indent=1)
        summ.append(e); print(json.dumps(e), flush=True)
    ok = lambda f: sum(bool(f(e)) for e in summ)
    verdict = dict(
        open_signal_within_10=ok(lambda e: e["open_dt"] is not None and abs(e["open_dt"]) <= 10),
        angle_mae_le_10=ok(lambda e: e["angle_mae_deg"] is not None and e["angle_mae_deg"] <= 10),
        width_mae_mean_m=round(float(np.mean([e["width_mae_m"] for e in summ if e["width_mae_m"] is not None])), 4),
        width_bias_mean_m=round(float(np.mean([e["width_bias_m"] for e in summ if e["width_bias_m"] is not None])), 4),
        blob_within_20=ok(lambda e: e["blob_dt"] is not None and 0 <= e["blob_dt"] <= 20),
        blob_false_alarms=ok(lambda e: e["blob_false_alarm"]), n=len(summ))
    json.dump(dict(per_state=summ, verdict=verdict), open(os.path.join(out, "summary.json"), "w"), indent=1)
    print(json.dumps(verdict))


if __name__ == "__main__":
    main()
