"""(i) v2: same replay as map_signals.py, signal definitions fixed after v1 (PREREG_next.md, 2026-11-20).
'Inside' signals use only space REVEALED since t0: agentview pixels (fixed camera) that were in box B at t0 and are now
>= 3 cm deeper (robot pixels removed), so the container's own outer surfaces never count.
Writes results/map_signals_v2/state<k>.json + summary.json.  Usage: python scripts/map_signals_v2.py [--states 0-9]"""
import argparse, json, os, sys
import numpy as np
from scipy.ndimage import binary_dilation
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from map_signals import (HERE, OBJ, MW, make_env, body_ids, geom_ids, cloud, door_angle, back_project, first,
                         OPEN_DEG, OPEN_HOLD, MIN_DOOR_PTS, REVEAL_M, IN_FRONT, FLOOR_LO, FLOOR_HI, WALL_IN, MIN_BLOB,
                         MIN_INSIDE, EDGE_PX)
from scripted_expert import entrance_width


def inside_points(R, B):
    """Revealed points behind the front plane (and inside B in x/z)."""
    lo, hi = B
    return R[(R[:, 1] > lo[1] + IN_FRONT) & (R[:, 0] > lo[0]) & (R[:, 0] < hi[0]) & (R[:, 2] > lo[2]) & (R[:, 2] < hi[2])]


def blob_v2(I):
    if len(I) < MIN_INSIDE:
        return None, 0
    floor = np.percentile(I[:, 2], 5)
    xl, xr = np.percentile(I[:, 0], 2), np.percentile(I[:, 0], 98)
    yb = np.percentile(I[:, 1], 98)
    S = I[(I[:, 2] > floor + FLOOR_LO) & (I[:, 2] < floor + FLOOR_HI) & (I[:, 0] > xl + WALL_IN) &
          (I[:, 0] < xr - WALL_IN) & (I[:, 1] < yb - WALL_IN)]
    return len(S) >= MIN_BLOB, len(S)


def run(k, env):
    z = np.load(os.path.join(HERE, "results", "expert", f"state{k}_traj.npz"))
    states, door_gt, phases = z["states"], z["door_deg"], z["phases"]
    init = np.load(os.path.join(HERE, "results", "init", "init_states.npz"))["states"][k]
    env.reset(); env.set_init_state(init)
    m, d = env.sim.model._model, env.sim.data._data
    robot_g = geom_ids(m, body_ids(m, ("robot0", "gripper0", "mount0")))
    mw_g, mug_g = geom_ids(m, body_ids(m, (MW,))), geom_ids(m, body_ids(m, (OBJ,)))
    obs = env.env._get_observations(force_update=True)
    Pm, views0 = cloud(env, obs, robot_g, only_geoms=mw_g)
    B = (Pm.min(0), Pm.max(0))
    v0 = views0[0]
    r0, c0 = np.nonzero(np.isfinite(v0.depth_m))
    inB = np.all((back_project(v0, r0, c0) >= B[0]) & (back_project(v0, r0, c0) <= B[1]), 1)
    rr, cc = r0[inB], c0[inB]; d0 = v0.depth_m[rr, cc]
    rows = []
    for t, s in enumerate(states):
        env.sim.set_state_from_flattened(s); env.sim.forward()
        obs = env.env._get_observations(force_update=True)
        P, views = cloud(env, obs, robot_g)
        th, D, n_door = door_angle(P, B)
        v = views[0]
        seg = obs["agentview_segmentation_element"][..., 0][::-1]
        keep = ~binary_dilation(np.isin(seg, robot_g), iterations=EDGE_PX)[rr, cc]
        rev = keep & ((v.depth_m[rr, cc] - d0) > REVEAL_M)
        R = back_project(v, rr[rev], cc[rev]) if rev.any() else np.zeros((0, 3))
        I = inside_points(R, B)
        has_blob, n_blob = blob_v2(I)
        if len(I) >= MIN_INSIDE and door_gt[t] > 5:
            xl, xr = np.percentile(I[:, 0], 2), np.percentile(I[:, 0], 98)
            dmax = D[:, 0].max() if th > 0 and len(D) else -np.inf
            w_hat = float(max(0.0, xr - max(xl, dmax)))
        else:
            w_hat = None
        px_a = int(np.isin(obs["agentview_segmentation_element"][..., 0], mug_g).sum())
        px_w = int(np.isin(obs["robot0_eye_in_hand_segmentation_element"][..., 0], mug_g).sum())
        rows.append(dict(t=t, phase=str(phases[t]), door_gt=float(door_gt[t]), door_hat=round(th, 1), n_door=n_door,
                         n_revealed=int(rev.sum()), n_inside=len(I), blob=bool(has_blob), n_blob=n_blob,
                         width_gt=entrance_width(m, d)[0], width_hat=None if w_hat is None else round(w_hat, 4),
                         mug_px_agent=px_a, mug_px_wrist=px_w))
    return rows, [B[0].round(3).tolist(), B[1].round(3).tolist()]


def evaluate(rows):
    t_open = first(rows, lambda r: r["door_gt"] > OPEN_DEG)
    t_open_hat = first(rows, lambda r: r["door_hat"] > OPEN_DEG, OPEN_HOLD)
    win = [r for r in rows if r["phase"].startswith(("open", "take"))]
    ang = [abs(r["door_hat"] - r["door_gt"]) for r in win if r["n_door"] >= MIN_DOOR_PTS and r["door_hat"] > 0]
    wr = [r for r in rows if r["door_gt"] > 5]
    we = [r["width_hat"] - r["width_gt"] for r in wr if r["width_hat"] is not None]
    t_vis_a = first(rows, lambda r: r["mug_px_agent"] > 0)
    t_vis_any = first(rows, lambda r: r["mug_px_agent"] > 0 or r["mug_px_wrist"] > 0)
    t_blob = first(rows, lambda r: r["blob"])
    return dict(t_open_gt=t_open, t_open_hat=t_open_hat,
                open_dt=None if None in (t_open, t_open_hat) else t_open_hat - t_open,
                angle_mae_deg=round(float(np.mean(ang)), 1) if ang else None, angle_n=len(ang),
                width_mae_m=round(float(np.mean(np.abs(we))), 4) if we else None,
                width_bias_m=round(float(np.mean(we)), 4) if we else None,
                width_eval_steps=len(we), width_missing_steps=len(wr) - len(we),
                t_mug_visible_agent=t_vis_a, t_mug_visible_any=t_vis_any, t_blob=t_blob,
                blob_dt=None if None in (t_vis_a, t_blob) else t_blob - t_vis_a,
                blob_false_alarm=t_blob is not None and (t_vis_a is None or t_blob < t_vis_a))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--states", default="0-9")
    a, b = map(int, ap.parse_args().states.split("-"))
    out = os.path.join(HERE, "results", "map_signals_v2"); os.makedirs(out, exist_ok=True)
    env = make_env(res=256, depth=True); env.reset()
    summ = []
    for k in range(a, b + 1):
        rows, B = run(k, env)
        e = dict(state=k, box_B=B, **evaluate(rows))
        json.dump(dict(eval=e, rows=rows), open(os.path.join(out, f"state{k}.json"), "w"), indent=1)
        summ.append(e); print(json.dumps(e), flush=True)
    ok = lambda f: sum(bool(f(e)) for e in summ)
    mean = lambda key: round(float(np.mean([e[key] for e in summ if e[key] is not None])), 4) if any(e[key] is not None for e in summ) else None
    verdict = dict(open_signal_within_10=ok(lambda e: e["open_dt"] is not None and abs(e["open_dt"]) <= 10),
                   angle_mae_le_10=ok(lambda e: e["angle_mae_deg"] is not None and e["angle_mae_deg"] <= 10),
                   width_mae_mean_m=mean("width_mae_m"), width_bias_mean_m=mean("width_bias_m"),
                   blob_within_20=ok(lambda e: e["blob_dt"] is not None and 0 <= e["blob_dt"] <= 20),
                   blob_false_alarms=ok(lambda e: e["blob_false_alarm"]), n=len(summ))
    json.dump(dict(per_state=summ, verdict=verdict), open(os.path.join(out, "summary.json"), "w"), indent=1)
    print(json.dumps(verdict))


if __name__ == "__main__":
    main()
