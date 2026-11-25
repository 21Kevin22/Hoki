"""PREREG (14): replay saved H3 episodes (init state + executed actions) without the policy; check reproduction and
dump decision points: agentview / wrist RGB and agentview segmentation (128x128, mujoco.Renderer, upright), the next
16 executed actions, the agentview RGB 16 steps later, and the label (a non-task body moves > 1 mm in those 16 steps;
the hidden box in its container frame). Usage: python main/replay14.py C0_side9 VIS_side9 LOOK_main10"""
import sys, os, json, glob
import numpy as np
import mujoco
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from libero.libero.envs import OffScreenRenderEnv
from parts import TaskParts, TASK_CONTAINER
from collisions import CollisionRecorder

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DUMMY = [0.0] * 6 + [-1.0]
H, EVERY, RES = 16, 5, 128
OUT = os.path.join(HERE, "main", "data14"); os.makedirs(OUT, exist_ok=True)
states = np.load(os.path.join(HERE, "main", "init_states_hidden", "H3.npz"))["eval"]
env = OffScreenRenderEnv(bddl_file_name=os.path.join(HERE, "main", "bddl_hidden", "H3.bddl"), camera_heights=8,
                         camera_widths=8, horizon=10 ** 7)
env.seed(7)
rep = []
SH = os.environ.get("SHARD", "0/1"); k_, n_ = map(int, SH.split("/"))
# MuJoCo's warm start carries over between episodes of one process: replay each original job chunk (5 episodes,
# idx 5c..5c+4, in order) in a FRESH env seeded like the original (env.seed(7)), exactly as run_closed_loop ran it.
chunks = [(tag, c) for tag in sys.argv[1:] for c in range(8)]
def episodes():
    global env
    for tag, c in chunks[k_::n_]:
        env.close()
        env = OffScreenRenderEnv(bddl_file_name=os.path.join(HERE, "main", "bddl_hidden", "H3.bddl"),
                                 camera_heights=256, camera_widths=256, camera_depths=True,
                                 camera_names=["agentview", "robot0_eye_in_hand"], horizon=10 ** 7)
        env.seed(7)
        for i in range(5 * c, 5 * c + 5):
            jf = os.path.join(HERE, "main", "runs", tag, f"T1_hid_eval{i:02d}.json")
            if os.path.exists(jf):
                yield tag, jf
for tag, jf in episodes():
    if True:
        stem = jf[:-5]; i = int(stem[-2:])
        J = json.load(open(jf)); A = np.load(stem + ".npz")["a"]
        env.reset(); env.set_init_state(states[i])
        m, d = env.sim.model._model, env.sim.data._data
        rec = CollisionRecorder(TaskParts("T1", m, d), m, d)
        root, _ = TASK_CONTAINER["T1"]; ref = m.body(f"{root}_main").id
        box = {b for b in rec.bystanders if rec.root_name[b].startswith("butter_9")}
        rel = lambda b: d.xmat[ref].reshape(3, 3).T @ (d.xpos[b] - d.xpos[ref]) if b in box else d.xpos[b].copy()
        r = mujoco.Renderer(m, RES, RES)
        r2 = mujoco.Renderer(m, 256, 256)
        big = {}; prop = {}
        robot = np.array([rec.is_robot[b] for b in range(m.nbody)])
        tgt = {b for b in range(m.nbody) if rec.obj_root(b) in rec.p.target_roots}
        byst = set(rec.bystanders)
        for _ in range(10):
            env.step(DUMMY)
        p0 = {b: rel(b) for b in rec.bystanders}
        P, imgs, wrist, seg = [], [], [], []
        def snap():
            out = []
            for cam in ("agentview", "robot0_eye_in_hand"):
                out.append(env.sim.render(width=RES, height=RES, camera_name=cam)[::-1].copy())   # LIBERO pipeline, upright
            r.update_scene(d, camera="agentview"); r.enable_segmentation_rendering(); s = r.render().copy(); r.disable_segmentation_rendering()
            g = np.where(s[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), s[..., 0], -1)
            b = np.where(g >= 0, m.geom_bodyid[np.clip(g, 0, None)], -1)
            lab = np.zeros(g.shape, np.int8)                      # 0 other, 1 robot, 2 target, 3 bystander
            lab[(b >= 0) & robot[np.clip(b, 0, None)]] = 1
            lab[np.isin(b, list(tgt))] = 2; lab[np.isin(b, list(byst))] = 3
            return out[0], out[1], lab
        def snap_big(step):                      # 256-px frames (Cosmos Policy) at decision steps and t+16; proprio
            if step % EVERY in (0, 1):
                out = []
                for cam in ("agentview", "robot0_eye_in_hand"):
                    out.append(env.sim.render(width=256, height=256, camera_name=cam)[::-1].copy())
                big[step] = out
            if step % EVERY == 0:
                ob = env.env._get_observations(force_update=False)
                prop[step] = np.concatenate((ob["robot0_gripper_qpos"], ob["robot0_eef_pos"], ob["robot0_eef_quat"]))
        a0, w0, l0 = snap(); imgs.append(a0); wrist.append(w0); seg.append(l0); snap_big(0)
        P.append(np.array([rel(b) for b in rec.bystanders]))
        for k, u in enumerate(A):
            env.step(np.asarray(u, float).tolist())
            P.append(np.array([rel(b) for b in rec.bystanders]))
            a_, w_, l_ = snap(); imgs.append(a_); wrist.append(w_); seg.append(l_)      # every step (index = step)
            snap_big(k + 1)
        P = np.array(P)                                           # (T+1, nbyst, 3), index = step after 10 settle steps
        boxi = [j for j, b in enumerate(rec.bystanders) if b in box]
        maxbox = float(np.abs(P[:, boxi] - P[0, boxi]).sum(-1).max()) if boxi else 0.0
        ref_box = J["collision"]["hidden"]["box_max_disp_rel_l1_m"]
        # decision points: every 5 executed steps; label and future image exactly 16 steps later
        dec = [t for t in range(0, len(A) - H + 1, EVERY)]
        lab = [bool(np.abs(P[t:t + H + 1] - P[t]).sum(-1).max() > 0.001) for t in dec]
        labb = [bool(np.abs(P[t:t + H + 1][:, boxi] - P[t][boxi]).sum(-1).max() > 0.001) if boxi else False for t in dec]
        n = len(dec)
        np.savez_compressed(os.path.join(OUT, f"{tag}_{i:02d}.npz"), agent=np.array([imgs[t] for t in dec]),
                            wrist=np.array([wrist[t] for t in dec]), seg=np.array([seg[t] for t in dec]),
                            agent_prev=np.array([imgs[max(t - 1, 0)] for t in dec]),
                            wrist_prev=np.array([wrist[max(t - 1, 0)] for t in dec]),
                            agent_future2=np.array([imgs[min(t + H + 1, len(imgs) - 1)] for t in dec]),
                            agent_future=np.array([imgs[t + H] for t in dec]),
                            actions=np.array([A[t:t + H] for t in dec]), label=np.array(lab), label_box=np.array(labb),
                            t=np.array(dec),
                            agent256=np.array([big[t][0] for t in dec]), wrist256=np.array([big[t][1] for t in dec]),
                            agent256_future=np.array([big[t + H][0] for t in dec]), proprio=np.array([prop[t] for t in dec]))
        dec = [dict(label=x) for x in lab]
        rep.append(dict(run=f"{tag}_{i:02d}", replay_box=round(maxbox, 4), orig_box=ref_box, ok=abs(maxbox - ref_box) <= 0.0005,
                        n_dec=n, pos=int(sum(d_["label"] for d_ in dec))))
        print(json.dumps(rep[-1]), flush=True)
        del r
json.dump(rep, open(os.path.join(OUT, f"replay_check_{k_}.json"), "w"), indent=1)
ok = sum(r_["ok"] for r_ in rep)
print(f"reproduced {ok}/{len(rep)}", flush=True)
