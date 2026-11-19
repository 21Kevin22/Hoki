"""Probe the original libero_10 KITCHEN_SCENE6 microwave scene (GPU-free): joints, door angle at every
official init state, heating-region site, and microwave geometry. Used for 6-2's comparison."""
import json, os, numpy as np, torch
_l = torch.load; torch.load = lambda *a, **k: _l(*a, **{**k, "weights_only": False})
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
suite = benchmark.get_benchmark_dict()["libero_10"]()
tid = [i for i, n in enumerate(suite.get_task_names()) if "microwave" in n][0]
task = suite.get_task(tid)
bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)
env.seed(0); env.reset()
m, d = env.sim.model._model, env.sim.data._data
joints = [m.joint(i).name for i in range(m.njnt) if "microwave" in m.joint(i).name]
sites = {m.site(i).name: dict(pos=d.site(i).xpos.round(4).tolist(), size=m.site_size[i].round(4).tolist())
         for i in range(m.nsite) if "microwave" in m.site(i).name}
door = m.joint([j for j in joints if "free" not in j and "joint0" not in j][0])
states = suite.get_task_init_states(tid)
angles = []
for s in states:
    env.set_init_state(s); angles.append(float(env.sim.data._data.qpos[door.qposadr[0]]))
bodies = [m.body(i).name for i in range(m.nbody) if "microwave" in m.body(i).name]
out = dict(task=suite.get_task_names()[tid], bddl=bddl, joints=joints, door_joint=door.name,
           door_range=m.jnt_range[door.id].tolist(), n_init_states=len(states),
           door_angle_init_rad=dict(min=min(angles), max=max(angles), mean=float(np.mean(angles))),
           door_angle_init_deg_mean=float(np.degrees(np.mean(angles))), sites=sites, bodies=bodies)
print(json.dumps(out, indent=1))
os.makedirs("results", exist_ok=True); json.dump(out, open("results/probe_scene6.json", "w"), indent=1)
