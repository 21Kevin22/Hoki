"""6-1 addendum: is the mug visible in the RGB images (what a policy sees), not just in segmentation?
LIBERO's microwave door glass has alpha 0.25 (segmentation: opaque, RGB: see-through); the task env makes it opaque.
For each init state: render RGB at 256x256 (the policy's render size) normally and with every mug geom made
fully transparent; count pixels whose max channel difference > THRESH. Writes results/init/rgb_visibility.json
and, per state, a [normal | mug hidden | |diff|x4] strip."""
import json, os, sys
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_init_states import HERE, OBJ, CAMS, zero_steps
from build_init_states import BDDL, OpaqueGlassEnv
THRESH = 10


def make_rgb_env():
    return OpaqueGlassEnv(bddl_file_name=BDDL, camera_heights=256, camera_widths=256, camera_names=CAMS)


def rgb_mug_pixels(env, state, save_prefix=None):
    """Load `state`, settle 10 zero steps (openpi's num_steps_wait), count RGB pixels changed by the mug."""
    env.reset(); env.set_init_state(state)
    obs = zero_steps(env, 10)
    m = env.sim.model._model
    g = [k for k in range(m.ngeom) if m.body(m.geom_bodyid[k]).name.startswith(OBJ)]
    a0 = m.geom_rgba[g, 3].copy()
    m.geom_rgba[g, 3] = 0.0                       # hide the mug for rendering only (no physics change)
    hid = env.env._get_observations(force_update=True)
    m.geom_rgba[g, 3] = a0
    r = {}
    for c in CAMS:
        A, B = obs[f"{c}_image"].astype(int), hid[f"{c}_image"].astype(int)
        diff = np.abs(A - B).max(-1)
        r[f"rgb_mug_pixels_{c}"] = int((diff > THRESH).sum())
        if save_prefix:   # panels left->right after the 180 deg flip: normal | mug hidden | |diff|x4
            img = np.concatenate([np.clip(diff * 4, 0, 255)[..., None].repeat(3, -1), B, A], 1).astype(np.uint8)
            Image.fromarray(img[::-1, ::-1]).save(f"{save_prefix}_{c}.png")
    return r


if __name__ == "__main__":
    env = make_rgb_env()
    states = np.load(os.path.join(HERE, "results", "init", "init_states.npz"))["states"]
    os.makedirs(os.path.join(HERE, "results", "init", "rgb_vis"), exist_ok=True)
    rows = []
    for i, s in enumerate(states):
        r = dict(state=i, **rgb_mug_pixels(env, s, os.path.join(HERE, "results", "init", "rgb_vis", f"state{i:02d}")))
        rows.append(r); print(json.dumps(r), flush=True)
    json.dump(dict(thresh=THRESH, res=256, rows=rows),
              open(os.path.join(HERE, "results", "init", "rgb_visibility.json"), "w"), indent=1)
