"""How does the door glass appear in RGB and depth, before (LIBERO alpha 0.25) and after (opaque) the patch?
For init state 0 with the door closed: in each camera, take the pixels of the glass geom (element segmentation)
and compare their depth with the door plane (= glass rendered as a surface) vs the interior behind it (= see-through).
Also compares depth of glass pixels with the arm-free / mug-free render to see if depth sees the mug.
Writes results/glass_depth_check.json and results/glass_depth/*.png."""
import json, os, sys
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_init_states import HERE, OBJ, CAMS, make_env, zero_steps, OPAQUE_MATERIALS
from robosuite.utils.camera_utils import get_real_depth_map

out = os.path.join(HERE, "results", "glass_depth"); os.makedirs(out, exist_ok=True)
states = np.load(os.path.join(HERE, "results", "init", "init_states.npz"))["states"]
env = make_env(res=256, depth=True)
rows = {}
env.reset(); env.set_init_state(states[0]); zero_steps(env, 10)
m = env.sim.model._model
mid = m.material(OPAQUE_MATERIALS[0]).id
glass_g = [g for g in range(m.ngeom) if m.geom_matid[g] == mid]
mug_g = [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.startswith(OBJ)]


def render(glass_alpha, mug_visible=True, glass_present=True):
    """Re-render the SAME physical state with the given visual settings (no stepping)."""
    m.mat_rgba[mid, 3] = glass_alpha
    ga, ma = m.geom_rgba[glass_g, 3].copy(), m.geom_rgba[mug_g, 3].copy()
    if not glass_present:
        m.geom_rgba[glass_g, 3] = 0.0; m.geom_group[glass_g] = 5      # group 5 is not rendered
    if not mug_visible:
        m.geom_rgba[mug_g, 3] = 0.0
    o = env.env._get_observations(force_update=True)
    m.geom_rgba[glass_g, 3], m.geom_rgba[mug_g, 3] = ga, ma
    m.geom_group[glass_g] = 1
    return o


views = dict(alpha025=render(0.25), opaque=render(1.0), no_glass=render(1.0, glass_present=False),
             alpha025_nomug=render(0.25, mug_visible=False), opaque_nomug=render(1.0, mug_visible=False))
for c in CAMS:
    seg = views["opaque"][f"{c}_segmentation_element"][..., 0]
    gm = np.isin(seg, glass_g)
    D = {k: get_real_depth_map(env.sim, v[f"{c}_depth"])[..., 0] for k, v in views.items()}
    R = {k: v[f"{c}_image"].astype(int) for k, v in views.items()}
    r = dict(glass_pixels=int(gm.sum()))
    if gm.any():
        r["depth_on_glass_px_median_m"] = {k: round(float(np.median(D[k][gm])), 4) for k in ("alpha025", "opaque", "no_glass")}
        r["frac_glass_px_where_alpha025_depth_equals_no_glass"] = round(float((np.abs(D["alpha025"] - D["no_glass"])[gm] < 0.002).mean()), 3)
        r["frac_glass_px_where_alpha025_depth_equals_opaque"] = round(float((np.abs(D["alpha025"] - D["opaque"])[gm] < 0.002).mean()), 3)
    r["rgb_px_changed_by_mug_alpha025"] = int((np.abs(R["alpha025"] - R["alpha025_nomug"]).max(-1) > 10).sum())
    r["rgb_px_changed_by_mug_opaque"] = int((np.abs(R["opaque"] - R["opaque_nomug"]).max(-1) > 10).sum())
    r["depth_px_changed_by_mug_alpha025"] = int((np.abs(D["alpha025"] - get_real_depth_map(env.sim, views["alpha025_nomug"][f"{c}_depth"])[..., 0]) > 0.005).sum())
    rows[c] = r
    for k in ("alpha025", "opaque"):
        v = np.clip((D[k] - 0.5) / 1.5, 0, 1)
        Image.fromarray(np.concatenate([views[k][f"{c}_image"], (np.stack([v] * 3, -1) * 255).astype(np.uint8)], 1)[::-1, ::-1]
                        ).save(os.path.join(out, f"{k}_{c}.png"))
print(json.dumps(rows, indent=1))
json.dump(dict(state=0, note="same physical state re-rendered; no stepping between views", cameras=rows),
          open(os.path.join(HERE, "results", "glass_depth_check.json"), "w"), indent=1)
