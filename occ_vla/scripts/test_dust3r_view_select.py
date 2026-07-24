"""Auto-select the least-occluded virtual viewpoint (user request,
2026-07-22): given a real occluded moment (arm_s_occ > ARM_OCC_THRESHOLD,
matching the existing gate convention in arm_free_subgoal.py), generate
several candidate camera poses around the target, render each via
dust3r-point-cloud splatting (test_dust3r_novel_view_render.py's
mechanism), score each candidate by what fraction of the TARGET's own
3D points actually win the z-buffer at their projected pixel (i.e. are
not occluded by something nearer in that view), and pick the best.

This is a pure-geometry occlusion score, not a learned/heuristic one:
a target point is "visible" in a candidate render if nothing else in
the point cloud projects in front of it at the same pixel. No neural
network, no generated content -- consistent with the rest of this
dust3r investigation (real observed geometry only).

Still uses the step40 pair (target visible in both real cameras --
the actual "hidden from agentview, recovered via wrist" case still
needs a frame this episode's 20-step sampling didn't include, see
CLAUDE.md) -- this script demonstrates and validates the SELECTION
mechanism itself, which is the piece needed before gating on
arm_s_occ in a real rollout.

Run in .venv_mmada, one GPU:
  source .venv_mmada/bin/activate
  python3 scripts/test_dust3r_view_select.py
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
_DUST3R_ROOT = _ROOT / "third_party" / "dust3r"
sys.path.insert(0, str(_DUST3R_ROOT))

import torch  # noqa: E402
from dust3r.image_pairs import make_pairs  # noqa: E402
from dust3r.inference import inference  # noqa: E402
from dust3r.model import AsymmetricCroCo3DStereo  # noqa: E402
from dust3r.utils.image import load_images  # noqa: E402
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode  # noqa: E402

FRAME_DIR = _ROOT / "dense_occlusion_frames" / "bowl_top_drawer"
STEP = 45
OUT_DIR = _ROOT / "texture_ceiling_probe" / "dust3r_check"
CONF_THRESHOLD_PERCENTILE = 40
ARM_OCC_THRESHOLD = 0.30  # matches world_model/arm_free_subgoal.py's ARM_OCC_THRESHOLD
RENDER_SIZE = (384, 512)
N_CANDIDATES = 10  # small grid swept around the target, in addition to the wrist's own pose


def look_at(eye, target, up=np.array([0.0, -1.0, 0.0])):
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / (np.linalg.norm(right) + 1e-8)
    true_up = np.cross(right, forward)
    cam2world = np.eye(4)
    cam2world[:3, 0] = right
    cam2world[:3, 1] = true_up
    cam2world[:3, 2] = forward
    cam2world[:3, 3] = eye
    return cam2world


def project(pts3d_world, cam2world, focal, image_size):
    h, w = image_size
    world2cam = np.linalg.inv(cam2world)
    pts_h = np.concatenate([pts3d_world, np.ones((pts3d_world.shape[0], 1))], axis=1)
    pts_cam = (world2cam @ pts_h.T).T[:, :3]
    cx, cy = w / 2, h / 2
    u = (pts_cam[:, 0] / pts_cam[:, 2]) * focal + cx
    v = (pts_cam[:, 1] / pts_cam[:, 2]) * focal + cy
    return u, v, pts_cam[:, 2]


def render_and_score(all_pts, colors, confs, target_mask, cam2world, focal, image_size, conf_thr):
    h, w = image_size
    keep = confs >= conf_thr
    pts, cols, is_target = all_pts[keep], colors[keep], target_mask[keep]
    u, v, z = project(pts, cam2world, focal, image_size)

    in_front = z > 1e-4
    ui, vi = np.round(u).astype(int), np.round(v).astype(int)
    in_frame = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    valid = in_front & in_frame

    # score uses exact single-pixel z-buffer matching (below); the
    # canvas below additionally splats a small disk per point purely
    # for human-viewable rendering -- without this, sparse points at a
    # closer/different-scale virtual camera render as isolated single
    # pixels that are visually indistinguishable from black at normal
    # viewing size (found by inspection, 2026-07-22: scores were
    # correct but every candidate image looked empty).
    canvas = np.zeros((h, w, 3), dtype=np.float64)
    zbuf = np.full((h, w), np.inf)
    order = np.argsort(-z[valid])  # far to near, so near overwrites far
    vu, vv, vz, vc = ui[valid][order], vi[valid][order], z[valid][order], cols[valid][order]
    disk_radius = 1
    for i in range(len(vu)):
        if vz[i] < zbuf[vv[i], vu[i]]:
            zbuf[vv[i], vu[i]] = vz[i]
        for du in range(-disk_radius, disk_radius + 1):
            for dv in range(-disk_radius, disk_radius + 1):
                pu, pv = vu[i] + du, vv[i] + dv
                if 0 <= pu < w and 0 <= pv < h and vz[i] <= zbuf[pv, pu] + 1e-3:
                    canvas[pv, pu] = vc[i]

    # BUG (found by visual inspection, 2026-07-22): a candidate pose that
    # simply points away from the whole scene lands almost no points in
    # frame at all -- if the handful of target points that DO land
    # happen to be mutually unoccluded, the naive "fraction of in-frame
    # target points that are the z-buffer winner" score below is
    # spuriously ~1.0 for an almost-empty/black image. Score must also
    # require a real minimum of target points actually landing in frame
    # (relative to how many target points exist in total), not just
    # self-relative visibility among whichever handful showed up.
    n_target_total = int(is_target.sum())
    tgt_valid = valid & is_target
    n_target_in_frame = int(tgt_valid.sum())
    target_coverage = n_target_in_frame / max(n_target_total, 1)
    if n_target_in_frame < 50 or target_coverage < 0.05:
        return (canvas * 255).astype(np.uint8), 0.0, target_coverage
    tu, tv, tz = ui[tgt_valid], vi[tgt_valid], z[tgt_valid]
    winner_z = zbuf[tv, tu]
    visible = np.isclose(tz, winner_z, atol=1e-4)
    visibility = float(visible.mean())
    # final score requires BOTH: target actually present/covered in this
    # view, AND not occluded within that coverage
    score = visibility * min(1.0, target_coverage / 0.3)  # coverage saturates at 30% (rarely 100% due to sparse points)
    return (canvas * 255).astype(np.uint8), score, target_coverage


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AsymmetricCroCo3DStereo.from_pretrained("naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt").to(device)

    filelist = [str(FRAME_DIR / f"step{STEP:04d}_agentview.png"), str(FRAME_DIR / f"step{STEP:04d}_wrist.png")]
    imgs = load_images(filelist, size=512, verbose=True)
    pairs = make_pairs(imgs, scene_graph="complete", prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=1, verbose=True)
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PairViewer, verbose=True)

    rgbimg = scene.imgs
    pts3d = [p.detach().cpu().numpy() for p in scene.get_pts3d()]
    confs_list = [c.detach().cpu().numpy() for c in scene.im_conf]
    poses = scene.get_im_poses().detach().cpu().numpy()
    focals = scene.get_focals().detach().cpu().numpy()

    all_pts = np.concatenate([p.reshape(-1, 3) for p in pts3d], axis=0)
    all_colors = np.concatenate([np.asarray(im).reshape(-1, 3) for im in rgbimg], axis=0)
    all_confs = np.concatenate([c.reshape(-1) for c in confs_list], axis=0)
    conf_thr = np.percentile(all_confs, CONF_THRESHOLD_PERCENTILE)

    top_mask = all_confs >= np.percentile(all_confs, 90)
    target_centroid = all_pts[top_mask].mean(axis=0)
    target_radius = 1.5 * np.linalg.norm(all_pts[top_mask] - target_centroid, axis=1).std()
    target_mask = np.linalg.norm(all_pts - target_centroid, axis=1) <= target_radius
    print(f"target centroid={target_centroid}, radius={target_radius:.4f}, target points={target_mask.sum()}", flush=True)

    wrist_pos = poses[1][:3, 3]
    agentview_pos = poses[0][:3, 3]
    baseline = np.linalg.norm(agentview_pos - wrist_pos)

    # candidate poses: wrist's own pose, plus a ring of offsets around it at a few angles/heights
    rng = np.random.default_rng(0)
    candidates = {"wrist_orig": wrist_pos}
    for i in range(N_CANDIDATES):
        angle = 2 * np.pi * i / N_CANDIDATES
        r = baseline * rng.uniform(0.4, 0.9)
        height = baseline * rng.uniform(-0.3, 0.5)
        offset = np.array([r * np.cos(angle), height, r * np.sin(angle)])
        candidates[f"cand_{i:02d}"] = wrist_pos + offset

    focal_v = float(np.mean(focals))
    results = []
    for name, eye in candidates.items():
        cam2world = look_at(eye, target_centroid)
        img, score, coverage = render_and_score(all_pts, all_colors, all_confs, target_mask, cam2world, focal_v, RENDER_SIZE, conf_thr)
        results.append({"name": name, "score": score, "img": img, "coverage": coverage})
        print(f"{name}: score={score:.3f} target_coverage={coverage:.3f}", flush=True)

    results.sort(key=lambda r: -r["score"])
    best = results[0]
    print(f"\nbest candidate: {best['name']} (score={best['score']:.3f})", flush=True)
    print(f"would trigger dust3r-view injection if arm_s_occ > {ARM_OCC_THRESHOLD} "
          f"and best score clears an acceptance bar (e.g. > 0.7)", flush=True)

    Image.fromarray(best["img"]).save(OUT_DIR / f"step{STEP:04d}_autoselected_best.png")

    # contact sheet: real agentview | real wrist | top 4 candidates by score
    h, w = RENDER_SIZE
    real_a = np.asarray(Image.fromarray((np.asarray(rgbimg[0]) * 255).astype(np.uint8)).resize((w, h)))
    real_w = np.asarray(Image.fromarray((np.asarray(rgbimg[1]) * 255).astype(np.uint8)).resize((w, h)))
    tiles = [real_a, real_w] + [r["img"] for r in results[:4]]
    labels = ["real_agentview", "real_wrist"] + [f"{r['name']}({r['score']:.2f})" for r in results[:4]]
    sheet = Image.new("RGB", (w * len(tiles), h))
    for i, t in enumerate(tiles):
        sheet.paste(Image.fromarray(t), (i * w, 0))
    sheet.save(OUT_DIR / f"step{STEP:04d}_viewselect_sheet.png")
    print(f"\nsaved: {OUT_DIR / f'step{STEP:04d}_viewselect_sheet.png'} ({', '.join(labels)})")


if __name__ == "__main__":
    main()
