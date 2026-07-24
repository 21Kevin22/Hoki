"""Grid-search viewpoint selection using TRUE camera poses (user's
"Option 2", 2026-07-22) instead of dust3r's own cross-view pose
estimation -- which the previous check
(test_dust3r_view_select.py on the real occluded step45 pair) found
breaks down specifically when agentview and wrist share little visual
content (exactly the occluded case this whole investigation targets).

Fix: each view's own per-view depth map from dust3r still looked
internally plausible in isolation (visually confirmed,
step0045_rgb_depth_conf.png) -- only the CROSS-view pose alignment was
broken. So: unproject each view's dust3r depth into that view's own
local camera frame (using dust3r's own estimated focal, consistent with
how that depth was produced), then place both point clouds into a
shared WORLD frame using the REAL simulator camera extrinsics
(collect_dense_occlusion_frames.py's v2 run saved cam_pos/cam_mat/fovy
for both cameras at every step) -- this is privileged sim-only info
(same category CameraProjector.from_sim already uses for agentview),
not dust3r-estimated.

Then: build a systematic grid of candidate camera positions around the
target (not random offsets, per user's "grid" request), score each by a
real S_occ analog (fraction of the target's own 3D points OCCLUDED --
not the z-buffer winner -- at that viewpoint; 0 = fully visible, matching
this project's existing S_occ convention where higher = more occluded),
and pick the grid point that minimizes it.

Run in .venv_mmada, one GPU:
  source .venv_mmada/bin/activate
  python3 scripts/test_dust3r_true_pose_grid_select.py
"""

import json
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

FRAME_DIR = _ROOT / "dense_occlusion_frames" / "bowl_top_drawer_v2_with_poses"
STEP = 44
OUT_DIR = _ROOT / "texture_ceiling_probe" / "dust3r_check"
CONF_THRESHOLD_PERCENTILE = 40
RENDER_SIZE = (384, 512)
N_AZIMUTH = 8
N_ELEVATION = 3
N_RADIUS = 2


def cam_to_world(cam_pos, cam_mat_flat):
    """MuJoCo cam_xpos/cam_xmat -> 4x4 cam2world. cam_mat is the
    camera-frame-to-world rotation (same convention CameraProjector uses:
    cam_mat.T rotates world vectors into camera frame, so cam_mat itself
    rotates camera-frame vectors into world)."""
    R = np.array(cam_mat_flat).reshape(3, 3)
    t = np.array(cam_pos)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def unproject_depth(depth, focal, principal=None):
    """depth: (H,W) in this view's own local camera-frame Z. Returns
    (H*W, 3) points in that camera's local frame (MuJoCo/OpenCV-style:
    +Z looking-direction convention matched to dust3r's own output)."""
    h, w = depth.shape
    cx, cy = (w / 2, h / 2) if principal is None else principal
    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    x = (us - cx) * depth / focal
    y = (vs - cy) * depth / focal
    z = depth
    return np.stack([x, y, z], axis=-1).reshape(-1, 3)


def look_at(eye, target, up=np.array([0.0, 0.0, 1.0])):
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
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


def s_occ_score(all_pts, colors, confs, target_mask, cam2world, focal, image_size, conf_thr):
    """Returns (rendered_canvas, s_occ) -- s_occ in [0,1], 0 = target
    fully visible (matches this project's S_occ convention: fraction of
    the target's clear footprint that's occluded)."""
    h, w = image_size
    keep = confs >= conf_thr
    pts, cols, is_target = all_pts[keep], colors[keep], target_mask[keep]
    u, v, z = project(pts, cam2world, focal, image_size)
    ui, vi = np.round(u).astype(int), np.round(v).astype(int)
    in_frame = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    valid = in_frame & (z > 1e-4)

    zbuf = np.full((h, w), np.inf)
    canvas = np.zeros((h, w, 3), dtype=np.float64)
    order = np.argsort(-z[valid])
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

    n_target_total = int(is_target.sum())
    tgt_valid = valid & is_target
    n_target_in_frame = int(tgt_valid.sum())
    target_coverage = n_target_in_frame / max(n_target_total, 1)
    if n_target_in_frame < 50 or target_coverage < 0.05:
        return (canvas * 255).astype(np.uint8), 1.0, target_coverage  # no real view -> maximally occluded, not "visible"
    tu, tv, tz = ui[tgt_valid], vi[tgt_valid], z[tgt_valid]
    winner_z = zbuf[tv, tu]
    occluded = ~np.isclose(tz, winner_z, atol=1e-4)
    s_occ = float(occluded.mean())
    # penalize low coverage even when the visible fraction looks clean
    s_occ = 1.0 - (1.0 - s_occ) * min(1.0, target_coverage / 0.3)
    return (canvas * 255).astype(np.uint8), s_occ, target_coverage


def main():
    manifest = json.loads((FRAME_DIR / "manifest.json").read_text())
    entry = next(e for e in manifest["steps"] if e["step"] == STEP)
    agentview_pose = entry["cam_poses"]["agentview"]
    wrist_pose = entry["cam_poses"]["robot0_eye_in_hand"]
    print(f"true agentview cam_pos={agentview_pose['cam_pos']}", flush=True)
    print(f"true wrist cam_pos={wrist_pose['cam_pos']}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AsymmetricCroCo3DStereo.from_pretrained("naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt").to(device)

    filelist = [str(FRAME_DIR / f"step{STEP:04d}_agentview.png"), str(FRAME_DIR / f"step{STEP:04d}_wrist.png")]
    imgs = load_images(filelist, size=512, verbose=True)
    pairs = make_pairs(imgs, scene_graph="complete", prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=1, verbose=True)
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PairViewer, verbose=True)

    rgbimg = scene.imgs
    depths = [d.detach().cpu().numpy() for d in scene.get_depthmaps()]
    confs_list = [c.detach().cpu().numpy() for c in scene.im_conf]
    dust3r_focals = scene.get_focals().detach().cpu().numpy()

    # unproject each view's own depth in its OWN local camera frame (dust3r's
    # estimated focal, consistent with how that depth was produced), then
    # place into world frame using the REAL sim camera pose -- not dust3r's
    # own (broken, for this pair) cross-view pose estimate.
    true_poses = [
        cam_to_world(agentview_pose["cam_pos"], agentview_pose["cam_mat"]),
        cam_to_world(wrist_pose["cam_pos"], wrist_pose["cam_mat"]),
    ]
    all_pts_list, all_colors_list, all_confs_list = [], [], []
    for i in range(2):
        local_pts = unproject_depth(depths[i], float(dust3r_focals[i]))
        # dust3r/OpenCV-style local frame (+Z forward, +Y down) -> MuJoCo
        # camera-local frame (+Z backward/out-of-screen, +Y up, per
        # CameraProjector's convention: p_cam = cam_mat.T @ (p_world - cam_pos),
        # screen_x = p_cam.x / -p_cam.z) -- flip Y and Z to match before
        # applying the true cam2world rotation.
        local_pts_mj = local_pts * np.array([1.0, -1.0, -1.0])
        world_pts = (true_poses[i][:3, :3] @ local_pts_mj.T).T + true_poses[i][:3, 3]
        all_pts_list.append(world_pts)
        all_colors_list.append(np.asarray(rgbimg[i]).reshape(-1, 3))
        all_confs_list.append(confs_list[i].reshape(-1))

    all_pts = np.concatenate(all_pts_list, axis=0)
    all_colors = np.concatenate(all_colors_list, axis=0)
    all_confs = np.concatenate(all_confs_list, axis=0)
    conf_thr = np.percentile(all_confs, CONF_THRESHOLD_PERCENTILE)

    # target centroid: high-confidence points from the WRIST view specifically
    # (view 1 -- the one that actually saw the target clearly at this occluded step)
    wrist_conf = all_confs_list[1]
    wrist_top = wrist_conf >= np.percentile(wrist_conf, 90)
    target_centroid = all_pts_list[1][wrist_top].mean(axis=0)
    target_radius = 1.5 * np.linalg.norm(all_pts_list[1][wrist_top] - target_centroid, axis=1).std()
    target_mask = np.linalg.norm(all_pts - target_centroid, axis=1) <= target_radius
    print(f"target centroid (world/true frame)={target_centroid}, radius={target_radius:.4f}, "
          f"target pts={target_mask.sum()}", flush=True)
    print(f"true baseline (agentview-wrist distance)="
          f"{np.linalg.norm(np.array(agentview_pose['cam_pos']) - np.array(wrist_pose['cam_pos'])):.4f} m", flush=True)

    # systematic grid: azimuth x elevation x radius, radius scaled by the
    # REAL true distance from wrist to target (metric, from true poses)
    wrist_to_target = np.linalg.norm(np.array(wrist_pose["cam_pos"]) - target_centroid)
    grid = {}
    for ir, radius_scale in enumerate(np.linspace(0.7, 1.3, N_RADIUS)):
        for ia, az in enumerate(np.linspace(0, 2 * np.pi, N_AZIMUTH, endpoint=False)):
            for ie, el in enumerate(np.linspace(-0.3, 0.6, N_ELEVATION)):
                r = wrist_to_target * radius_scale
                offset = r * np.array([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)])
                eye = target_centroid + offset
                grid[f"az{ia}_el{ie}_r{ir}"] = eye
    grid["wrist_orig"] = np.array(wrist_pose["cam_pos"])
    grid["agentview_orig"] = np.array(agentview_pose["cam_pos"])

    focal_v = float(np.mean(dust3r_focals))
    results = []
    for name, eye in grid.items():
        cam2world = look_at(eye, target_centroid)
        img, s_occ, coverage = s_occ_score(all_pts, all_colors, all_confs, target_mask, cam2world, focal_v, RENDER_SIZE, conf_thr)
        results.append({"name": name, "s_occ": s_occ, "coverage": coverage, "img": img})

    results.sort(key=lambda r: r["s_occ"])
    print(f"\n{len(results)} grid candidates scored", flush=True)
    for r in results[:8]:
        print(f"  {r['name']}: S_occ={r['s_occ']:.3f} coverage={r['coverage']:.3f}", flush=True)
    print("  ...", flush=True)
    for r in results[-3:]:
        print(f"  {r['name']}: S_occ={r['s_occ']:.3f} coverage={r['coverage']:.3f}", flush=True)

    best = results[0]
    print(f"\nbest (lowest S_occ): {best['name']} (S_occ={best['s_occ']:.3f})", flush=True)

    h, w = RENDER_SIZE
    real_a = np.asarray(Image.fromarray((np.asarray(rgbimg[0]) * 255).astype(np.uint8)).resize((w, h)))
    real_w = np.asarray(Image.fromarray((np.asarray(rgbimg[1]) * 255).astype(np.uint8)).resize((w, h)))
    top4 = results[:4]
    tiles = [real_a, real_w] + [r["img"] for r in top4]
    sheet = Image.new("RGB", (w * len(tiles), h))
    for i, t in enumerate(tiles):
        sheet.paste(Image.fromarray(t), (i * w, 0))
    sheet.save(OUT_DIR / f"step{STEP:04d}_truepose_gridselect_sheet.png")
    top4_labels = ", ".join(f"{r['name']}(Socc={r['s_occ']:.2f})" for r in top4)
    print(f"\nsaved: real_agentview | real_wrist | {top4_labels} "
          f"-> {OUT_DIR / f'step{STEP:04d}_truepose_gridselect_sheet.png'}")


if __name__ == "__main__":
    main()
