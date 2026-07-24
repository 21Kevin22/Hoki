"""Run the already-validated dust3r true-pose recovery pipeline
(test_dust3r_true_pose_grid_select.py's approach, confirmed clean --
S_occ=0.000, correct color, no artifacts -- on bowl_top_drawer) on the
fresh occlusion moment collected by collect_occlusion_moment_with_state.py,
producing a 224x224 recovered image ready to inject into pi0.5 via
OccVlaLiberoInputs' subgoal_image slot.

Run in .venv_mmada, one GPU:
  source .venv_mmada/bin/activate
  python3 scripts/generate_recovery_for_injection_test.py
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

FRAME_DIR = _ROOT / "texture_ceiling_probe" / "pi05_injection_test"
RENDER_SIZE = (384, 512)
N_AZIMUTH = 8
N_ELEVATION = 3
N_RADIUS = 2
CONF_THRESHOLD_PERCENTILE = 40


def cam_to_world(cam_pos, cam_mat_flat):
    R = np.array(cam_mat_flat).reshape(3, 3)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.array(cam_pos)
    return T


def unproject_depth(depth, focal):
    h, w = depth.shape
    cx, cy = w / 2, h / 2
    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    x = (us - cx) * depth / focal
    y = (vs - cy) * depth / focal
    return np.stack([x, y, depth], axis=-1).reshape(-1, 3)


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
        return (canvas * 255).astype(np.uint8), 1.0, target_coverage
    tu, tv, tz = ui[tgt_valid], vi[tgt_valid], z[tgt_valid]
    winner_z = zbuf[tv, tu]
    occluded = ~np.isclose(tz, winner_z, atol=1e-4)
    s_occ = float(occluded.mean())
    s_occ = 1.0 - (1.0 - s_occ) * min(1.0, target_coverage / 0.3)
    return (canvas * 255).astype(np.uint8), s_occ, target_coverage


def main():
    meta = json.loads((FRAME_DIR / "meta.json").read_text())
    agentview_pose = meta["cam_poses"]["agentview"]
    wrist_pose = meta["cam_poses"]["robot0_eye_in_hand"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AsymmetricCroCo3DStereo.from_pretrained("naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt").to(device)

    filelist = [str(FRAME_DIR / "occluded_agentview_256.png"), str(FRAME_DIR / "occluded_wrist_256.png")]
    imgs = load_images(filelist, size=512, verbose=True)
    pairs = make_pairs(imgs, scene_graph="complete", prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=1, verbose=True)
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PairViewer, verbose=True)

    rgbimg = scene.imgs
    depths = [d.detach().cpu().numpy() for d in scene.get_depthmaps()]
    confs_list = [c.detach().cpu().numpy() for c in scene.im_conf]
    dust3r_focals = scene.get_focals().detach().cpu().numpy().reshape(-1)

    true_poses = [
        cam_to_world(agentview_pose["cam_pos"], agentview_pose["cam_mat"]),
        cam_to_world(wrist_pose["cam_pos"], wrist_pose["cam_mat"]),
    ]
    all_pts_list, all_colors_list, all_confs_list = [], [], []
    for i in range(2):
        local_pts = unproject_depth(depths[i], float(dust3r_focals[i]))
        local_pts_mj = local_pts * np.array([1.0, -1.0, -1.0])
        world_pts = (true_poses[i][:3, :3] @ local_pts_mj.T).T + true_poses[i][:3, 3]
        all_pts_list.append(world_pts)
        all_colors_list.append(np.asarray(rgbimg[i]).reshape(-1, 3))
        all_confs_list.append(confs_list[i].reshape(-1))

    all_pts = np.concatenate(all_pts_list, axis=0)
    all_colors = np.concatenate(all_colors_list, axis=0)
    all_confs = np.concatenate(all_confs_list, axis=0)
    conf_thr = np.percentile(all_confs, CONF_THRESHOLD_PERCENTILE)

    wrist_conf = all_confs_list[1]
    wrist_top = wrist_conf >= np.percentile(wrist_conf, 90)
    target_centroid = all_pts_list[1][wrist_top].mean(axis=0)
    target_radius = 1.5 * np.linalg.norm(all_pts_list[1][wrist_top] - target_centroid, axis=1).std()
    target_mask = np.linalg.norm(all_pts - target_centroid, axis=1) <= target_radius
    print(f"target centroid={target_centroid}, radius={target_radius:.4f}, target pts={target_mask.sum()}", flush=True)

    wrist_pos = np.array(wrist_pose["cam_pos"])
    agentview_pos = np.array(agentview_pose["cam_pos"])
    baseline = np.linalg.norm(agentview_pos - wrist_pos)

    real_dirs = [(target_centroid - wrist_pos), (target_centroid - agentview_pos)]
    real_dirs = [d / np.linalg.norm(d) for d in real_dirs]

    def angle_penalty(eye):
        cand_dir = (target_centroid - eye)
        cand_dir = cand_dir / np.linalg.norm(cand_dir)
        min_angle = min(np.arccos(np.clip(cand_dir @ rd, -1.0, 1.0)) for rd in real_dirs)
        return min_angle

    grid = {"wrist_orig": wrist_pos, "agentview_orig": agentview_pos}
    for ir, radius_scale in enumerate(np.linspace(0.7, 1.3, N_RADIUS)):
        for ia, az in enumerate(np.linspace(0, 2 * np.pi, N_AZIMUTH, endpoint=False)):
            for ie, el in enumerate(np.linspace(-0.3, 0.6, N_ELEVATION)):
                r = baseline * radius_scale
                offset = r * np.array([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)])
                eye = target_centroid + offset
                grid[f"az{ia}_el{ie}_r{ir}"] = eye

    focal_v = float(np.mean(dust3r_focals))
    results = []
    for name, eye in grid.items():
        cam2world = look_at(eye, target_centroid)
        img, s_occ, coverage = s_occ_score(all_pts, all_colors, all_confs, target_mask, cam2world, focal_v, RENDER_SIZE, conf_thr)
        ang = angle_penalty(eye)
        combined = s_occ + (ang / np.pi)
        results.append({"name": name, "s_occ": s_occ, "coverage": coverage, "angle_deg": np.degrees(ang), "combined": combined, "img": img})

    results.sort(key=lambda r: r["combined"])
    print(f"\ntop candidates:", flush=True)
    for r in results[:5]:
        print(f"  {r['name']}: S_occ={r['s_occ']:.3f} angle={r['angle_deg']:.1f}deg coverage={r['coverage']:.3f}", flush=True)

    best = results[0]
    print(f"\nbest: {best['name']} (S_occ={best['s_occ']:.3f})", flush=True)

    best_224 = np.asarray(Image.fromarray(best["img"]).resize((224, 224)))
    Image.fromarray(best_224).save(FRAME_DIR / "recovered_224.png")
    print(f"saved recovered image for injection: {FRAME_DIR / 'recovered_224.png'}", flush=True)


if __name__ == "__main__":
    main()
