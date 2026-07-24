"""Fix for the "double vision" artifact found in
test_dust3r_pastview_grid_select.py (2026-07-22, same day): that script
fed dust3r THREE images (past-agentview, current-agentview, current-
wrist) and independently estimated depth for both agentview captures --
since dust3r's per-call depth estimate for the same static scene isn't
pixel-perfect reproducible, the two near-identical reconstructions of
the same physical region (past vs. current agentview) came out slightly
misaligned, so fusing both produced a ghosted double image.

User proposed fixing this via a MUSt3R-style architecture (common
coordinate system predicted directly, KV-cache memory across frames) --
checked: MUSt3R/MASt3R are not available in this project (not cloned
under third_party/, only vanilla dust3r). Rather than adopting a new,
uncloned external model, this achieves the same underlying goal
(one canonical reconstruction per physical region, no competing
estimates) the simple way: **only include ONE depth estimate for the
static agentview region at all** -- drop current-agentview from the
input entirely (it contributes no information current-agentview
adds to past-agentview: it's the SAME static camera pose, and the
region we actually care about, occluded now, is exactly where
current-agentview has nothing to contribute) and reconstruct from just
(past-agentview, current-wrist) -- 2 images, `PairViewer` mode, exactly
the pattern already validated on bowl_top_drawer.

Run in .venv_mmada, one GPU:
  source .venv_mmada/bin/activate
  python3 scripts/test_dust3r_pastview_dedup_select.py
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

FRAME_DIR = _ROOT / "dense_occlusion_frames" / "mug_in_microwave_v2_with_poses"
PAST_STEP = 70
CUR_STEP = 110
OUT_DIR = _ROOT / "texture_ceiling_probe" / "dust3r_check"
CONF_THRESHOLD_PERCENTILE = 40
RENDER_SIZE = (384, 512)
N_AZIMUTH = 8
N_ELEVATION = 3
N_RADIUS = 2


def cam_to_world(cam_pos, cam_mat_flat):
    R = np.array(cam_mat_flat).reshape(3, 3)
    t = np.array(cam_pos)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def unproject_depth(depth, focal, principal=None):
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
    manifest = json.loads((FRAME_DIR / "manifest.json").read_text())
    past_entry = next(e for e in manifest["steps"] if e["step"] == PAST_STEP)
    cur_entry = next(e for e in manifest["steps"] if e["step"] == CUR_STEP)
    past_av_pose = past_entry["cam_poses"]["agentview"]
    cur_av_pose = cur_entry["cam_poses"]["agentview"]
    cur_wrist_pose = cur_entry["cam_poses"]["robot0_eye_in_hand"]
    print(f"past agentview cam_pos={past_av_pose['cam_pos']}", flush=True)
    print(f"current agentview cam_pos={cur_av_pose['cam_pos']} (should match -- static camera)", flush=True)
    print(f"current wrist cam_pos={cur_wrist_pose['cam_pos']}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AsymmetricCroCo3DStereo.from_pretrained("naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt").to(device)

    # ONLY 2 images -- deliberately drop current-occluded-agentview
    # (see module docstring): it's the same static camera as
    # past-agentview and adds a second, slightly-inconsistent estimate
    # of the same physical region instead of new information.
    filelist = [
        str(FRAME_DIR / f"step{PAST_STEP:04d}_agentview.png"),
        str(FRAME_DIR / f"step{CUR_STEP:04d}_wrist.png"),
    ]
    imgs = load_images(filelist, size=512, verbose=True)
    pairs = make_pairs(imgs, scene_graph="complete", prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=1, verbose=True)
    # 2 images -> PairViewer mode (same as the bowl_top_drawer success);
    # only the PER-VIEW depth output is used below, not dust3r's own
    # relative-pose estimate.
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PairViewer, verbose=True)

    rgbimg = scene.imgs
    depths = [d.detach().cpu().numpy() for d in scene.get_depthmaps()]
    confs_list = [c.detach().cpu().numpy() for c in scene.im_conf]
    dust3r_focals = scene.get_focals().detach().cpu().numpy().reshape(-1)  # (N,1) or (N,) -> flat (N,)
    print(f"focals: {dust3r_focals}", flush=True)

    true_poses = [
        cam_to_world(past_av_pose["cam_pos"], past_av_pose["cam_mat"]),
        cam_to_world(cur_wrist_pose["cam_pos"], cur_wrist_pose["cam_mat"]),
    ]
    view_names = ["past_agentview", "cur_wrist"]
    all_pts_list, all_colors_list, all_confs_list = [], [], []
    for i in range(2):
        local_pts = unproject_depth(depths[i], float(dust3r_focals[i]))
        local_pts_mj = local_pts * np.array([1.0, -1.0, -1.0])
        world_pts = (true_poses[i][:3, :3] @ local_pts_mj.T).T + true_poses[i][:3, 3]
        all_pts_list.append(world_pts)
        all_colors_list.append(np.asarray(rgbimg[i]).reshape(-1, 3))
        all_confs_list.append(confs_list[i].reshape(-1))
        print(f"view {i} ({view_names[i]}): depth range [{depths[i].min():.3f},{depths[i].max():.3f}] "
              f"mean_conf={confs_list[i].mean():.3f}", flush=True)

    all_pts = np.concatenate(all_pts_list, axis=0)
    all_colors = np.concatenate(all_colors_list, axis=0)
    all_confs = np.concatenate(all_confs_list, axis=0)
    conf_thr = np.percentile(all_confs, CONF_THRESHOLD_PERCENTILE)

    # target centroid from the PAST agentview view (view 0) specifically --
    # the one that actually saw the target clearly
    past_conf = all_confs_list[0]
    past_top = past_conf >= np.percentile(past_conf, 90)
    target_centroid = all_pts_list[0][past_top].mean(axis=0)
    target_radius = 1.5 * np.linalg.norm(all_pts_list[0][past_top] - target_centroid, axis=1).std()
    target_mask = np.linalg.norm(all_pts - target_centroid, axis=1) <= target_radius
    print(f"\ntarget centroid (from past agentview)={target_centroid}, radius={target_radius:.4f}, "
          f"target pts={target_mask.sum()}", flush=True)

    ref_pos = np.array(cur_av_pose["cam_pos"])
    ref_to_target = np.linalg.norm(ref_pos - target_centroid)
    grid = {}
    for ir, radius_scale in enumerate(np.linspace(0.7, 1.3, N_RADIUS)):
        for ia, az in enumerate(np.linspace(0, 2 * np.pi, N_AZIMUTH, endpoint=False)):
            for ie, el in enumerate(np.linspace(-0.3, 0.6, N_ELEVATION)):
                r = ref_to_target * radius_scale
                offset = r * np.array([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)])
                eye = target_centroid + offset
                grid[f"az{ia}_el{ie}_r{ir}"] = eye
    grid["cur_agentview_orig"] = np.array(cur_av_pose["cam_pos"])
    grid["cur_wrist_orig"] = np.array(cur_wrist_pose["cam_pos"])
    grid["past_agentview_orig"] = np.array(past_av_pose["cam_pos"])

    # angular-distance penalty (fix for the coverage-gap finding above):
    # S_occ alone doesn't penalize a candidate that reveals a facet no
    # real camera ever observed -- it only scores points that DID land.
    # Real coverage is fundamentally bounded by the real cameras' own
    # viewing directions, so penalize candidates whose viewing direction
    # (eye -> target) is angularly far from BOTH real cameras'.
    real_dirs = [
        (target_centroid - np.array(past_av_pose["cam_pos"])),
        (target_centroid - np.array(cur_wrist_pose["cam_pos"])),
    ]
    real_dirs = [d / np.linalg.norm(d) for d in real_dirs]

    def angle_penalty(eye):
        cand_dir = (target_centroid - eye)
        cand_dir = cand_dir / np.linalg.norm(cand_dir)
        min_angle = min(np.arccos(np.clip(cand_dir @ rd, -1.0, 1.0)) for rd in real_dirs)
        return min_angle  # radians; 0 = exactly matches a real camera's own direction

    focal_v = float(np.mean(dust3r_focals))
    results = []
    for name, eye in grid.items():
        cam2world = look_at(eye, target_centroid)
        img, s_occ, coverage = s_occ_score(all_pts, all_colors, all_confs, target_mask, cam2world, focal_v, RENDER_SIZE, conf_thr)
        ang = angle_penalty(eye)
        # combined score: S_occ dominates when angle is small (real
        # camera-adjacent), angle takes over once we're far enough that
        # coverage gaps become likely regardless of what S_occ measured
        combined = s_occ + (ang / np.pi)  # ang/pi in [0,1], same scale as s_occ
        results.append({"name": name, "s_occ": s_occ, "coverage": coverage, "angle_deg": np.degrees(ang), "combined": combined, "img": img})

    results.sort(key=lambda r: r["combined"])
    print(f"\n{len(results)} grid candidates scored (sorted by S_occ + angle penalty)", flush=True)
    for r in results[:8]:
        print(f"  {r['name']}: S_occ={r['s_occ']:.3f} angle={r['angle_deg']:.1f}deg combined={r['combined']:.3f} coverage={r['coverage']:.3f}", flush=True)
    print("  ...", flush=True)
    for r in results[-3:]:
        print(f"  {r['name']}: S_occ={r['s_occ']:.3f} angle={r['angle_deg']:.1f}deg combined={r['combined']:.3f} coverage={r['coverage']:.3f}", flush=True)

    best = results[0]
    print(f"\nbest (lowest combined score): {best['name']} (S_occ={best['s_occ']:.3f}, angle={best['angle_deg']:.1f}deg)", flush=True)

    h, w = RENDER_SIZE
    real_past_av = np.asarray(Image.fromarray((np.asarray(rgbimg[0]) * 255).astype(np.uint8)).resize((w, h)))
    real_wrist = np.asarray(Image.fromarray((np.asarray(rgbimg[1]) * 255).astype(np.uint8)).resize((w, h)))
    # current occluded agentview: real image loaded directly, display-only
    # (not fed into reconstruction -- that's the whole point of this variant)
    real_cur_av = np.asarray(Image.open(FRAME_DIR / f"step{CUR_STEP:04d}_agentview.png").convert("RGB").resize((w, h)))
    top4 = results[:4]
    tiles = [real_past_av, real_cur_av, real_wrist] + [r["img"] for r in top4]
    sheet = Image.new("RGB", (w * len(tiles), h))
    for i, t in enumerate(tiles):
        sheet.paste(Image.fromarray(t), (i * w, 0))
    sheet.save(OUT_DIR / f"mug_step{CUR_STEP:04d}_angle_scored_sheet.png")
    top4_labels = ", ".join(f"{r['name']}(Socc={r['s_occ']:.2f},ang={r['angle_deg']:.0f}d)" for r in top4)
    print(f"\nsaved: past_agentview | cur_agentview(occluded, display-only) | cur_wrist | {top4_labels} "
          f"-> {OUT_DIR / f'mug_step{CUR_STEP:04d}_angle_scored_sheet.png'}")

    # save the winning candidate separately, at 224x224 (matching
    # obs.base_image / pi0.5's expected resolution) for the pi0.5
    # injection test next
    best_224 = np.asarray(Image.fromarray(best["img"]).resize((224, 224)))
    Image.fromarray(best_224).save(OUT_DIR / f"mug_step{CUR_STEP:04d}_best_recovered_224.png")
    print(f"saved winning candidate at 224x224 for pi0.5 injection: "
          f"{OUT_DIR / f'mug_step{CUR_STEP:04d}_best_recovered_224.png'}")


if __name__ == "__main__":
    main()
