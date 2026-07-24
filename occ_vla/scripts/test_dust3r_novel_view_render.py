"""Cheap follow-up (user request, 2026-07-22): can we render a virtual
viewpoint from dust3r's reconstructed point cloud, chosen specifically to
avoid the arm's current occlusion, without any generative model?

This is NOT novel-content generation -- it can only show geometry that
was actually observed by at least one real camera (agentview or wrist)
at this instant. It reprojects real, already-captured 3D points into a
new 2D frame via simple confidence-filtered point splatting (nearest-
point z-buffer, small disk radius per point to reduce gaps) -- no
neural rendering, no 3DGS, reusing the same pinhole-projection math
style already validated in pklp/pixel_to_action.py's CameraProjector.

Uses the same step40 agentview+wrist pair validated in
test_dust3r_reconstruction.py (high confidence, target clearly visible
in both real views) -- this run specifically demonstrates the mechanism
(does splatting produce a coherent, useful image at all in this domain),
not yet the actual occluded case (no agentview-occluded/wrist-visible
pair was available at this episode's 20-step sampling granularity --
see CLAUDE.md).

Run in .venv_mmada, one GPU:
  source .venv_mmada/bin/activate
  python3 scripts/test_dust3r_novel_view_render.py
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
CONF_THRESHOLD_PERCENTILE = 40  # drop the lowest-confidence 40% of points before splatting
RENDER_SIZE = (384, 512)  # match dust3r's own output resolution for this pair


def look_at(eye, target, up=np.array([0.0, -1.0, 0.0])):
    """Standard look-at camera-to-world pose (4x4), camera looking from
    `eye` toward `target`."""
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


def splat_render(pts3d_world, colors, confs, cam2world, focal, image_size, conf_thr):
    h, w = image_size
    world2cam = np.linalg.inv(cam2world)
    pts_h = np.concatenate([pts3d_world, np.ones((pts3d_world.shape[0], 1))], axis=1)
    pts_cam = (world2cam @ pts_h.T).T[:, :3]

    keep = (confs >= conf_thr) & (pts_cam[:, 2] > 1e-4)
    pts_cam, cols = pts_cam[keep], colors[keep]

    cx, cy = w / 2, h / 2
    u = (pts_cam[:, 0] / pts_cam[:, 2]) * focal + cx
    v = (pts_cam[:, 1] / pts_cam[:, 2]) * focal + cy
    z = pts_cam[:, 2]

    canvas = np.zeros((h, w, 3), dtype=np.float64)
    zbuf = np.full((h, w), np.inf)
    ui, vi = np.round(u).astype(int), np.round(v).astype(int)
    radius = 1  # splat a (2*radius+1)^2 disk per point to reduce gaps from sparse points
    for du in range(-radius, radius + 1):
        for dv in range(-radius, radius + 1):
            uu, vv = ui + du, vi + dv
            valid = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
            uu_v, vv_v, z_v, c_v = uu[valid], vv[valid], z[valid], cols[valid]
            # sort far-to-near so nearer points overwrite farther ones when iterated in order
            order = np.argsort(-z_v)
            for idx in order:
                if z_v[idx] < zbuf[vv_v[idx], uu_v[idx]]:
                    zbuf[vv_v[idx], uu_v[idx]] = z_v[idx]
                    canvas[vv_v[idx], uu_v[idx]] = c_v[idx]
    coverage = float((zbuf < np.inf).mean())
    return (canvas * 255).astype(np.uint8) if canvas.max() <= 1.0 else canvas.astype(np.uint8), coverage


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
    confs = [c.detach().cpu().numpy() for c in scene.im_conf]
    poses = scene.get_im_poses().detach().cpu().numpy()  # cam2world, view1 (wrist) is identity
    focals = scene.get_focals().detach().cpu().numpy()

    all_pts = np.concatenate([p.reshape(-1, 3) for p in pts3d], axis=0)
    all_colors = np.concatenate([np.asarray(im).reshape(-1, 3) for im in rgbimg], axis=0)
    all_confs = np.concatenate([c.reshape(-1) for c in confs], axis=0)
    conf_thr = np.percentile(all_confs, CONF_THRESHOLD_PERCENTILE)
    print(f"total points={len(all_pts)}, conf threshold (p{CONF_THRESHOLD_PERCENTILE})={conf_thr:.3f}", flush=True)

    # target region centroid (highest-confidence points), used to aim virtual cameras
    top_mask = all_confs >= np.percentile(all_confs, 90)
    target_centroid = all_pts[top_mask].mean(axis=0)
    print(f"approx target centroid (world/wrist frame): {target_centroid}", flush=True)

    wrist_pos = poses[1][:3, 3]
    agentview_pos = poses[0][:3, 3]
    print(f"wrist cam pos: {wrist_pos}, agentview cam pos: {agentview_pos}", flush=True)

    # candidate virtual viewpoints: offset laterally from the wrist camera
    # (which already had a clear, high-confidence view) toward a few
    # different angles around the target, all looking at the target centroid
    baseline = np.linalg.norm(agentview_pos - wrist_pos)
    candidates = {
        "wrist_orig": (wrist_pos, target_centroid),
        "offset_left": (wrist_pos + np.array([baseline * 0.6, 0, 0]), target_centroid),
        "offset_up": (wrist_pos + np.array([0, -baseline * 0.6, baseline * 0.2]), target_centroid),
        "offset_back": (wrist_pos + (wrist_pos - agentview_pos) * 0.8, target_centroid),
    }

    focal_v = float(np.mean(focals))
    renders = {}
    for name, (eye, target) in candidates.items():
        cam2world = look_at(eye, target)
        img, coverage = splat_render(all_pts, all_colors, all_confs, cam2world, focal_v, RENDER_SIZE, conf_thr)
        renders[name] = img
        print(f"{name}: coverage={coverage:.3f}", flush=True)
        Image.fromarray(img).save(OUT_DIR / f"step{STEP:04d}_novelview_{name}.png")

    # contact sheet: real agentview | real wrist | 4 virtual renders
    h, w = RENDER_SIZE
    real_a = np.asarray(Image.fromarray((np.asarray(rgbimg[0]) * 255).astype(np.uint8)).resize((w, h)))
    real_w = np.asarray(Image.fromarray((np.asarray(rgbimg[1]) * 255).astype(np.uint8)).resize((w, h)))
    tiles = [real_a, real_w] + [renders[k] for k in candidates]
    sheet = Image.new("RGB", (w * len(tiles), h))
    for i, t in enumerate(tiles):
        sheet.paste(Image.fromarray(t), (i * w, 0))
    sheet.save(OUT_DIR / f"step{STEP:04d}_novelview_sheet.png")
    print(f"\nsaved contact sheet (real agentview | real wrist | {' | '.join(candidates)}): "
          f"{OUT_DIR / f'step{STEP:04d}_novelview_sheet.png'}")


if __name__ == "__main__":
    main()
