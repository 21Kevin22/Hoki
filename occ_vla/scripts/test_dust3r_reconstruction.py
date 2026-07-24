"""Cheap sanity check (per user request, 2026-07-22): before investing in
any dust3r-based geometric-anchor pipeline (Idea 3 from the novel-view-
synthesis discussion), does dust3r produce a geometrically sane 3D
reconstruction from real occ_vla frames at all? dust3r was trained on
real-world photos; LIBERO's synthetic MuJoCo rendering (procedural wood
texture, flat-shaded geoms) could be far enough out of distribution that
reconstruction quality is unusable here, which would need to be known
before building anything on top of it.

Uses an already-collected real agentview+wrist pair (same step, same
episode) from data/multiview/t08_heavy/ (2026-07-15 multiview collection)
-- a real occluded moment (t08 moka_pots, HEAVY difficulty), not a
duplicated single image.

Run in .venv_mmada (roma/trimesh/huggingface_hub installed there for
this test), CPU or GPU:
  source .venv_mmada/bin/activate
  python3 scripts/test_dust3r_reconstruction.py
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


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model = AsymmetricCroCo3DStereo.from_pretrained("naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt").to(device)

    filelist = [
        str(FRAME_DIR / f"step{STEP:04d}_agentview.png"),
        str(FRAME_DIR / f"step{STEP:04d}_wrist.png"),
    ]
    print(f"input pair: {filelist}", flush=True)
    imgs = load_images(filelist, size=512, verbose=True)
    pairs = make_pairs(imgs, scene_graph="complete", prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=1, verbose=True)

    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PairViewer, verbose=True)

    rgbimg = scene.imgs
    depths = [d.detach().cpu().numpy() if torch.is_tensor(d) else d for d in scene.get_depthmaps()]
    confs = [c.detach().cpu().numpy() if torch.is_tensor(c) else c for c in scene.im_conf]
    pts3d = [p.detach().cpu().numpy() if torch.is_tensor(p) else p for p in scene.get_pts3d()]
    focals = scene.get_focals()
    poses = scene.get_im_poses()

    print(f"focals: {focals}", flush=True)
    print(f"poses:\n{poses}", flush=True)
    for i, (d, c, p) in enumerate(zip(depths, confs, pts3d)):
        print(
            f"view {i}: depth range [{d.min():.4f}, {d.max():.4f}], "
            f"conf range [{c.min():.4f}, {c.max():.4f}], mean_conf={c.mean():.4f}, "
            f"pts3d shape={p.shape}, pts3d range x[{p[...,0].min():.3f},{p[...,0].max():.3f}] "
            f"y[{p[...,1].min():.3f},{p[...,1].max():.3f}] z[{p[...,2].min():.3f},{p[...,2].max():.3f}]",
            flush=True,
        )

    # visual contact sheet: rgb | depth (jet) | confidence (jet), per view, stacked
    import matplotlib.pyplot as plt  # noqa: PLC0415

    cmap = plt.get_cmap("jet")
    tiles = []
    for i in range(len(rgbimg)):
        rgb = (np.asarray(rgbimg[i]) * 255).astype(np.uint8) if np.asarray(rgbimg[i]).max() <= 1.0 else np.asarray(rgbimg[i]).astype(np.uint8)
        d_norm = (depths[i] - depths[i].min()) / (depths[i].max() - depths[i].min() + 1e-8)
        d_rgb = (cmap(d_norm)[..., :3] * 255).astype(np.uint8)
        c_norm = (confs[i] - confs[i].min()) / (confs[i].max() - confs[i].min() + 1e-8)
        c_rgb = (cmap(c_norm)[..., :3] * 255).astype(np.uint8)
        tiles.extend([rgb, d_rgb, c_rgb])

    h, w = tiles[0].shape[:2]
    sheet = Image.new("RGB", (w * 3, h * len(rgbimg)))
    for i in range(len(rgbimg)):
        for j in range(3):
            tile = Image.fromarray(tiles[i * 3 + j]).resize((w, h))
            sheet.paste(tile, (j * w, i * h))
    sheet.save(OUT_DIR / f"step{STEP:04d}_rgb_depth_conf.png")
    print(f"\nsaved visualization to {OUT_DIR / f'step{STEP:04d}_rgb_depth_conf.png'}")

    # also dump a point cloud (as .ply) for optional external viewing
    try:
        import trimesh  # noqa: PLC0415

        all_pts = np.concatenate([p.reshape(-1, 3) for p in pts3d], axis=0)
        all_colors = np.concatenate(
            [
                (np.asarray(rgbimg[i]) * 255).astype(np.uint8).reshape(-1, 3)
                if np.asarray(rgbimg[i]).max() <= 1.0
                else np.asarray(rgbimg[i]).astype(np.uint8).reshape(-1, 3)
                for i in range(len(rgbimg))
            ],
            axis=0,
        )
        cloud = trimesh.PointCloud(all_pts, colors=all_colors)
        cloud.export(str(OUT_DIR / f"step{STEP:04d}_pointcloud.ply"))
        print(f"saved point cloud to {OUT_DIR / f'step{STEP:04d}_pointcloud.ply'}")
    except Exception as e:  # noqa: BLE001
        print(f"point cloud export skipped: {e}")


if __name__ == "__main__":
    main()
