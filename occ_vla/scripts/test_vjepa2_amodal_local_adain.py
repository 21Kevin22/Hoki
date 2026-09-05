"""Follow-up to global variance-rescaling: implements a LOCAL (spatially-
adaptive) version, grounded in AdaIN (Huang & Belongie, ICCV 2017,
"Arbitrary Style Transfer in Real-time with Adaptive Instance
Normalization") and SPADE (Park et al., CVPR 2019, "Semantic Image
Synthesis with Spatially-Adaptive Normalization") -- both real,
well-established techniques for exactly this problem (make a
synthesized/completed region's local statistics match its SPATIAL
neighborhood, not one global reference, since real scenes have
spatially-varying local statistics -- e.g. table vs. cabinet vs. shadow
regions in this exact frame).

For each occluded token, the reference mean/std is computed from its K
nearest UNOCCLUDED neighbor tokens (spatial k-NN on the 16x16 grid),
not the whole-image context -- still zero new training, a pure
inference-time statistical operation on the real V-JEPA2 predictor's
own output.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
VJEPA2_DIR = REPO_ROOT / "thirdparty" / "vjepa2"
sys.path.insert(0, str(VJEPA2_DIR))
OFT_DIR = REPO_ROOT / "thirdparty" / "openvla-oft"
OUT_DIR = OFT_DIR / "vjepa2_amodal_check"
OUT_DIR.mkdir(exist_ok=True)

FRAME_PATH = OFT_DIR / "distillation_pairs_task1_smoke" / "task1_ep0_t00050_agentview.png"
K_NEIGHBORS = 12


def to_clip(arr_uint8, device, size=256):
    img = Image.fromarray(arr_uint8).resize((size, size))
    arr = (np.asarray(img).astype(np.float32) / 255.0 - 0.5) / 0.5
    frame = torch.from_numpy(arr).permute(2, 0, 1).to(device)
    return torch.stack([frame, frame], dim=1).unsqueeze(0)


def pca_rgb(tokens_np, n_components=3):
    x = tokens_np - tokens_np.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    proj = x @ vt[:n_components].T
    proj -= proj.min(axis=0, keepdims=True)
    denom = proj.max(axis=0, keepdims=True) - proj.min(axis=0, keepdims=True)
    proj = proj / np.clip(denom, 1e-8, None)
    return (proj.reshape(16, 16, n_components) * 255).astype(np.uint8)


def local_adain(pred_at_occ, occluded_idx, z_real_full, unoccluded_idx, k=K_NEIGHBORS):
    """SPADE/AdaIN-style LOCAL rescaling: each occluded token's reference
    mean/std comes from its k nearest UNOCCLUDED tokens on the 16x16
    spatial grid (Euclidean grid distance), not one global reference."""
    coords = np.stack(np.meshgrid(np.arange(16), np.arange(16), indexing="ij"), axis=-1).reshape(-1, 2)
    occ_coords = coords[occluded_idx]  # (K_occ,2)
    unocc_coords = coords[unoccluded_idx]  # (K_ctx,2)
    real_context = z_real_full[unoccluded_idx]  # (K_ctx,1408)

    dists = np.linalg.norm(occ_coords[:, None, :] - unocc_coords[None, :, :], axis=-1)  # (K_occ,K_ctx)
    nn_idx = np.argsort(dists, axis=1)[:, :k]  # (K_occ,k) -- indices INTO unoccluded_idx

    out = pred_at_occ.clone()
    for i in range(pred_at_occ.shape[0]):
        neighbors = real_context[nn_idx[i]]  # (k,1408)
        ref_mean = neighbors.mean(dim=0, keepdim=True)
        ref_std = neighbors.std(dim=0, keepdim=True)
        p = pred_at_occ[i : i + 1]
        p_mean = p.mean(dim=0, keepdim=True)  # single-token: use its OWN cross-channel... see note below
        # occ_vla note: with a single token there's no within-token
        # "std across samples" the way the global version had (K tokens).
        # Standardize using the PREDICTOR'S OWN overall (all occluded
        # tokens) mean/std as the source distribution (same role the
        # global version's pred_mean/pred_std played), then map to this
        # token's LOCAL target reference.
        out[i] = (pred_at_occ[i] - pred_at_occ.mean(dim=0)) / (pred_at_occ.std(dim=0) + 1e-6) * ref_std[0] + ref_mean[0]
    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from run_libero_occluded_oracle_headroom import load_real_vjepa2_base, vjepa2_amodal_complete

    print("[load] real BASE (non-AC) V-JEPA2...")
    encoder, predictor = load_real_vjepa2_base(device)
    print("[load] done")

    arr = np.asarray(Image.open(FRAME_PATH).convert("RGB").resize((256, 256)))
    clip = to_clip(arr, device)
    mask = np.zeros((16, 16), dtype=bool)
    mask[6:13, 2:9] = True
    token_mask_256 = mask.reshape(-1)
    occluded_idx = np.flatnonzero(token_mask_256)
    unoccluded_idx = np.flatnonzero(~token_mask_256)

    with torch.no_grad():
        z_real_full = encoder(clip)[0]

    completed = vjepa2_amodal_complete(encoder, predictor, clip, token_mask_256, device)
    pred_at_occ = completed[occluded_idx]

    local_rescaled = local_adain(pred_at_occ, occluded_idx, z_real_full, unoccluded_idx)
    print(f"[stats] local-AdaIN std: {local_rescaled.float().std().item():.4f} "
          f"(real context std: {z_real_full[unoccluded_idx].float().std().item():.4f})")

    hybrid_local = z_real_full.clone()
    hybrid_local[token_mask_256] = local_rescaled.to(hybrid_local.dtype)

    def upscale(img16):
        return np.array(Image.fromarray(img16).resize((256, 256), Image.NEAREST))

    mask_overlay = arr.copy()
    mask_up = np.array(Image.fromarray((mask.astype(np.uint8) * 255)).resize((256, 256), Image.NEAREST))
    mask_overlay[mask_up > 0] = (0.5 * mask_overlay[mask_up > 0].astype(np.float32) + 0.5 * np.array([255, 0, 0])).astype(np.uint8)

    real_pca = upscale(pca_rgb(z_real_full.float().cpu().numpy()))
    local_pca = upscale(pca_rgb(hybrid_local.float().cpu().numpy()))

    row = np.concatenate([mask_overlay, real_pca, local_pca], axis=1)
    out_path = OUT_DIR / "amodal_local_adain_check.png"
    Image.fromarray(row).save(out_path)
    print(f"[saved] {out_path} (real+mask | real PCA | LOCAL-AdaIN-rescaled completion)")


if __name__ == "__main__":
    main()
