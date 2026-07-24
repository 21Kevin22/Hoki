"""PCA analysis of the mug_in_microwave action-manifold data collected
by scripts/collect_action_manifold_data.py (2026-07-20): does the
ungated-blend's collapsed action land off the distribution of pi0.5's
own typical actions? sklearn isn't installed in this environment, so
PCA is done directly via SVD on the mean-centered data (mathematically
identical to sklearn.decomposition.PCA) rather than adding a new
dependency for it.

Run: python3 scripts/analyze_action_manifold.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = _ROOT / "action_manifold_data.json"
OUT_PATH = _ROOT / "action_manifold_pca.png"


def pca_fit(x: np.ndarray, n_components: int = 2):
    mean = x.mean(axis=0)
    centered = x - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:n_components]
    return mean, components


def pca_transform(x: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    return (x - mean) @ components.T


def main():
    data = json.loads(DATA_PATH.read_text())
    background = np.array(data["background_actions"])  # (N, 7)
    events = data["conflict_events"]

    mean, components = pca_fit(background, n_components=2)
    bg_2d = pca_transform(background, mean, components)

    vla_raw = np.array([e["vla_raw"] for e in events])
    blended = np.array([e["blended"] for e in events])
    cos_angles = np.array([e["cos_angle"] if e["cos_angle"] is not None else np.nan for e in events])

    vla_2d = pca_transform(vla_raw, mean, components)
    blended_2d = pca_transform(blended, mean, components)

    agree_mask = cos_angles > 0
    conflict_mask = cos_angles <= 0

    # The background cloud turned out bimodal (see the saved PNG): a
    # broad left cluster (general reach/approach motion) and a tight
    # right cluster (PC1 > 1.0 -- the near-target/occlusion-adjacent
    # phase, where vla_raw's occluded-step actions also land). The
    # GLOBAL centroid sits in the empty gap between the two, so
    # distance-from-global-centroid is a misleading off-manifold proxy
    # here -- it says nothing about whether a point is inside or outside
    # the *locally relevant* cluster. Use the right cluster's own
    # centroid/spread instead, since that's the actual neighborhood
    # occluded-step actions belong to.
    right_cluster = bg_2d[bg_2d[:, 0] > 1.0]
    right_centroid = right_cluster.mean(axis=0)
    right_std = right_cluster.std(axis=0)
    right_dist = np.linalg.norm((right_cluster - right_centroid) / right_std, axis=1)
    vla_dist = np.linalg.norm((vla_2d - right_centroid) / right_std, axis=1)
    blended_dist = np.linalg.norm((blended_2d - right_centroid) / right_std, axis=1)

    print(f"background actions: n={len(background)}, explained spread (std of PC1,PC2): {bg_2d.std(axis=0)}")
    print(f"right (near-target) cluster: n={len(right_cluster)}, own std={right_std}")
    print(f"right-cluster points' own mean normalized distance from their centroid: {right_dist.mean():.3f} (+-{right_dist.std():.3f})")
    print(f"conflict events: n={len(events)} (agree cos>0: {agree_mask.sum()}, conflict cos<=0: {conflict_mask.sum()})")
    print(f"vla_raw (pre-blend) mean normalized distance from right-cluster centroid: {vla_dist.mean():.3f} (+-{vla_dist.std():.3f})")
    print(f"blended, agree subset, mean normalized distance: {blended_dist[agree_mask].mean() if agree_mask.any() else float('nan'):.3f}")
    print(f"blended, conflict subset, mean normalized distance: {blended_dist[conflict_mask].mean() if conflict_mask.any() else float('nan'):.3f}")

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 8))
    for ax in (ax0, ax1):
        ax.scatter(bg_2d[:, 0], bg_2d[:, 1], s=8, alpha=0.25, color="tab:gray", label=f"background (n={len(background)}, pi0.5's own typical actions)")
        ax.scatter(vla_2d[:, 0], vla_2d[:, 1], s=20, alpha=0.6, color="tab:blue", marker="^", label=f"vla_raw pre-blend, occluded steps (n={len(events)})")
        if agree_mask.any():
            ax.scatter(blended_2d[agree_mask, 0], blended_2d[agree_mask, 1], s=30, alpha=0.8, color="tab:green", marker="o", label=f"blended, cos_angle>0 agree (n={agree_mask.sum()})")
        if conflict_mask.any():
            ax.scatter(blended_2d[conflict_mask, 0], blended_2d[conflict_mask, 1], s=40, alpha=0.9, color="tab:red", marker="x", label=f"blended, cos_angle<=0 conflict (n={conflict_mask.sum()})")
        ax.scatter(*right_centroid, s=150, color="black", marker="*", label="near-target cluster centroid", zorder=5)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    ax0.legend(loc="best", fontsize=8)
    ax0.set_title("full view")
    pad = 0.15
    ax1.set_xlim(right_centroid[0] - 4 * right_std[0], right_centroid[0] + 4 * right_std[0])
    ax1.set_ylim(right_centroid[1] - 4 * right_std[1] - pad, right_centroid[1] + 4 * right_std[1] + pad)
    ax1.set_title("zoomed on the near-target (occlusion-adjacent) cluster")
    fig.suptitle("mug_in_microwave action manifold (PCA of 7-dim actions)\nbackground = pi0.5 baseline rollouts; red = ungated-blend conflict collapse")
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"wrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
