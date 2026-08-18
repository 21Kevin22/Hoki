"""
plot_covariate_shift.py

Visual evidence for the "trigger classifier domain-mismatch" finding
(2026-08-08, goal_domain_routing_fix_summary.json): PCA projection of
real LLM hidden-state activations (the same feature space
occlusion_classifier.npz/occlusion_classifier_libero_goal.npz score) from
libero_10 (moka_pots + mug_in_microwave) vs. libero_goal (middle_drawer),
clean vs. occluded, to show that libero_goal's activations sit in a
different region of this space entirely -- not just "the occluded class
is harder to separate there", the whole distribution has shifted.

Fits PCA on the POOLED clean-vs-occluded activations from BOTH suites
combined (a shared, task-agnostic 2D basis) rather than fitting a
separate PCA per suite, so the two suites' point clouds are directly
comparable in the same coordinate system -- fitting per-suite PCA bases
would make any positional difference uninterpretable (each basis would
just re-center on its own data by construction).

Also overlays each classifier's real decision boundary (score=0.5
contour) projected into this shared 2D basis, by scoring a dense grid of
points in PC-space after inverse-transforming back to the original
D-dim space -- an approximation (the true boundary is a D-1 dim
hyperplane; this shows where it crosses the 2D subspace the data mostly
lives in), not an exact rendering, but enough to show visually that the
libero_10-fit boundary sits nowhere near the goal points while the
goal-fit boundary actually separates them.

Run with the openvla-oft conda env (needs matplotlib + sklearn, both
already present):
  python scripts/plot_covariate_shift.py --out covariate_shift.png
"""

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
OCC_VLA_ROOT = os.path.dirname(SCRIPTS_DIR)
OFT_ROOT = os.path.join(OCC_VLA_ROOT, "thirdparty/openvla-oft")


def load_activations(data_dir):
    """One row per VLA call, pooled across all episode .npz files in data_dir."""
    rows = []
    for path in sorted(glob.glob(os.path.join(OFT_ROOT, data_dir, "episode_*.npz"))):
        d = np.load(path)
        rows.append(d["activations"].astype(np.float64))
    if not rows:
        raise FileNotFoundError(f"No episode_*.npz found in {os.path.join(OFT_ROOT, data_dir)}")
    return np.concatenate(rows, axis=0)


def load_classifier(path):
    # Matches occlusion_classifier.py's own save() field names exactly
    # (scaler_mean/scaler_scale/coef/intercept) -- verified against that
    # module's fit()/load() before writing this, not guessed.
    d = np.load(path)
    return {"mean": d["scaler_mean"], "scale": d["scaler_scale"], "coef": d["coef"], "intercept": d["intercept"]}


def classifier_score(params, X):
    Xs = (X - params["mean"]) / params["scale"]
    z = Xs @ params["coef"].reshape(-1) + float(params["intercept"][0])
    return 1.0 / (1.0 + np.exp(-z))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="covariate_shift.png")
    args = parser.parse_args()

    print("Loading activations...")
    l10_clean = np.concatenate([
        load_activations("failure_probe_data_moka_clean"),
        load_activations("failure_probe_data_mug_clean"),
    ])
    l10_occ = np.concatenate([
        load_activations("failure_probe_data_moka"),
        load_activations("failure_probe_data_mug"),
    ])
    goal_clean = load_activations("failure_probe_data_goal_task0_clean")
    goal_occ = load_activations("failure_probe_data_goal_task0_occ")

    print(f"  libero_10 clean: {l10_clean.shape}, occluded: {l10_occ.shape}")
    print(f"  libero_goal clean: {goal_clean.shape}, occluded: {goal_occ.shape}")

    # Shared PCA basis, fit on everything pooled -- see module docstring
    # for why per-suite bases would be uninterpretable here.
    all_X = np.concatenate([l10_clean, l10_occ, goal_clean, goal_occ])
    pca = PCA(n_components=2)
    pca.fit(all_X)
    print(f"  PCA explained variance ratio (PC1, PC2): {pca.explained_variance_ratio_}")

    def proj(X):
        return pca.transform(X)

    l10_clean_2d, l10_occ_2d = proj(l10_clean), proj(l10_occ)
    goal_clean_2d, goal_occ_2d = proj(goal_clean), proj(goal_occ)

    # Real classifier decision boundaries, projected into the same 2D basis.
    clf_old = load_classifier(os.path.join(OCC_VLA_ROOT, "occlusion_classifier.npz"))
    clf_new = load_classifier(os.path.join(OCC_VLA_ROOT, "occlusion_classifier_libero_goal.npz"))

    all_2d = np.concatenate([l10_clean_2d, l10_occ_2d, goal_clean_2d, goal_occ_2d])
    pad = 0.15 * (all_2d.max(0) - all_2d.min(0))
    lo, hi = all_2d.min(0) - pad, all_2d.max(0) + pad
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], 220), np.linspace(lo[1], hi[1], 220))
    grid_2d = np.stack([gx.ravel(), gy.ravel()], axis=1)
    grid_D = pca.inverse_transform(grid_2d)  # back to original activation space

    score_old = classifier_score(clf_old, grid_D).reshape(gx.shape)
    score_new = classifier_score(clf_new, grid_D).reshape(gx.shape)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharex=True, sharey=True)

    for ax, score, title in [
        (axes[0], score_old, "occlusion_classifier.npz (fit on libero_10 only)"),
        (axes[1], score_new, "occlusion_classifier_libero_goal.npz (fit on libero_goal)"),
    ]:
        ax.contour(gx, gy, score, levels=[0.5], colors="black", linewidths=2.0, linestyles="--")
        ax.contourf(gx, gy, score, levels=[0.5, 1.0], colors=["#00000000"], alpha=0.0)  # keep axes scale consistent
        ax.scatter(l10_clean_2d[:, 0], l10_clean_2d[:, 1], s=8, alpha=0.35, c="#2E6E8E", label="libero_10 clean")
        ax.scatter(l10_occ_2d[:, 0], l10_occ_2d[:, 1], s=8, alpha=0.35, c="#B23B4A", label="libero_10 occluded")
        ax.scatter(goal_clean_2d[:, 0], goal_clean_2d[:, 1], s=8, alpha=0.55, c="#1E9E7C", marker="^", label="libero_goal clean")
        ax.scatter(goal_occ_2d[:, 0], goal_occ_2d[:, 1], s=8, alpha=0.55, c="#D9821F", marker="^", label="libero_goal occluded")
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].legend(loc="best", fontsize=8, framealpha=0.9)
    fig.suptitle(
        "Covariate shift in the trigger's feature space: libero_goal's activations sit outside\n"
        "libero_10's own distribution -- the libero_10-fit boundary (left, dashed) never separates them,\n"
        "the goal-fit boundary (right, dashed) does.",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
