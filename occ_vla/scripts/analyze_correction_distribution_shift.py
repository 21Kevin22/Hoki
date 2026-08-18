"""
analyze_correction_distribution_shift.py

Per user request (2026-08-18): does the oracle mid-layer correction push
OpenVLA-OFT's internal representation out of what it normally operates on?

Two comparisons, both against the same reference (real, plain `libero_10`
baseline rollouts of the matched task -- see
collect_feature_distribution_reference.py):

  (A) the injected patch_clean tensor itself (keys "dino"/"siglip" in
      --save-oracle-features-dir's .npz files) -- expected to look
      IN-distribution, since it's computed from a real, un-occluded image
      through the model's own featurizer, same as any reference sample.
  (B) the final representation after patch_clean has been carried through
      the remaining transformer blocks alongside the rest of the (still
      occluded) sequence (keys "dino_final"/"siglip_final") -- the thing
      that actually reaches the action head. This is where a real
      distribution shift, if any, should show up: the model was never
      trained on token sequences mixing content from two different
      points in time/occlusion-state.

Method: mean-pool each sample's patches to one D-dim vector (same pooling
for reference and test, per feature key, so the two sides are structurally
comparable), fit PCA (whitened) + a Gaussian on the REFERENCE set only,
score both reference (held-out via leave-one-out-ish resubstitution -- n
is small, not split further) and test samples by Mahalanobis distance in
that reduced space. A real shift shows up as test samples scoring
systematically higher (further from the reference mean) than the
reference set's own internal spread.

Usage:
  python analyze_correction_distribution_shift.py \\
      --reference-dir feature_distribution_reference_task1 \\
      --test-dir oracle_features_task1_distshift_test
"""
import argparse
import glob
import os

import numpy as np


FEATURE_KEYS = ["dino", "siglip", "dino_final", "siglip_final"]


def load_pooled(npz_dir, key):
    """Loads every .npz in npz_dir, mean-pools the given key's patch
    tensor (shape [1, N_patches, D]) over patches -> one D-dim vector per
    file. Skips files missing this key (e.g. reference .npz always has all
    4 keys; older test .npz saved before the _final addition would only
    have dino/siglip -- skip those gracefully rather than crash)."""
    vecs = []
    for path in sorted(glob.glob(os.path.join(npz_dir, "*.npz"))):
        d = np.load(path)
        if key not in d:
            continue
        arr = d[key]  # [1, N_patches, D]
        vecs.append(arr.reshape(-1, arr.shape[-1]).mean(axis=0))
    return np.stack(vecs, axis=0) if vecs else np.zeros((0, 0))


def mahalanobis_scores(ref, test, n_components=None):
    """PCA (via SVD on the mean-centered reference) + Mahalanobis distance
    in the reduced space, matching the same small-n/high-dim technique
    already used elsewhere in this codebase family for exactly this
    situation (can't invert a D x D covariance with D >> n_ref).
    Returns (ref_scores, test_scores, n_components_used)."""
    n_ref, d = ref.shape
    if n_components is None:
        n_components = max(1, min(30, n_ref - 1, d))
    mu = ref.mean(axis=0)
    ref_c = ref - mu
    # SVD-based PCA: ref_c = U S Vt, components = Vt[:n_components]
    U, S, Vt = np.linalg.svd(ref_c, full_matrices=False)
    comps = Vt[:n_components]
    ref_proj = ref_c @ comps.T
    var = ref_proj.var(axis=0, ddof=1)
    var = np.maximum(var, 1e-8)  # guard degenerate directions

    def score(x):
        x_proj = (x - mu) @ comps.T
        return np.sqrt(((x_proj ** 2) / var).sum(axis=1))

    return score(ref), score(test), n_components


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--test-dir", required=True)
    args = parser.parse_args()

    print(f"reference: {args.reference_dir}   test: {args.test_dir}\n")
    for key in FEATURE_KEYS:
        ref = load_pooled(args.reference_dir, key)
        test = load_pooled(args.test_dir, key)
        if ref.shape[0] < 3 or test.shape[0] < 1:
            print(f"[{key}] insufficient samples (ref={ref.shape[0]}, test={test.shape[0]}) -- skipping")
            continue
        ref_scores, test_scores, ncomp = mahalanobis_scores(ref, test)
        print(f"[{key}] ref n={ref.shape[0]} (dim={ref.shape[1]}, {ncomp} PCA comps)  test n={test.shape[0]}")
        print(f"    reference Mahalanobis: mean={ref_scores.mean():.2f} median={np.median(ref_scores):.2f} "
              f"p90={np.percentile(ref_scores, 90):.2f} max={ref_scores.max():.2f}")
        print(f"    test      Mahalanobis: mean={test_scores.mean():.2f} median={np.median(test_scores):.2f} "
              f"p90={np.percentile(test_scores, 90):.2f} max={test_scores.max():.2f}")
        # rank-based separation proxy (0.5 = no separation, matching the
        # same diagnostic convention already used elsewhere in this
        # codebase family for exactly this kind of held-out-vs-test check)
        combined = np.concatenate([ref_scores, test_scores])
        ranks = combined.argsort().argsort()
        test_ranks = ranks[len(ref_scores):]
        separation = test_ranks.mean() / max(len(combined) - 1, 1)
        print(f"    separation proxy (0.5=none, 1.0=test always further than ref): {separation:.3f}\n")


if __name__ == "__main__":
    main()
