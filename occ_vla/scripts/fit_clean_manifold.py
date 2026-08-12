"""
fit_clean_manifold.py

CLI wrapper around clean_manifold_detector.fit(): fits the PCA+Mahalanobis
"healthy manifold" detector on failure_probe_data_*_clean/ episodes only,
then reports score distributions on held-out clean data AND (if given)
occluded (wrist_partial) data -- a sanity check that occluded activations
actually score as more anomalous BEFORE trusting any threshold derived
from this for run_dynamic_gating_eval.py.

Run with the openvla-oft conda env:
  python scripts/fit_clean_manifold.py \
    --clean-dirs failure_probe_data_moka_clean failure_probe_data_mug_clean \
    --occluded-dirs failure_probe_data_moka failure_probe_data_mug \
    --n-components 32 --out clean_manifold.npz
"""

import argparse
import os
import sys

import numpy as np

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)

import clean_manifold_detector as cmd  # noqa: E402
from train_failure_probe import load_episodes, resolve_dir, split_episodes  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-dirs", nargs="+", required=True)
    parser.add_argument("--occluded-dirs", nargs="*", default=[],
                         help="optional -- only used for the printed separation sanity check, not fitting")
    parser.add_argument("--n-components", type=int, default=32)
    parser.add_argument("--val-frac", type=float, default=0.2,
                         help="held-out clean episodes for the sanity check, excluded from fitting")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="clean_manifold.npz")
    args = parser.parse_args()

    clean_episodes = []
    for d in args.clean_dirs:
        clean_episodes += load_episodes(resolve_dir(d), label=0)
    train_eps, val_eps = split_episodes(clean_episodes, args.val_frac, args.seed)

    X_train = np.concatenate([ep["activations"] for ep in train_eps], axis=0)
    params = cmd.fit(X_train, n_components=args.n_components)
    cmd.save(params, args.out)
    print(f"Fit on {len(train_eps)} clean episodes ({len(X_train)} rows, n_components={args.n_components}), saved to {args.out}")

    X_val = np.concatenate([ep["activations"] for ep in val_eps], axis=0)
    val_scores = cmd.score(params, X_val)
    print(f"Held-out CLEAN scores (n={len(val_scores)}): mean={val_scores.mean():.2f} "
          f"median={np.median(val_scores):.2f} p95={np.percentile(val_scores, 95):.2f} "
          f"p99={np.percentile(val_scores, 99):.2f} max={val_scores.max():.2f}")

    if args.occluded_dirs:
        occ_episodes = []
        for d in args.occluded_dirs:
            occ_episodes += load_episodes(resolve_dir(d), label=1)
        X_occ = np.concatenate([ep["activations"] for ep in occ_episodes], axis=0)
        occ_scores = cmd.score(params, X_occ)
        print(f"OCCLUDED (wrist_partial) scores (n={len(occ_scores)}): mean={occ_scores.mean():.2f} "
              f"median={np.median(occ_scores):.2f} p05={np.percentile(occ_scores, 5):.2f} min={occ_scores.min():.2f}")

        from sklearn.metrics import roc_auc_score
        y = np.concatenate([np.zeros(len(val_scores)), np.ones(len(occ_scores))])
        s = np.concatenate([val_scores, occ_scores])
        auc = roc_auc_score(y, s)
        print(f"\nAUC of raw (unsupervised) Mahalanobis score, held-out-clean vs occluded: {auc:.4f}")
        print(f"Suggested threshold candidates (percentile of held-out CLEAN scores): "
              f"p95={np.percentile(val_scores, 95):.2f}  p99={np.percentile(val_scores, 99):.2f}")


if __name__ == "__main__":
    main()
