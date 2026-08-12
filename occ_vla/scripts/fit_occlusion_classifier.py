"""
fit_occlusion_classifier.py

CLI wrapper: fits occlusion_classifier on ALL failure_probe_data_*_clean/
(y=0) and failure_probe_data_{moka,mug}/ i.e. wrist_partial (y=1)
episodes, prints a quick episode-held-out sanity AUC (not the final
generalization claim -- that's train_failure_probe.py's job, already
done, val AUC=0.9997), then saves the fitted scaler+classifier for
run_dynamic_gating_eval.py to load as its real-time trigger.

Run with the openvla-oft conda env:
  python scripts/fit_occlusion_classifier.py \
    --clean-dirs failure_probe_data_moka_clean failure_probe_data_mug_clean \
    --occluded-dirs failure_probe_data_moka failure_probe_data_mug \
    --out occlusion_classifier.npz
"""

import argparse
import os
import sys

import numpy as np

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)

import occlusion_classifier as oc  # noqa: E402
from train_failure_probe import load_episodes, resolve_dir, split_episodes  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-dirs", nargs="+", required=True)
    parser.add_argument("--occluded-dirs", nargs="+", required=True)
    parser.add_argument("--C", type=float, default=0.01)
    parser.add_argument("--out", default="occlusion_classifier.npz")
    args = parser.parse_args()

    clean_eps, occ_eps = [], []
    for d in args.clean_dirs:
        clean_eps += load_episodes(resolve_dir(d), label=0)
    for d in args.occluded_dirs:
        occ_eps += load_episodes(resolve_dir(d), label=1)

    # quick episode-held-out sanity check (20%) before fitting on everything
    clean_train, clean_val = split_episodes(clean_eps, 0.2, 0)
    occ_train, occ_val = split_episodes(occ_eps, 0.2, 1)
    X_clean_train = np.concatenate([e["activations"] for e in clean_train], axis=0)
    X_occ_train = np.concatenate([e["activations"] for e in occ_train], axis=0)
    params_check = oc.fit(X_clean_train, X_occ_train, C=args.C)
    X_val = np.concatenate([e["activations"] for e in clean_val] + [e["activations"] for e in occ_val], axis=0)
    y_val = np.concatenate([np.zeros(sum(len(e["activations"]) for e in clean_val)),
                             np.ones(sum(len(e["activations"]) for e in occ_val))])
    from sklearn.metrics import roc_auc_score
    val_auc = roc_auc_score(y_val, oc.score(params_check, X_val))
    print(f"Held-out sanity check (20% episodes, not used in final fit): val AUC={val_auc:.4f}")

    # final fit on everything
    X_clean = np.concatenate([e["activations"] for e in clean_eps], axis=0)
    X_occ = np.concatenate([e["activations"] for e in occ_eps], axis=0)
    params = oc.fit(X_clean, X_occ, C=args.C)
    oc.save(params, args.out)
    print(f"Fit on {len(clean_eps)} clean + {len(occ_eps)} occluded episodes "
          f"({len(X_clean)}+{len(X_occ)} rows), saved to {args.out}")


if __name__ == "__main__":
    main()
