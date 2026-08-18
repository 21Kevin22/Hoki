"""
occlusion_classifier.py

The real-time trigger signal for run_dynamic_gating_eval.py: a
StandardScaler + LogisticRegression classifier over OpenVLA-OFT's
per-call hidden state, distinguishing clean (unoccluded) from wrist_partial
(occluded) activations. This is the same model family as
train_failure_probe.py's "Test 1" (val AUC=0.9997) -- promoted from a
research/evaluation probe to a deployed gating component after the
PCA+Mahalanobis anomaly detector (clean_manifold_detector.py) was found
to invert on this exact failure mode: occlusion COLLAPSES activation
variance (the network's response flattens when visual information is
lost) rather than expanding it away from the clean manifold, so
distance-from-clean-centroid detectors score occluded activations as
MORE typical, not less (AUC=0.12-0.20, backwards). A supervised
classifier learns the actual discriminative hyperplane directly and
isn't fooled by which class has more spread.

fit(): pools ALL given clean+occluded rows (no held-out split -- this is
now a deployed component whose job is to trigger correctly on live data,
not a research question being evaluated for generalization; the
generalization question was already answered by train_failure_probe.py).
score(): P(occluded) in [0, 1] for a new activation.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def fit(clean_activations, occluded_activations, C=0.01):
    X = np.concatenate([clean_activations, occluded_activations], axis=0)
    y = np.concatenate([np.zeros(len(clean_activations)), np.ones(len(occluded_activations))])
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(scaler.transform(X), y)
    return {
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "coef": clf.coef_[0],
        "intercept": clf.intercept_,
    }


def save(params, path):
    np.savez(path, **params)


def load(path):
    d = np.load(path)
    return {k: d[k] for k in d.files}


def score(params, activations):
    """activations: (D,) or (N, D). Returns P(occluded) in [0, 1], scalar or (N,)."""
    single = activations.ndim == 1
    X = activations.reshape(1, -1) if single else activations
    Xs = (X - params["scaler_mean"]) / params["scaler_scale"]
    z = Xs @ params["coef"] + params["intercept"][0]
    p = 1.0 / (1.0 + np.exp(-z))
    return float(p[0]) if single else p
