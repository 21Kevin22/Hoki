"""
clean_manifold_detector.py

PCA + Mahalanobis-distance anomaly detector over OpenVLA-OFT's per-call
action-token LLM hidden state (occ_vla, 2026-08-04) -- the "healthy
manifold" a dynamic occlusion-recovery gate scores live activations
against, for run_dynamic_gating_eval.py.

Per user's explicit design choice: a full autoencoder would overfit badly
at this data scale (~60 clean episodes total, 4096-dim activations --
n much smaller than D). PCA(n_components) + a shrinkage-regularized
(Ledoit-Wolf) covariance in the reduced space is the same family of
approach already validated in the sibling pi0.5+MMaDA project's own SMD
gate (control/smd.py: "SVD-based PCA to 64 components before inverting
the covariance, since 283 raw samples can't support inverting a
2048x2048 covariance").

fit(): StandardScaler -> PCA -> LedoitWolf covariance, fit ONLY on
clean/unoccluded rollout activations (never on occluded data -- the
manifold this scores distance from must be the "healthy" one).
score(): squared Mahalanobis distance of a new activation from the clean
manifold's mean, in PCA space. Higher = more anomalous / further from
how a clean rollout's hidden state normally looks.
"""

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def fit(clean_activations, n_components=32):
    """clean_activations: (N, D) float array, pooled across clean episodes."""
    scaler = StandardScaler().fit(clean_activations)
    X = scaler.transform(clean_activations)
    pca = PCA(n_components=n_components).fit(X)
    X_pca = pca.transform(X)
    cov = LedoitWolf().fit(X_pca)
    return {
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "pca_components": pca.components_,
        "pca_mean": pca.mean_,
        "cov_mean": cov.location_,
        "cov_precision": cov.precision_,  # inverse covariance, already shrinkage-regularized
        "n_components": np.array(n_components),
    }


def save(params, path):
    np.savez(path, **params)


def load(path):
    d = np.load(path)
    return {k: d[k] for k in d.files}


def score(params, activations):
    """activations: (D,) or (N, D). Returns a scalar or (N,) array of
    squared Mahalanobis distances from the fitted clean manifold."""
    single = activations.ndim == 1
    X = activations.reshape(1, -1) if single else activations
    X = (X - params["scaler_mean"]) / params["scaler_scale"]
    X_pca = (X - params["pca_mean"]) @ params["pca_components"].T
    diff = X_pca - params["cov_mean"]
    d2 = np.einsum("ij,jk,ik->i", diff, params["cov_precision"], diff)
    return float(d2[0]) if single else d2
