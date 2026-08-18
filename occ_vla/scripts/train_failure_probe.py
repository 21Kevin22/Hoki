"""
train_failure_probe.py

Intervention-gating probe (occ_vla, 2026-08-04): a logistic-regression probe
on OpenVLA-OFT's own action-token final-layer LLM hidden state (the same
per-call activation collect_failure_probe_data.py already captures via
get_vla_action(..., return_hidden_states=True)), trained to distinguish
"wrist camera currently occluded, no correction applied" (y=1, the state a
real deployment would want to detect and hand off to the vjepa predictor)
from "clean, unoccluded rollout" (y=0) -- WITHOUT access to a ground-truth
occlusion mask at inference time.

Reframed from the original per-episode "predict eventual success/failure"
plan (occ_vla, 2026-08-03) after an n=30 re-run confirmed moka_pots'
wrist_partial success rate is genuinely ~0-3% on this machine (1/55 across
all wrist_partial trials this session, task8_wrist_partial_n30.json) --
that makes episode-outcome-as-label degenerate for moka_pots specifically
(single class, no contrast to learn from). Per-step occlusion-vs-clean
labeling sidesteps this: every row in a wrist_partial episode is a real
y=1 example regardless of whether that episode ultimately succeeds.

NOTE on what this probe does and does not test: comparing wrist_partial
against a *clean* (unoccluded) baseline tests whether the model's internal
state reveals "is the wrist camera occluded right now" -- expected to be
learnable, since the raw pixel input genuinely differs between the two
conditions (this is closer to an occlusion detector than a subtle
early-warning/failure-prediction signal). A harder, more interesting
follow-up (not run here) would contrast wrist_partial against
wrist_partial_vjepa, where the raw pixel input is IDENTICAL in both
conditions and only the vjepa correction differs -- that would test
whether the probe can detect "did the correction actually resolve the
occlusion," not just "is a gray patch present."

Splits by EPISODE, not by row, so temporally-correlated activations from
the same rollout never leak across train/val.

Run with the openvla-oft conda env:
  python scripts/train_failure_probe.py \
    --positive-dirs failure_probe_data_moka failure_probe_data_mug \
    --negative-dirs failure_probe_data_moka_clean failure_probe_data_mug_clean \
    --results-path failure_probe_results.json
"""

import argparse
import glob
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
OCC_VLA_ROOT = os.path.dirname(SCRIPTS_DIR)
OFT_ROOT = os.path.join(OCC_VLA_ROOT, "thirdparty/openvla-oft")


def resolve_dir(d):
    """--*-dirs are plain names (e.g. 'failure_probe_data_moka'), which
    collect_failure_probe_data.py actually wrote under OFT_ROOT (its own
    os.chdir target) regardless of caller cwd -- same convention as every
    other --out-dir in this project."""
    if os.path.isabs(d) and os.path.isdir(d):
        return d
    for candidate in (d, os.path.join(OFT_ROOT, d), os.path.join(OCC_VLA_ROOT, d)):
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(f"Could not find data dir '{d}' (looked in cwd, {OFT_ROOT}, {OCC_VLA_ROOT})")


def load_episodes(data_dir, label, max_calls=None):
    """One entry per episode.npz: (activations (n_calls, D), label, episode_id, task_id, success).

    max_calls: if set, truncate to the first N VLA-call rows only -- an
    "early warning" test (does the signal exist before a doomed episode
    has had time to just run long/hit timeout, which is a trivial giveaway
    a failed episode is 65/65 calls while a successful one stops early)."""
    episodes = []
    for path in sorted(glob.glob(os.path.join(data_dir, "episode_*.npz"))):
        d = np.load(path)
        acts = d["activations"].astype(np.float64)
        if max_calls is not None:
            acts = acts[:max_calls]
        episodes.append({
            "activations": acts,
            "label": label,
            "path": path,
            "task_id": int(d["task_id"]),
            "success": bool(d["success"]),
        })
    return episodes


def split_episodes(episodes, val_frac, seed):
    """Stratified-by-task-id 80/20 episode split (not row split)."""
    rng = np.random.default_rng(seed)
    by_task = {}
    for ep in episodes:
        by_task.setdefault(ep["task_id"], []).append(ep)
    train, val = [], []
    for task_id, eps in by_task.items():
        idx = rng.permutation(len(eps))
        n_val = max(1, round(len(eps) * val_frac))
        val_idx = set(idx[:n_val].tolist())
        for i, ep in enumerate(eps):
            (val if i in val_idx else train).append(ep)
    return train, val


def flatten(episodes):
    X = np.concatenate([ep["activations"] for ep in episodes], axis=0)
    y = np.concatenate([np.full(len(ep["activations"]), ep["label"]) for ep in episodes])
    groups = np.concatenate([np.full(len(ep["activations"]), i) for i, ep in enumerate(episodes)])
    return X, y, groups


def per_episode_summary(episodes, probs_by_episode):
    rows = []
    for ep, p in zip(episodes, probs_by_episode):
        rows.append({
            "path": os.path.basename(ep["path"]),
            "label": int(ep["label"]),
            "task_id": ep["task_id"],
            "episode_success": ep["success"],
            "mean_prob": float(np.mean(p)),
            "frac_steps_above_0.5": float(np.mean(p > 0.5)),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive-dirs", nargs="+", required=True,
                         help="occluded (y=1, intervention-needed) episode dirs -- or, with "
                              "--label-from-success, the full pool of dirs to label by outcome")
    parser.add_argument("--negative-dirs", nargs="*", default=[],
                         help="clean (y=0, no-intervention-needed) episode dirs -- ignored if --label-from-success")
    parser.add_argument(
        "--label-from-success", action="store_true",
        help="ignore the fixed positive/negative-dir labeling; instead use each episode's own "
             "'success' field as y (1=succeeded, 0=failed). For within-condition "
             "recovery-quality classification (e.g. wrist_partial_vjepa success vs failure), "
             "as opposed to the default between-condition (occluded vs clean) labeling. "
             "Only --positive-dirs is read in this mode.",
    )
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--C", type=float, default=1.0, help="inverse L2 regularization strength for LogisticRegression")
    parser.add_argument("--results-path", default="failure_probe_results.json")
    parser.add_argument(
        "--max-calls", type=int, default=None,
        help="truncate each episode to its first N VLA-call rows only -- an early-warning "
             "control: a failed/timeout episode is always the max n_calls (e.g. 65) while a "
             "successful one stops early, so using full episodes lets the probe trivially "
             "learn 'this looks late/long' instead of a genuine pre-outcome signal. Set this "
             "to rule that out (e.g. --max-calls 3 tests calls 0-2 of every episode).",
    )
    args = parser.parse_args()

    pos_episodes, neg_episodes = [], []
    if args.label_from_success:
        pool = []
        for d in args.positive_dirs:
            pool += load_episodes(resolve_dir(d), label=None, max_calls=args.max_calls)
        for ep in pool:
            ep["label"] = int(ep["success"])
        pos_episodes = [ep for ep in pool if ep["label"] == 1]
        neg_episodes = [ep for ep in pool if ep["label"] == 0]
        print(f"Labeling by episode outcome (--label-from-success): "
              f"{len(pos_episodes)} succeeded (y=1), {len(neg_episodes)} failed (y=0)")
    else:
        for d in args.positive_dirs:
            pos_episodes += load_episodes(resolve_dir(d), label=1, max_calls=args.max_calls)
        for d in args.negative_dirs:
            neg_episodes += load_episodes(resolve_dir(d), label=0, max_calls=args.max_calls)

    print(f"Loaded {len(pos_episodes)} positive (occluded) episodes, {len(neg_episodes)} negative (clean) episodes")
    for ep in pos_episodes + neg_episodes:
        assert ep["activations"].ndim == 2, ep["path"]

    pos_train, pos_val = split_episodes(pos_episodes, args.val_frac, args.seed)
    neg_train, neg_val = split_episodes(neg_episodes, args.val_frac, args.seed + 1)
    train_episodes = pos_train + neg_train
    val_episodes = pos_val + neg_val
    print(f"Episode split: train={len(train_episodes)} ({len(pos_train)} pos / {len(neg_train)} neg), "
          f"val={len(val_episodes)} ({len(pos_val)} pos / {len(neg_val)} neg)")

    X_train, y_train, _ = flatten(train_episodes)
    X_val, y_val, val_groups = flatten(val_episodes)
    print(f"Row counts: train={len(y_train)} (pos={int(y_train.sum())}), val={len(y_val)} (pos={int(y_val.sum())})")

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)

    clf = LogisticRegression(C=args.C, max_iter=2000, class_weight="balanced")
    clf.fit(X_train_s, y_train)

    val_probs = clf.predict_proba(X_val_s)[:, 1]
    train_probs = clf.predict_proba(X_train_s)[:, 1]

    row_auc = roc_auc_score(y_val, val_probs) if len(set(y_val.tolist())) > 1 else float("nan")
    row_ap = average_precision_score(y_val, val_probs) if len(set(y_val.tolist())) > 1 else float("nan")
    train_row_auc = roc_auc_score(y_train, train_probs)

    # Per-episode aggregate (mean prob across an episode's steps) -- coarser,
    # more deployment-realistic view than a single row's prediction.
    val_probs_by_episode = []
    offset = 0
    for ep in val_episodes:
        n = len(ep["activations"])
        val_probs_by_episode.append(val_probs[offset:offset + n])
        offset += n
    episode_mean_probs = np.array([p.mean() for p in val_probs_by_episode])
    episode_labels = np.array([ep["label"] for ep in val_episodes])
    episode_auc = (roc_auc_score(episode_labels, episode_mean_probs)
                   if len(set(episode_labels.tolist())) > 1 else float("nan"))

    print(f"\n=== Results ===")
    print(f"Row-level  : train AUC={train_row_auc:.4f} | val AUC={row_auc:.4f}  val AP={row_ap:.4f}")
    print(f"Episode-level (mean prob over episode) val AUC={episode_auc:.4f}")

    results = {
        "args": vars(args),
        "n_positive_episodes": len(pos_episodes),
        "n_negative_episodes": len(neg_episodes),
        "n_train_episodes": len(train_episodes),
        "n_val_episodes": len(val_episodes),
        "n_train_rows": int(len(y_train)),
        "n_val_rows": int(len(y_val)),
        "row_level": {"train_auc": float(train_row_auc), "val_auc": float(row_auc), "val_ap": float(row_ap)},
        "episode_level": {"val_auc": float(episode_auc)},
        "val_episode_details": per_episode_summary(val_episodes, val_probs_by_episode),
        "logreg_coef_norm": float(np.linalg.norm(clf.coef_)),
    }
    with open(args.results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {args.results_path}")


if __name__ == "__main__":
    main()
