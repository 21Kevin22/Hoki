"""occ_vla addition (2026-08-21), per user's contact-risk-predictor proposal
(their message beginning "Q. どちらのカメラを学習するのか"): builds a
(state, candidate_action) -> "contact within k replan-steps" dataset from
ALL existing task1 result logs, and trains a small numpy-only logistic
regression (no sklearn -- not installed in .venv_openvla_oft and this
venv has no pip to add it) as a v0 feasibility check for the user's
"本命" proposal (contact-risk predictor / safety filter).

DATA HYGIENE (per the user's own explicit warning about which
trajectories must NOT be used, applied here to a DIFFERENT training
target than the one they warned about):
  - `no_collision` / `oracle_no_collision` episodes are EXCLUDED entirely:
    physical collision is disabled for the whole episode, so
    occluder_contact is trivially always False regardless of the actual
    trajectory -- these "contact never happens" labels are an artifact
    of the intervention, not a real physics signal, and would teach the
    classifier "this region is safe" when it is not.
  - `no_collision_after_contact` / `scripted_recovery_after_contact`
    episodes: only the PRE-TRIGGER portion (t < reactive_trigger_t) is
    included -- post-trigger, physics is either faked (no_collision) or
    the action is a scripted (non-policy) retreat/lift, neither of which
    is a natural "policy proposes an action, does it lead to contact"
    sample.
  - Every other condition (baseline, low_mobility, oracle(16,18),
    composite_visual_only, etc.) has REAL, uninterrupted physics for its
    entire length -- included in full. Oracle's visual-completion
    correction changes what action gets proposed, not whether contact
    physics is real, so its samples are valid training data for this
    (different) target.

FEATURES, two variants (to separate what's learnable from proprioception
alone vs. what needs privileged occluder-position knowledge):
  - "realistic" (13-dim): eef_pos(3) + gripper_qpos(2) +
    eef_speed_since_last_replan(1, imputed 0 for the first replan step)
    + action_first(7) -- everything a real robot's own proprioception +
    proposed action already provides, no vision, no occluder position.
  - "oracle" (14-dim): realistic + eef_to_occluder_dist(1) -- privileged
    (uses true simulator occluder position), included ONLY as an upper-
    bound feasibility check, never as a claim about real-robot
    deployability.

LABEL: contact_within_k -- occluder_contact is True at the CURRENT
replan-step or any of the next K (default 2) replan-steps within the
SAME episode (real physics contact, from run_episode's own MuJoCo
contact-pair detection, not a distance threshold).

Held out by EPISODE (not by step) to avoid leaking adjacent steps of the
same trajectory across train/eval.
"""
import argparse
import glob
import json
import os

import numpy as np

EXCLUDE_CONDITIONS = {"no_collision", "oracle_no_collision"}
TRUNCATE_AT_TRIGGER_CONDITIONS = {"no_collision_after_contact", "scripted_recovery_after_contact"}


def load_occluder_positions(path):
    """occ_vla addition (2026-08-21), per user's directional-feature
    follow-up: occluder position is static per task, so it's cheap to
    look up once (via a fresh env query, see the companion snippet that
    generated `occluder_positions.json`) and reuse for every existing
    logged row -- no new rollouts needed, matching the user's own
    "既存データだけで作れる" framing for this whole line of work."""
    with open(path) as f:
        return {k: np.array(v, dtype=float) for k, v in json.load(f).items()}


def build_dataset(results_dir, task_json_names=("task1.json",), k_ahead=2, occluder_positions=None):
    rows = []  # each: (episode_uid, features_realistic, feature_oracle_extra, label, task_label, feat_directional)
    n_episodes_used = 0
    n_episodes_skipped = 0
    files = []
    for name in task_json_names:
        for f in glob.glob(os.path.join(results_dir, "*", name)):
            files.append((f, name))
    for f, task_json_name in sorted(files):
        dirpath = os.path.dirname(f)
        dirname = os.path.basename(dirpath)
        # BUG FOUND (2026-08-21, real contamination caught while extending
        # this script to task6/task8): globbing by result-FILENAME (e.g.
        # "task1.json") is not the same as globbing by SCENARIO -- a
        # `--use-stock-suite` run's stock task_id can coincidentally
        # produce a file named "task1.json" for a completely different,
        # non-occluded scene (e.g. `stock_libero10_baseline_task6equiv_n20/
        # task1.json` is task6's stock EQUIVALENT, stock task_id=1 --
        # nothing to do with occluded task1). Such runs have NO occluder
        # body at all, so occluder_contact is trivially always False and
        # eef_to_occluder_dist is always None -- same "contact never
        # happens because there is nothing to contact" contamination as
        # no_collision/oracle_no_collision, just from a different source.
        # Confirmed present in the ORIGINAL (pre-fix) task1-only run of
        # this script (`stock_libero10_baseline_task6equiv_n20` was
        # silently included, contaminating ~20 episodes' worth of
        # trivial-negative-label rows into the "realistic" variant, which
        # doesn't use eef_to_occluder_dist so wasn't caught by the earlier
        # None-value bug fix). Read run_config.json's own
        # `use_stock_suite` flag (the authoritative source, not dir-name
        # pattern-matching) and skip the whole dir if True.
        cfg_path = os.path.join(dirpath, "run_config.json")
        if os.path.exists(cfg_path):
            try:
                cfg = json.load(open(cfg_path))
                if cfg.get("use_stock_suite"):
                    continue
            except Exception:
                pass
        try:
            d = json.load(open(f))
        except Exception:
            continue
        results = d.get("results", {})
        if not isinstance(results, dict):
            continue
        for cond, res in results.items():
            if not isinstance(res, list):
                continue
            if cond in EXCLUDE_CONDITIONS:
                n_episodes_skipped += len(res)
                continue
            for r in res:
                plog = r.get("proprio_log")
                atrace = r.get("action_trace")
                if not plog or not atrace:
                    continue
                # align proprio_log[i] with action_trace[i] by matching t
                # (both are appended once per replan step, in order, in
                # run_episode -- confirmed same length in all inspected dirs)
                if len(plog) != len(atrace):
                    n_episodes_skipped += 1
                    continue
                cutoff = len(plog)
                if cond in TRUNCATE_AT_TRIGGER_CONDITIONS and r.get("reactive_triggered"):
                    trig_t = r.get("reactive_trigger_t")
                    if trig_t is not None:
                        cutoff = sum(1 for p in plog if p["t"] < trig_t)
                if cutoff < 2:
                    n_episodes_skipped += 1
                    continue
                task_label = task_json_name.replace(".json", "")
                episode_uid = f"{task_label}:{dirname}:{cond}:{r.get('episode')}"
                occ_contact = [bool(p.get("occluder_contact")) for p in plog[:cutoff]]
                prev_speed = None
                for i in range(cutoff):
                    p = plog[i]
                    a = atrace[i]
                    speed = p.get("eef_speed_since_last_replan")
                    speed = speed if speed is not None else 0.0
                    feat_realistic = list(p["eef_pos"]) + list(p["gripper_qpos"]) + [speed] + list(a["action_first"])
                    # BUG FOUND (2026-08-21, this script): dict.get(key, 0.0)
                    # only substitutes the default when the KEY is absent --
                    # several episodes have the key present with value None
                    # (occluder-distance computation edge cases), which
                    # silently became NaN once cast to float, corrupting
                    # standardization (mean/std -> NaN) and the "oracle"
                    # variant's entire training run. Imputing 0.0 would be
                    # actively misleading for a DISTANCE feature (0 reads as
                    # "touching") -- drop these samples instead (~2.4% of
                    # the full dataset), same count used for both variants
                    # so the realistic-vs-oracle comparison stays apples-
                    # to-apples on identical rows.
                    raw_extra = p.get("eef_to_occluder_dist")
                    if raw_extra is None:
                        continue
                    # occ_vla addition (2026-08-21), per user's own
                    # follow-up proposal: a scalar distance carries no
                    # DIRECTIONAL information (can't tell "approaching"
                    # from "retreating"). Directional feature = (unit
                    # vector from eef toward the occluder) . (unit vector
                    # of the proposed action's xyz component) -- positive
                    # = this action moves toward the obstacle, negative =
                    # away. Task-independent in principle (always
                    # relative to the current eef/occluder geometry, not
                    # an absolute coordinate) -- the actual point of this
                    # experiment is to test whether that's true in
                    # practice too.
                    feat_directional = None
                    if occluder_positions is not None and task_label in occluder_positions:
                        occ_pos = occluder_positions[task_label]
                        eef = np.array(p["eef_pos"], dtype=float)
                        to_occ = occ_pos - eef
                        to_occ_norm = np.linalg.norm(to_occ)
                        act_xyz = np.array(a["action_first"][:3], dtype=float)
                        act_norm = np.linalg.norm(act_xyz)
                        if to_occ_norm > 1e-6 and act_norm > 1e-6:
                            feat_directional = float(np.dot(to_occ / to_occ_norm, act_xyz / act_norm))
                    # Only require the directional feature (and thus drop
                    # the row if it can't be computed) when the caller
                    # actually asked for it -- keeps this function fully
                    # backward compatible with every earlier call site
                    # that doesn't pass occluder_positions at all.
                    if occluder_positions is not None and feat_directional is None:
                        continue
                    label = any(occ_contact[i : min(cutoff, i + 1 + k_ahead)])
                    rows.append((episode_uid, feat_realistic, raw_extra, label, task_label, feat_directional if feat_directional is not None else 0.0))
                n_episodes_used += 1
    return rows, n_episodes_used, n_episodes_skipped


def train_logreg(X, y, epochs=300, lr=0.1, l2=1e-3, seed=0):
    rng = np.random.RandomState(seed)
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(epochs):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        grad_w = X.T @ (p - y) / n + l2 * w
        grad_b = np.mean(p - y)
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def predict_proba(X, w, b):
    z = X @ w + b
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def auc_score(y_true, y_score):
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_ranks_pos = ranks[y_true == 1].sum()
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def evaluate(w, b, X, y, mean, std):
    Xn = (X - mean) / std
    p = predict_proba(Xn, w, b)
    pred = (p >= 0.5).astype(int)
    acc = (pred == y).mean()
    tp = ((pred == 1) & (y == 1)).sum()
    fp = ((pred == 1) & (y == 0)).sum()
    fn = ((pred == 0) & (y == 1)).sum()
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    auc = auc_score(y, p)
    return dict(n=len(y), pos_rate=float(y.mean()), acc=float(acc), precision=float(prec), recall=float(rec), auc=float(auc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=".")
    ap.add_argument("--task-json-names", nargs="+", default=["task1.json"])
    ap.add_argument("--k-ahead", type=int, default=2)
    ap.add_argument("--eval-frac", type=float, default=0.2)
    ap.add_argument(
        "--eval-mode", choices=["episode", "task"], default="episode",
        help="'episode': random held-out episodes within the same task pool (in-distribution). "
             "'task': leave-one-task-out -- eval on the task named by --eval-task, train on the rest "
             "(the real generalization test, per the user's own requirement #4).",
    )
    ap.add_argument("--eval-task", default=None, help="task_label (e.g. 'task6') to hold out entirely, for --eval-mode task")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-name", default="contact_risk_predictor_v0_results.json")
    ap.add_argument("--occluder-positions", default=None, help="path to occluder_positions.json, enables the directional feature variant")
    args = ap.parse_args()

    occluder_positions = load_occluder_positions(args.occluder_positions) if args.occluder_positions else None
    rows, n_used, n_skipped = build_dataset(
        args.results_dir, task_json_names=args.task_json_names, k_ahead=args.k_ahead,
        occluder_positions=occluder_positions,
    )
    print(f"episodes used: {n_used}, episodes skipped (excluded/misaligned): {n_skipped}")
    print(f"total (state,action)->contact samples: {len(rows)}")
    task_counts = {}
    for r in rows:
        task_counts[r[4]] = task_counts.get(r[4], 0) + 1
    print(f"per-task sample counts: {task_counts}")

    if args.eval_mode == "task":
        assert args.eval_task, "--eval-task required for --eval-mode task"
        episode_uids = sorted(set(r[0] for r in rows))
        eval_eps = {u for u in episode_uids if u.split(":")[0] == args.eval_task}
        train_eps = {u for u in episode_uids if u.split(":")[0] != args.eval_task}
        print(f"CROSS-TASK generalization: holding out ALL of {args.eval_task} "
              f"({len(eval_eps)} episodes), training on {sorted(set(u.split(':')[0] for u in train_eps))} "
              f"({len(train_eps)} episodes)")
    else:
        episode_uids = sorted(set(r[0] for r in rows))
        rng = np.random.RandomState(args.seed)
        rng.shuffle(episode_uids)
        n_eval_ep = max(1, int(len(episode_uids) * args.eval_frac))
        eval_eps = set(episode_uids[:n_eval_ep])
        train_eps = set(episode_uids[n_eval_ep:])
        print(f"episodes: {len(episode_uids)} total -> {len(train_eps)} train / {len(eval_eps)} eval (held out whole, in-distribution)")

    def split(rows, eps):
        X_r = np.array([r[1] for r in rows if r[0] in eps], dtype=float)
        X_extra = np.array([r[2] for r in rows if r[0] in eps], dtype=float).reshape(-1, 1)
        y = np.array([r[3] for r in rows if r[0] in eps], dtype=float)
        X_dir = np.array([r[5] for r in rows if r[0] in eps], dtype=float).reshape(-1, 1) if len(rows[0]) > 5 else None
        return X_r, X_extra, y, X_dir

    Xr_train, Xex_train, y_train, Xdir_train = split(rows, train_eps)
    Xr_eval, Xex_eval, y_eval, Xdir_eval = split(rows, eval_eps)

    print(f"\ntrain pos_rate={y_train.mean():.3f} n={len(y_train)}  eval pos_rate={y_eval.mean():.3f} n={len(y_eval)}")

    variants = [
        ("realistic_13dim", Xr_train, Xr_eval),
        ("oracle_scalar_dist_14dim", np.hstack([Xr_train, Xex_train]), np.hstack([Xr_eval, Xex_eval])),
    ]
    if occluder_positions is not None:
        variants.append((
            "directional_14dim",
            np.hstack([Xr_train, Xdir_train]), np.hstack([Xr_eval, Xdir_eval]),
        ))
    results = {}
    for variant, Xtr, Xev in variants:
        mean = Xtr.mean(axis=0)
        std = Xtr.std(axis=0)
        std[std < 1e-8] = 1.0
        Xtr_n = (Xtr - mean) / std
        w, b = train_logreg(Xtr_n, y_train, epochs=3000, lr=0.3, l2=1e-3, seed=args.seed)
        train_metrics = evaluate(w, b, Xtr, y_train, mean, std)
        eval_metrics = evaluate(w, b, Xev, y_eval, mean, std)
        # majority-class baseline for comparison
        majority_pred = 0 if y_train.mean() < 0.5 else 1
        majority_acc_eval = (y_eval == majority_pred).mean()
        print(f"\n=== {variant} ===")
        print(f"  train: {train_metrics}")
        print(f"  eval:  {eval_metrics}")
        print(f"  majority-class baseline eval acc: {majority_acc_eval:.3f}")
        results[variant] = dict(train=train_metrics, eval=eval_metrics, majority_acc_eval=float(majority_acc_eval))

    with open(args.out_name, "w") as f:
        json.dump(dict(
            eval_mode=args.eval_mode, eval_task=args.eval_task,
            task_json_names=args.task_json_names, task_counts=task_counts,
            n_episodes_used=n_used, n_episodes_skipped=n_skipped, n_samples=len(rows),
            n_train_episodes=len(train_eps), n_eval_episodes=len(eval_eps),
            k_ahead=args.k_ahead, results=results,
        ), f, indent=2)
    print(f"\nsaved {args.out_name}")


if __name__ == "__main__":
    main()
