#!/usr/bin/env python
"""Measure the three things that decide how much the headline score is worth.

A single number on a 72-sample test set invites more confidence than it can
carry. This script computes the context that belongs next to it, writes the
answers to ``artifacts/honesty.json``, and can re-check them:

1. **How wide is the error bar?** A bootstrap interval over the held-out
   biopsies. If competing models sit inside each other's intervals, the ranking
   between them is not a result.

2. **What does the patient-grouped split cost?** GSE14520 is paired, so a plain
   stratified split leaves most held-out biopsies with their own patient's
   other tissue in training. Running the identical pipeline both ways, over
   many seeds, prices the leak instead of arguing about it.

3. **How hard is the task to begin with?** The AUC of the single best probe,
   thresholded, with no model at all. If one number out of 22,277 nearly
   solves it, a tuned model scoring a little higher is engineering, not a
   discovery -- and saying so is the difference between a defensible portfolio
   piece and an overclaim.

Usage:
    python scripts/evaluate_honestly.py
    python scripts/evaluate_honestly.py --check     # fail if committed answers are stale
    python scripts/evaluate_honestly.py --seeds 20  # tighter split comparison
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from liver_hcc import config, data, features

OUT = REPO / "artifacts" / "honesty.json"
N_BOOTSTRAP = 10_000


def bootstrap_f1(y_true: np.ndarray, y_pred: np.ndarray, seed: int = 0) -> dict:
    """Percentile bootstrap interval for F1 on the held-out biopsies.

    Resamples the test set with replacement. Draws that end up single-class are
    dropped rather than scored: F1 is undefined there, and counting them as 0
    would widen the interval for a reason that has nothing to do with the model.
    """
    rng = np.random.default_rng(seed)
    scores = []
    n = len(y_true)
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(f1_score(y_true[idx], y_pred[idx]))
    scores = np.asarray(scores)
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return {
        "point": float(f1_score(y_true, y_pred)),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "ci_width": float(hi - lo),
        "n_test": int(n),
        "draws": len(scores),
    }


def _fit_once(X, yb, train_idx, test_idx, groups) -> tuple[float, float, float]:
    """One end-to-end fit: label-free reduction, then the leakage-free pipeline.

    Hyperparameters are held fixed rather than tuned, because the question here
    is what the *split* is worth. Re-tuning per split would let the search
    absorb some of the difference and blur exactly the effect being measured.
    """
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    X_train, X_test, _ = features.drop_zero_variance(X_train, X_test)
    X_train, X_test, _ = features.drop_high_null_and_impute(X_train, X_test)
    X_train, X_test, _ = features.variance_filter_raw(X_train, X_test)

    pipe = features.build_model_pipeline(
        LogisticRegression(solver="liblinear", max_iter=1000, C=1,
                           penalty="l1", random_state=config.RANDOM_SEED)
    )
    pipe.set_params(rfe__n_features_to_select=20)
    pipe.fit(X_train, yb[train_idx])
    prob = pipe.predict_proba(X_test)[:, 1]
    overlap = float(np.isin(groups[test_idx], groups[train_idx]).mean())
    return (
        float(f1_score(yb[test_idx], (prob >= 0.5).astype(int))),
        float(roc_auc_score(yb[test_idx], prob)),
        overlap,
    )


def price_the_leak(X, yb, groups, seeds: int) -> dict:
    """Same pipeline, same budget, two ways of splitting."""
    n_splits = round(1 / config.TEST_SIZE)
    rows = {"random": [], "grouped": []}
    for seed in range(seeds):
        plain = StratifiedShuffleSplit(1, test_size=config.TEST_SIZE, random_state=seed)
        train_idx, test_idx = next(plain.split(X, yb))
        rows["random"].append(_fit_once(X, yb, train_idx, test_idx, groups))

        grouped = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        train_idx, test_idx = next(iter(grouped.split(X, yb, groups)))
        rows["grouped"].append(_fit_once(X, yb, train_idx, test_idx, groups))
        print(f"  seed {seed}: random {rows['random'][-1][0]:.4f} "
              f"(patient overlap {rows['random'][-1][2]:.0%})  "
              f"grouped {rows['grouped'][-1][0]:.4f}", flush=True)

    summary = {}
    for name, values in rows.items():
        arr = np.asarray(values)
        summary[name] = {
            "f1_mean": float(arr[:, 0].mean()),
            "f1_std": float(arr[:, 0].std()),
            "auc_mean": float(arr[:, 1].mean()),
            "patient_overlap": float(arr[:, 2].mean()),
        }
    summary["cost_of_grouping_f1"] = round(
        summary["grouped"]["f1_mean"] - summary["random"]["f1_mean"], 6
    )
    summary["seeds"] = seeds
    return summary


def single_probe_ceiling(X: pd.DataFrame, yb: np.ndarray) -> dict:
    """How far one probe gets on its own, with no model and no training.

    Direction is not something a bare threshold knows, so a probe that separates
    the classes perfectly the other way round is just as informative; AUC is
    folded to ``max(auc, 1 - auc)`` accordingly.
    """
    values = X.to_numpy()
    aucs = np.array([roc_auc_score(yb, values[:, i]) for i in range(values.shape[1])])
    aucs = np.maximum(aucs, 1 - aucs)
    best = int(aucs.argmax())
    return {
        "best_probe": str(X.columns[best]),
        "best_auc": float(aucs[best]),
        "probes_over_95": int((aucs >= 0.95).sum()),
        "probes_over_90": int((aucs >= 0.90).sum()),
        "median_auc": float(np.median(aucs)),
        "n_probes": int(values.shape[1]),
    }


def stratified_split_leak(patients: pd.Series, y: pd.Series) -> dict:
    """What the split this project used to make would have leaked.

    Reproduced from the labels and the patient map alone -- the leak is a
    property of which rows land where, not of the expression values -- so this
    stays checkable long after the 128 MB dataset is not to hand.
    """
    from sklearn.model_selection import train_test_split

    index = pd.RangeIndex(len(y))
    train_idx, test_idx = train_test_split(
        index, test_size=config.TEST_SIZE, stratify=y,
        random_state=config.RANDOM_SEED,
    )
    train_patients = set(patients.iloc[train_idx])
    leaked = patients.iloc[test_idx].isin(train_patients)
    return {
        "leaked": int(leaked.sum()),
        "of": len(test_idx),
        "share": float(leaked.mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="fail if the committed answers are out of date")
    parser.add_argument("--seeds", type=int, default=10,
                        help="splits to average the leak comparison over")
    args = parser.parse_args()

    raw = data.load_raw()
    patients = data.load_patients(raw)
    X, y = data.split_features_target(data.basic_clean(raw))
    yb = data.y_to_binary(y).to_numpy()
    groups = patients.to_numpy()

    print("bootstrapping the held-out interval ...")
    bundle = joblib.load(config.MODEL_PATH)
    examples = json.loads(config.EXAMPLES_PATH.read_text(encoding="utf-8"))
    genes = bundle["genes"]
    X_test = pd.DataFrame(
        [[s["features"][g] for g in genes] for s in examples["samples"]], columns=genes
    )
    y_test = np.array([1 if s["label"] == config.CLASS_POS else 0
                       for s in examples["samples"]])
    prob = bundle["model"].predict_proba(X_test)[:, list(bundle["model"].classes_).index(1)]
    interval = bootstrap_f1(y_test, (prob >= 0.5).astype(int))
    print(f"  F1 {interval['point']:.4f}  95% CI "
          f"[{interval['ci_low']:.4f}, {interval['ci_high']:.4f}]")

    print(f"pricing the patient leak over {args.seeds} splits ...")
    leak = price_the_leak(X, yb, groups, args.seeds)

    print("scoring every probe on its own ...")
    ceiling = single_probe_ceiling(X, yb)
    print(f"  best single probe {ceiling['best_probe']} AUC {ceiling['best_auc']:.4f}")

    payload = {
        "stratified_split_leak": stratified_split_leak(patients, y),
        "held_out_interval": interval,
        "split_comparison": leak,
        "single_probe_ceiling": ceiling,
        "model_type": bundle["model_type"],
    }
    text = json.dumps(payload, indent=2) + "\n"

    if args.check:
        if not OUT.exists():
            print(f"FAIL  {OUT} does not exist")
            return 1
        if OUT.read_text(encoding="utf-8") != text:
            print(f"FAIL  {OUT.name} is stale; run: python scripts/evaluate_honestly.py")
            return 1
        print(f"{OUT.name} is up to date.")
        return 0

    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
