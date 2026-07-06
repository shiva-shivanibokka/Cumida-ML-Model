#!/usr/bin/env python
"""End-to-end training CLI.

Runs the whole pipeline the notebooks describe, but headless and reproducible:

    raw CSV -> clean -> stratified split -> label-free reduction
            -> leakage-free tuning of Logistic Regression and Gradient Boosting
            -> evaluate both on the held-out test set
            -> save the deployable model + selected genes + metrics.json

The deployed model is a compact ``StandardScaler -> classifier`` trained on only
the genes RFE selected during tuning, so the serving API takes a handful of gene
values rather than all 22,277 probes.

Usage:
    python train.py                 # full run, saves artifacts/
    python train.py --gb-iters 10   # faster Bayesian search for a quick smoke test
"""

from __future__ import annotations

import argparse
import json
import time

import warnings

import joblib

from liver_hcc import config, data, evaluate, features, models

# sklearn 1.8 deprecated LogisticRegression's `penalty=` in favour of `l1_ratio`;
# we keep `penalty` for compatibility with the older sklearn on Colab. Silence
# the (harmless) deprecation noise so the training log stays readable.
warnings.filterwarnings("ignore")


def _log(msg: str) -> None:
    print(f"[train] {msg}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the liver HCC classifier.")
    parser.add_argument("--gb-iters", type=int, default=20,
                        help="Bayesian search iterations for Gradient Boosting.")
    parser.add_argument("--variance-threshold", type=float, default=0.05,
                        help="Raw-scale variance filter threshold (label-free).")
    args = parser.parse_args()

    t0 = time.time()
    print(config.describe())
    config.ensure_artifacts_dir()

    # --- Load, clean, split --------------------------------------------------
    _log("loading raw dataset ...")
    df = data.basic_clean(data.load_raw())
    X, y = data.split_features_target(df)
    X_train, X_test, y_train, y_test = data.make_split(X, y)
    _log(f"split: {X_train.shape[0]} train / {X_test.shape[0]} test samples")

    # --- Label-free reduction (no target used; safe to fit once on train) ----
    X_train, X_test, zero_cols = features.drop_zero_variance(X_train, X_test)
    X_train, X_test, null_cols = features.drop_high_null_and_impute(X_train, X_test)
    X_train, X_test, kept = features.variance_filter_raw(
        X_train, X_test, threshold=args.variance_threshold
    )
    _log(
        f"label-free reduction: 22,277 -> {X_train.shape[1]} probes "
        f"(zero-var -{len(zero_cols)}, high-null -{len(null_cols)}, "
        f"var<{args.variance_threshold} -{22277 - len(zero_cols) - len(null_cols) - X_train.shape[1]})"
    )

    # Persist the reduced split so the notebooks can reload it identically.
    y_train_bin = data.y_to_binary(y_train)
    y_test_bin = data.y_to_binary(y_test)
    X_train.to_csv(config.X_TRAIN_CSV, index=False)
    X_test.to_csv(config.X_TEST_CSV, index=False)
    y_train.reset_index(drop=True).to_csv(config.Y_TRAIN_CSV, index=False)
    y_test.reset_index(drop=True).to_csv(config.Y_TEST_CSV, index=False)

    # --- Tune both models (leakage-free: selection re-fit inside each fold) ---
    _log("tuning Logistic Regression (GridSearchCV) ...")
    lr_search = models.tune_logistic_regression(X_train, y_train_bin)
    _log(f"  best CV F1={lr_search.best_score_:.4f}  params={lr_search.best_params_}")

    _log(f"tuning Gradient Boosting (BayesSearchCV, {args.gb_iters} iters) ...")
    gb_search = models.tune_gradient_boosting(X_train, y_train_bin, n_iter=args.gb_iters)
    _log(f"  best CV F1={gb_search.best_score_:.4f}  params={dict(gb_search.best_params_)}")

    # --- Recover selected genes and fit the deployable models ----------------
    lr_genes = features.selected_gene_names(lr_search.best_estimator_, X_train.columns)
    gb_genes = features.selected_gene_names(gb_search.best_estimator_, X_train.columns)

    lr_model = models.deployable_model_from_search(
        lr_search, X_train, y_train_bin, lr_genes
    )
    gb_model = models.deployable_model_from_search(
        gb_search, X_train, y_train_bin, gb_genes
    )

    # --- Evaluate on the held-out test set -----------------------------------
    lr_pred, lr_prob = evaluate.predict_binary_and_proba(lr_model, X_test[lr_genes])
    gb_pred, gb_prob = evaluate.predict_binary_and_proba(gb_model, X_test[gb_genes])
    lr_metrics = evaluate.evaluate_binary(y_test_bin, lr_pred, lr_prob)
    gb_metrics = evaluate.evaluate_binary(y_test_bin, gb_pred, gb_prob)

    winner = evaluate.winner_by_f1(lr_metrics, gb_metrics)
    best_model = gb_model if winner == "Gradient Boosting" else lr_model
    best_genes = gb_genes if winner == "Gradient Boosting" else lr_genes

    # --- Demo data for the UI (baked into the serving image) -----------------
    best_metrics = gb_metrics if winner == "Gradient Boosting" else lr_metrics
    cm = best_metrics["confusion"]
    # Top genes the model weights. LR gives signed coefficients (HCC=1, so a
    # positive weight pushes toward cancer); GB gives unsigned importances.
    clf = best_model.named_steps["clf"]
    if hasattr(clf, "coef_"):
        weights = [(g, round(float(w), 3)) for g, w in zip(best_genes, clf.coef_[0])]
    else:
        weights = [(g, round(float(w), 3)) for g, w in zip(best_genes, clf.feature_importances_)]
    top_genes = sorted(weights, key=lambda t: abs(t[1]), reverse=True)[:8]

    demo = {
        "genes": list(best_genes),
        "model_type": winner,
        "weighted": hasattr(clf, "coef_"),  # True = signed coefficients
        "top_genes": [{"gene": g, "weight": w} for g, w in top_genes],
        "meta": {
            "roc_auc": round(best_metrics["roc_auc"], 4),
            "f1": round(best_metrics["f1"], 4),
            "precision": round(best_metrics["precision"], 4),
            "recall": round(best_metrics["recall"], 4),
            "accuracy": round((cm["tp"] + cm["tn"]) / (cm["tp"] + cm["tn"] + cm["fp"] + cm["fn"]), 4),
            "confusion": cm,
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
            "n_probes": 22277,
        },
        # per-gene mean/std from TRAINING data, so the UI can z-score each cell
        # and colour it on the expression heatmap scale.
        "stats": {
            g: {"mean": round(float(X_train[g].mean()), 4),
                "std": round(float(X_train[g].std()) or 1.0, 4)}
            for g in best_genes
        },
        # every held-out test sample, so the UI can pick a random real biopsy.
        "samples": [
            {"label": lab, "features": {g: round(float(X_test.iloc[i][g]), 4) for g in best_genes}}
            for i, lab in enumerate(y_test.reset_index(drop=True))
        ],
    }
    config.EXAMPLES_PATH.write_text(json.dumps(demo))

    # --- Persist artifacts ---------------------------------------------------
    joblib.dump(
        {
            "model": best_model,
            "genes": list(best_genes),
            "model_type": winner,
            "class_pos": config.CLASS_POS,
            "class_neg": config.CLASS_NEG,
        },
        config.MODEL_PATH,
    )

    metrics = {
        "logistic_regression": {
            **lr_metrics,
            "cv_f1": float(lr_search.best_score_),
            "best_params": {k: _jsonable(v) for k, v in lr_search.best_params_.items()},
            "n_genes": len(lr_genes),
        },
        "gradient_boosting": {
            **gb_metrics,
            "cv_f1": float(gb_search.best_score_),
            "best_params": {k: _jsonable(v) for k, v in dict(gb_search.best_params_).items()},
            "n_genes": len(gb_genes),
        },
        "winner": winner,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_probes_after_label_free_reduction": int(X_train.shape[1]),
    }
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    _log(f"saved model -> {config.MODEL_PATH} ({winner}, {len(best_genes)} genes)")
    _log(f"saved metrics -> {config.METRICS_PATH}")
    _print_summary(lr_metrics, gb_metrics, winner)
    _log(f"done in {time.time() - t0:.0f}s")


def _jsonable(v):
    try:
        return v.item()  # numpy scalar -> python scalar
    except AttributeError:
        return v


def _print_summary(lr_metrics, gb_metrics, winner) -> None:
    print("\n" + "=" * 52)
    print(f"{'Metric':<12}{'LogReg':>12}{'GradBoost':>14}")
    print("-" * 52)
    for key, label in [("f1", "F1"), ("roc_auc", "ROC-AUC"),
                       ("precision", "Precision"), ("recall", "Recall")]:
        print(f"{label:<12}{lr_metrics[key]:>12.4f}{gb_metrics[key]:>14.4f}")
    print("-" * 52)
    print(f"Winner by F1: {winner}")
    print("=" * 52)


if __name__ == "__main__":
    main()
