"""Tests for feature reduction, including a regression test for the original
VarianceThreshold-after-scaling bug and a guard for leakage-free selection.
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from liver_hcc import features


def _toy_frame(n=40, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "informative": rng.normal(0, 5, n),      # high variance
            "tiny_var": rng.normal(0, 0.001, n),     # near-constant
            "constant": np.ones(n),                  # zero variance
        }
    )


def test_variance_filter_runs_on_raw_and_removes_low_variance():
    X = _toy_frame()
    kept, _, cols = features.variance_filter_raw(X, X.copy(), threshold=0.05)
    assert "informative" in kept.columns
    assert "tiny_var" not in kept.columns      # removed: variance well below 0.05
    assert "constant" not in kept.columns       # removed: zero variance
    assert set(cols) == set(kept.columns)


def test_variance_threshold_after_scaling_is_a_noop():
    """Regression test: this is the original bug the refactor fixed.

    After StandardScaler every column has variance ~1.0, so VarianceThreshold
    with any threshold < 1 removes nothing. The pipeline must therefore filter
    on RAW data (as variance_filter_raw does), not on scaled data.
    """
    X = _toy_frame().drop(columns=["constant"])  # drop true zero-var first
    X_scaled = StandardScaler().fit_transform(X)
    vt = VarianceThreshold(threshold=0.01).fit(X_scaled)
    assert vt.get_support().sum() == X.shape[1]  # nothing removed -> the no-op


def test_model_pipeline_selects_before_classifying():
    """Leakage guard: scaling + selection must precede the classifier so they
    are re-fit inside each CV fold when the pipeline is wrapped by a search."""
    pipe = features.build_model_pipeline(LogisticRegression())
    names = [name for name, _ in pipe.steps]
    assert names == ["scaler", "prefilter", "rfe", "clf"]
    assert names.index("clf") == len(names) - 1  # classifier is last


def test_variance_filter_fit_on_train_applies_to_test():
    X_train = _toy_frame(seed=1)
    X_test = _toy_frame(seed=2)
    kept_train, kept_test, _ = features.variance_filter_raw(
        X_train, X_test, threshold=0.05
    )
    # Test set is filtered to the SAME columns chosen from training data.
    assert list(kept_train.columns) == list(kept_test.columns)
