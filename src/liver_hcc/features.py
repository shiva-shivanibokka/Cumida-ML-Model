"""Feature reduction.

Two kinds of reduction live here, and the distinction is the whole point:

1. **Label-free (unsupervised) cleaning** — zero-variance filter, high-null
   filter + median impute, and a raw-scale variance filter. None of these look
   at the target, so fitting them once on the training data does not leak test
   information *and* does not inflate cross-validation scores. Notebook 02 does
   this and saves a smaller matrix.

2. **Supervised selection (RFE / univariate F-test)** — these DO use the target.
   Running them once on the full training set and then cross-validating the model
   on the result is a classic *selection leakage* trap: every CV fold's model was
   built on features chosen with that fold's own labels, so CV scores come out
   optimistically high. The fix is to put supervised selection *inside* a
   ``Pipeline`` so it is re-fit within each CV fold. That is what
   :func:`build_model_pipeline` does, and it is why the model notebooks tune the
   number of selected features as a hyperparameter instead of hard-coding it.

Important subtlety about the raw variance filter: ``VarianceThreshold`` must run
on the *raw* expression values, NOT after ``StandardScaler``. Standardizing
forces every column to variance 1.0, which makes any variance threshold below 1
a silent no-op. The original pipeline scaled first and then thresholded, so the
step removed nothing — for the wrong reason. Here we threshold on raw data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, VarianceThreshold, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config

# Number of genes the fast univariate F-test keeps *inside* each CV fold before
# the slower RFE ranks them. Bounds RFE's cost without touching the test set.
PREFILTER_K: int = 300


# --- 1. Label-free cleaning (fit on train, apply to test) --------------------
def drop_zero_variance(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Drop probes that are constant across all training samples."""
    var = X_train.var()
    zero_cols = var[var == 0].index.tolist()
    return X_train.drop(columns=zero_cols), X_test.drop(columns=zero_cols), zero_cols


def drop_high_null_and_impute(
    X_train: pd.DataFrame, X_test: pd.DataFrame, null_thresh: float = 0.90
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Drop probes missing in >``null_thresh`` of training samples; median-impute the rest."""
    null_rate = X_train.isnull().mean()
    high_null = null_rate[null_rate > null_thresh].index.tolist()
    X_train = X_train.drop(columns=high_null)
    X_test = X_test.drop(columns=high_null)

    if X_train.isnull().values.any():
        medians = X_train.median()
        X_train = X_train.fillna(medians)
        X_test = X_test.fillna(medians)
    return X_train, X_test, high_null


def variance_filter_raw(
    X_train: pd.DataFrame, X_test: pd.DataFrame, threshold: float = 0.05
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Remove near-constant probes using variance on the RAW (unscaled) data.

    Fitted on training data only. Returns the reduced frames and the kept-column
    names. See the module docstring for why this must not run after scaling.
    """
    vt = VarianceThreshold(threshold=threshold).fit(X_train)
    keep = X_train.columns[vt.get_support()]
    return X_train[keep].copy(), X_test[keep].copy(), np.asarray(keep)


# --- 2. Supervised selection lives INSIDE the model pipeline -----------------
def build_selector_steps(
    prefilter_k: int = PREFILTER_K,
) -> list[tuple[str, object]]:
    """The leakage-safe front half of every model pipeline.

    scaler -> univariate prefilter (fast, F-test) -> RFE ranking (Random Forest).
    All three are fit inside each CV fold when wrapped by GridSearchCV/BayesSearchCV.
    """
    return [
        ("scaler", StandardScaler()),
        ("prefilter", SelectKBest(f_classif, k=prefilter_k)),
        (
            "rfe",
            RFE(
                estimator=RandomForestClassifier(
                    n_estimators=100,
                    random_state=config.RANDOM_SEED,
                    # single-threaded on purpose: the outer GridSearch/BayesSearch
                    # parallelizes across folds, so a parallel RF here would
                    # oversubscribe every core and run *slower*.
                    n_jobs=1,
                ),
                n_features_to_select=30,  # tuned by the search; this is just a default
                step=0.5,
            ),
        ),
    ]


def build_model_pipeline(estimator, prefilter_k: int = PREFILTER_K) -> Pipeline:
    """Full leakage-free pipeline: scale -> prefilter -> RFE -> ``estimator``."""
    steps = build_selector_steps(prefilter_k)
    steps.append(("clf", estimator))
    return Pipeline(steps)


def selected_gene_names(fitted_pipeline: Pipeline, input_columns) -> np.ndarray:
    """Recover the gene-probe names the fitted pipeline actually kept.

    Composes the boolean masks of the prefilter and the RFE step so the
    coefficient / importance charts can be labeled with real probe IDs.
    """
    cols = np.asarray(input_columns)
    cols = cols[fitted_pipeline.named_steps["prefilter"].get_support()]
    cols = cols[fitted_pipeline.named_steps["rfe"].get_support()]
    return cols
