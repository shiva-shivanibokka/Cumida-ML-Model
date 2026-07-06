"""Model definitions and hyperparameter tuning.

Both models are trained through the leakage-free pipeline from
:mod:`liver_hcc.features`, so the reported cross-validation scores are honest:
feature scaling and selection are re-fit inside every CV fold, and the number of
selected genes is tuned alongside the model's own hyperparameters.

Labels are binarized to 1/0 (HCC/normal) up front so F1 has an unambiguous
positive class and both models are scored identically.
"""

from __future__ import annotations

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .features import build_model_pipeline

# Candidate feature counts for RFE, tuned as a pipeline hyperparameter.
RFE_CANDIDATES = [10, 20, 30, 50, 75, 100]


def _cv() -> StratifiedKFold:
    return StratifiedKFold(
        n_splits=5, shuffle=True, random_state=config.RANDOM_SEED
    )


def deployable_model_from_search(search, X_train, y_train_binary, selected_genes):
    """Fit the compact model that actually gets deployed.

    Tuning finds the best hyperparameters and the genes RFE selected; the model
    we ship is a plain ``StandardScaler -> classifier`` trained on the full
    training data using only those genes, so the serving API takes a handful of
    gene values. This is the standard "select via CV, then refit on all training
    data" pattern — the test set is still never touched.
    """
    params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()
              if k.startswith("clf__")}
    estimator = clone_best_classifier(search.best_estimator_.named_steps["clf"], params)
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", estimator)])
    pipe.fit(X_train[list(selected_genes)], y_train_binary)
    return pipe


def clone_best_classifier(template_estimator, params):
    """Return a fresh classifier of the same type as ``template_estimator`` with ``params``."""
    from sklearn.base import clone

    est = clone(template_estimator)
    # cast numpy scalars from the search to plain python for a clean estimator repr
    clean = {k: (v.item() if hasattr(v, "item") else v) for k, v in params.items()}
    est.set_params(**clean)
    return est


# --- Logistic Regression -----------------------------------------------------
def tune_logistic_regression(X_train, y_train_binary, verbose: int = 0) -> GridSearchCV:
    """Grid-search Logistic Regression (C, penalty) *and* the RFE feature count."""
    pipe = build_model_pipeline(
        LogisticRegression(
            solver="liblinear", max_iter=1000, random_state=config.RANDOM_SEED
        )
    )
    param_grid = {
        "rfe__n_features_to_select": RFE_CANDIDATES,
        "clf__C": [0.001, 0.01, 0.1, 1, 10, 100],
        "clf__penalty": ["l1", "l2"],
    }
    search = GridSearchCV(
        pipe, param_grid, cv=_cv(), scoring="f1", n_jobs=-1, verbose=verbose
    )
    search.fit(X_train, y_train_binary)
    return search


# --- Gradient Boosting -------------------------------------------------------
def tune_gradient_boosting(X_train, y_train_binary, n_iter: int = 20, verbose: int = 0):
    """Bayesian-search Gradient Boosting hyperparameters *and* the RFE feature count.

    Uses scikit-optimize's ``BayesSearchCV`` when available; falls back to
    ``RandomizedSearchCV`` (equivalent search budget) if scikit-optimize is not
    installed, so the pipeline never hard-fails on a missing optional dependency.
    """
    pipe = build_model_pipeline(
        GradientBoostingClassifier(random_state=config.RANDOM_SEED)
    )
    try:
        from skopt import BayesSearchCV
        from skopt.space import Categorical, Integer, Real

        search_space = {
            "rfe__n_features_to_select": Categorical(RFE_CANDIDATES),
            "clf__n_estimators": Integer(50, 300),
            "clf__learning_rate": Real(0.01, 0.3, prior="log-uniform"),
            "clf__max_depth": Integer(2, 6),
            "clf__subsample": Real(0.5, 1.0),
        }
        search = BayesSearchCV(
            pipe,
            search_space,
            n_iter=n_iter,
            cv=_cv(),
            scoring="f1",
            n_jobs=-1,
            random_state=config.RANDOM_SEED,
            verbose=verbose,
        )
    except ImportError:
        from scipy.stats import loguniform, randint, uniform
        from sklearn.model_selection import RandomizedSearchCV

        param_dist = {
            "rfe__n_features_to_select": RFE_CANDIDATES,
            "clf__n_estimators": randint(50, 300),
            "clf__learning_rate": loguniform(0.01, 0.3),
            "clf__max_depth": randint(2, 7),
            "clf__subsample": uniform(0.5, 0.5),
        }
        search = RandomizedSearchCV(
            pipe,
            param_dist,
            n_iter=n_iter,
            cv=_cv(),
            scoring="f1",
            n_jobs=-1,
            random_state=config.RANDOM_SEED,
            verbose=verbose,
        )

    search.fit(X_train, y_train_binary)
    return search
