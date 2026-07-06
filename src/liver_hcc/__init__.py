"""Liver HCC classification from GSE14520 microarray gene expression.

A small, reproducible ML package: data loading, leakage-free feature selection,
two tuned classifiers (Logistic Regression, Gradient Boosting), evaluation, and
a FastAPI serving layer. The four teaching notebooks import from these modules.
"""

from __future__ import annotations

__version__ = "1.0.0"

from . import config, data, evaluate, features, models  # noqa: F401
