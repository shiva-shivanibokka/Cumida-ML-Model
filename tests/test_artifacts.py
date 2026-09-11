"""Does the model that ships still produce the numbers that ship with it?

Every other test here builds a synthetic model so the suite stays fast and
hermetic. That is the right trade for testing the API contract, and exactly the
wrong one for testing the claim the README makes -- a synthetic model cannot
tell you that ``model.joblib`` and ``metrics.json`` still agree.

So this file re-runs the committed model over the committed held-out biopsies
and checks it lands on the committed scores. It needs no dataset and no
retraining: the 72 biopsies travel with the demo data.
"""

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from liver_hcc import config

BUNDLE = joblib.load(config.MODEL_PATH)
METRICS = json.loads(config.METRICS_PATH.read_text(encoding="utf-8"))
EXAMPLES = json.loads(config.EXAMPLES_PATH.read_text(encoding="utf-8"))
SPLIT = json.loads(config.SPLIT_PATH.read_text(encoding="utf-8"))

KEY = {"Logistic Regression": "logistic_regression", "Gradient Boosting": "gradient_boosting"}


def _held_out(genes):
    X = pd.DataFrame(
        [[row["features"][g] for g in genes] for row in EXAMPLES["samples"]], columns=genes
    )
    y = np.array([1 if row["label"] == config.CLASS_POS else 0 for row in EXAMPLES["samples"]])
    return X, y


def test_the_shipped_demo_samples_are_the_held_out_set():
    assert len(EXAMPLES["samples"]) == METRICS["n_test"] == len(SPLIT["test"])
    assert EXAMPLES["genes"] == BUNDLE["genes"]


@pytest.mark.parametrize("name", ["Logistic Regression", "Gradient Boosting"])
def test_committed_model_reproduces_committed_metrics(name):
    """The check that a synthetic-model test suite structurally cannot make."""
    spec = BUNDLE["all_models"][name]
    genes = spec["genes"]
    X, y = _held_out(genes)

    model = spec["model"]
    prob = model.predict_proba(X)[:, list(model.classes_).index(1)]
    pred = (prob >= 0.5).astype(int)

    want = METRICS[KEY[name]]
    got = {
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, prob),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
    }
    for metric, value in got.items():
        assert value == pytest.approx(want[metric], abs=1e-9), (
            f"{name} {metric}: model gives {value:.6f}, metrics.json says {want[metric]:.6f}"
        )

    tp, fn, fp, tn = (
        int(((y == 1) & (pred == 1)).sum()), int(((y == 1) & (pred == 0)).sum()),
        int(((y == 0) & (pred == 1)).sum()), int(((y == 0) & (pred == 0)).sum()),
    )
    assert {"tp": tp, "fn": fn, "fp": fp, "tn": tn} == want["confusion"]
    assert len(genes) == want["n_genes"]


def test_the_winner_is_the_model_that_is_served():
    assert BUNDLE["model_type"] == METRICS["winner"]
    served = BUNDLE["model"].named_steps["clf"]
    named = BUNDLE["all_models"][BUNDLE["model_type"]]["model"].named_steps["clf"]
    assert type(served) is type(named)
    assert BUNDLE["genes"] == BUNDLE["all_models"][BUNDLE["model_type"]]["genes"]


def test_identical_scores_are_not_identical_predictions():
    """The distinction the site is built around, checked rather than asserted.

    An earlier version of this test asserted the two models return the same 72
    predictions, on the strength of their confusion matrices being byte-
    identical. It failed, which is how the page's headline came to be wrong:
    the same counts can be reached by getting *different* biopsies wrong. They
    are each wrong three times and share two of those mistakes.
    """
    verdicts = {}
    for name, spec in BUNDLE["all_models"].items():
        X, truth = _held_out(spec["genes"])
        model = spec["model"]
        prob = model.predict_proba(X)[:, list(model.classes_).index(1)]
        verdicts[name] = (prob >= 0.5).astype(int)

    a, b = verdicts.values()
    assert (METRICS["logistic_regression"]["confusion"]
            == METRICS["gradient_boosting"]["confusion"]), "the scores tie"
    assert not np.array_equal(a, b), (
        "the two models now return identical predictions; the page's point is "
        "that identical scores hide different mistakes, so it needs rewriting"
    )

    _, truth = _held_out(BUNDLE["genes"])
    wrong = [set(np.where(p != truth)[0]) for p in (a, b)]
    assert len(wrong[0]) == len(wrong[1]), "equal error counts are why the scores tie"
    assert wrong[0] != wrong[1], "different mistakes are why the models are not the same"
