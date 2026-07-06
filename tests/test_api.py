"""API contract tests. A tiny synthetic model bundle is written to a temp path
so these run in milliseconds without needing the real dataset or a trained model.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from liver_hcc import config, serve

GENES = ["gene_a", "gene_b", "gene_c"]


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # Train a trivial but real sklearn pipeline on separable synthetic data.
    rng = np.random.default_rng(0)
    n = 60
    X = np.vstack([rng.normal(0, 1, (n, 3)), rng.normal(4, 1, (n, 3))])
    y = np.array([0] * n + [1] * n)
    model = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
    model.fit(X, y)

    bundle_path = tmp_path / "model.joblib"
    import joblib

    joblib.dump(
        {
            "model": model,
            "genes": GENES,
            "model_type": "Logistic Regression",
            "class_pos": config.CLASS_POS,
            "class_neg": config.CLASS_NEG,
        },
        bundle_path,
    )

    # Point the service at the temp model and clear any cached bundle.
    monkeypatch.setattr(config, "MODEL_PATH", bundle_path)
    monkeypatch.setattr(serve, "_BUNDLE", None)
    return TestClient(serve.app)


def test_health_reports_model_available(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["model_available"] is True


def test_model_info_lists_expected_genes(client):
    r = client.get("/model")
    assert r.status_code == 200
    assert r.json()["genes"] == GENES


def test_predict_returns_label_and_probability(client):
    r = client.post("/predict", json={"features": {g: 5.0 for g in GENES}})
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in {config.CLASS_POS, config.CLASS_NEG}
    assert 0.0 <= body["probability_hcc"] <= 1.0


def test_predict_rejects_missing_genes(client):
    r = client.post("/predict", json={"features": {"gene_a": 1.0}})
    assert r.status_code == 422
    assert "Missing" in r.json()["detail"]
