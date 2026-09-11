"""FastAPI serving layer for the trained liver HCC classifier.

Loads the model saved by ``train.py`` and exposes:

    GET  /         -> a short index of the endpoints
    GET  /health   -> liveness/readiness probe (used by Docker / orchestrators)
    GET  /model    -> metadata: model type, the genes it expects, class labels
    POST /predict  -> {gene_probe: value, ...} -> predicted class + P(HCC)

Every prediction is logged as a structured (JSON) line so the service is
observable in production log aggregators without extra tooling.

Run locally:
    uvicorn liver_hcc.serve:app --reload
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from . import config

# --- Structured logging ------------------------------------------------------
logger = logging.getLogger("liver_hcc.serve")
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


def _log_event(event: str, **fields: Any) -> None:
    logger.info(json.dumps({"event": event, **fields}))


# --- Model loading (lazy, cached) --------------------------------------------
_BUNDLE: dict | None = None


def load_bundle() -> dict:
    """Load and cache the model bundle saved by train.py."""
    global _BUNDLE
    if _BUNDLE is None:
        if not config.MODEL_PATH.exists():
            raise FileNotFoundError(
                f"No model at {config.MODEL_PATH}. Run `python train.py` first."
            )
        _BUNDLE = joblib.load(config.MODEL_PATH)
        _log_event(
            "model_loaded",
            model_type=_BUNDLE["model_type"],
            n_genes=len(_BUNDLE["genes"]),
        )
    return _BUNDLE


# --- Request / response schemas ----------------------------------------------
class PredictRequest(BaseModel):
    features: dict[str, float] = Field(
        ...,
        description="Mapping of gene-probe id -> expression value. "
        "Must include every gene listed at GET /model.",
        examples=[{"209365_s_at": 11.2, "216661_x_at": 4.1}],
    )


class PredictResponse(BaseModel):
    # model_type would collide with pydantic's protected "model_" namespace.
    model_config = {"protected_namespaces": ()}

    prediction: str
    probability_hcc: float
    model_type: str


app = FastAPI(
    title="Liver HCC Classifier",
    description="Classifies liver tissue as HCC or normal from microarray gene expression.",
    version="1.0.0",
)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """A short index of the endpoints. Renders even with no model present."""
    try:
        model_type = load_bundle()["model_type"]
    except Exception:
        model_type = "not loaded — run `python train.py`"
    return INDEX.replace("__MODEL__", model_type)


@app.get("/health")
def health() -> dict:
    """Liveness probe. Reports whether a trained model is available."""
    return {"status": "ok", "model_available": config.MODEL_PATH.exists()}


@app.get("/model")
def model_info() -> dict:
    """Metadata about the loaded model, including the exact genes it expects."""
    bundle = load_bundle()
    return {
        "model_type": bundle["model_type"],
        "classes": {"positive": bundle["class_pos"], "negative": bundle["class_neg"]},
        "n_genes": len(bundle["genes"]),
        "genes": bundle["genes"],
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """Predict HCC vs normal for one sample's gene-expression values."""
    bundle = load_bundle()
    genes = bundle["genes"]

    missing = [g for g in genes if g not in req.features]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing {len(missing)} required gene(s), e.g. {missing[:5]}",
        )

    # Order features exactly as the model expects. Build a named 1-row frame:
    # the deployed pipeline was fitted on a DataFrame, so passing column names
    # keeps predictions warning-free (a bare list triggers sklearn's
    # "X does not have valid feature names" UserWarning on every request,
    # polluting the structured JSON logs).
    row = pd.DataFrame([[float(req.features[g]) for g in genes]], columns=genes)

    t0 = time.time()
    model = bundle["model"]
    prob_hcc = float(model.predict_proba(row)[0][list(model.classes_).index(1)])
    label = bundle["class_pos"] if prob_hcc >= 0.5 else bundle["class_neg"]

    _log_event(
        "prediction",
        prediction=label,
        probability_hcc=round(prob_hcc, 4),
        latency_ms=round((time.time() - t0) * 1000, 2),
    )
    return PredictResponse(
        prediction=label, probability_hcc=prob_hcc, model_type=bundle["model_type"]
    )


# --- Index -------------------------------------------------------------------
# This service is the API. The interactive demo is a static page (web/) that
# runs the same models in the browser, so it has no backend to be hosted with
# and cannot go dark. An earlier version embedded a ~300-line copy of that page
# here and called /predict from it, which meant two demos drifting apart; this
# is the one that has to agree with the model, so it is the one that stayed.
INDEX = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Liver HCC Classifier API</title>
<style>
 body{font:15px/1.6 ui-sans-serif,system-ui,sans-serif;margin:0;padding:3rem 1.25rem;
      background:#0e1310;color:#e8efe9}
 main{max-width:44rem;margin:0 auto}
 h1{font-size:1.4rem;margin:0 0 .4rem}
 p{color:#93a098;max-width:60ch}
 code,a{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
 a{color:#58c39f}
 table{border-collapse:collapse;margin-top:1.4rem;width:100%}
 td{padding:.35rem .6rem .35rem 0;border-bottom:1px solid #263029;vertical-align:top}
 td:first-child{white-space:nowrap;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
</style></head><body><main>
<h1>Liver HCC Classifier &mdash; API</h1>
<p>Classifies a liver biopsy as hepatocellular carcinoma or normal tissue from
microarray gene expression. Model: __MODEL__.</p>
<p>The interactive demo lives at
<a href="https://github.com/shiva-shivanibokka/Cumida-ML-Model">the project page</a>
and runs the same models client-side.</p>
<table>
<tr><td>GET /health</td><td>liveness probe; reports whether a model is loaded</td></tr>
<tr><td>GET /model</td><td>model type, class labels, and the exact genes it expects</td></tr>
<tr><td>POST /predict</td><td>{"features": {gene: value, ...}} &rarr; class + P(HCC)</td></tr>
<tr><td>GET /docs</td><td><a href="/docs">interactive OpenAPI documentation</a></td></tr>
</table>
<p style="margin-top:2rem">Built by Shivani Bokka.</p>
</main></body></html>"""
