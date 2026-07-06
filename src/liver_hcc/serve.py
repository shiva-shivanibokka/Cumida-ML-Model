"""FastAPI serving layer for the trained liver HCC classifier.

Loads the model saved by ``train.py`` and exposes:

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
    """Human-friendly landing page so the root URL isn't a bare 404."""
    try:
        bundle = load_bundle()
        model_line = (
            f"{bundle['model_type']} on {len(bundle['genes'])} selected gene probes"
        )
    except Exception:
        model_line = "model not loaded — run `python train.py`"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Liver HCC Classifier API</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
    max-width: 640px; margin: 6vh auto; padding: 0 1.2rem; line-height: 1.6; }}
  h1 {{ margin-bottom: .2rem; }}
  .sub {{ opacity: .75; margin-top: 0; }}
  code {{ background: rgba(128,128,128,.18); padding: .1rem .35rem; border-radius: 4px; }}
  a {{ color: #2f81f7; }}
  .card {{ border: 1px solid rgba(128,128,128,.3); border-radius: 10px;
    padding: 1rem 1.2rem; margin: 1rem 0; }}
  ul {{ padding-left: 1.1rem; }}
</style></head><body>
  <h1>🧬 Liver HCC Classifier</h1>
  <p class="sub">Classifies liver tissue as <b>HCC</b> (hepatocellular carcinoma)
     or <b>normal</b> from microarray gene expression.</p>
  <div class="card">
    <b>Model:</b> {model_line}<br>
    <b>Status:</b> healthy
  </div>
  <h3>Endpoints</h3>
  <ul>
    <li><a href="/docs">/docs</a> — interactive API documentation (try it here)</li>
    <li><a href="/health">/health</a> — liveness probe</li>
    <li><a href="/model">/model</a> — the exact gene probes the model expects</li>
    <li><code>POST /predict</code> — send gene values, get a prediction</li>
  </ul>
  <p class="sub">Source &amp; write-up:
    <a href="https://github.com/shiva-shivanibokka/Cumida-ML-Model">github.com/shiva-shivanibokka/Cumida-ML-Model</a></p>
</body></html>"""


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

    # Order features exactly as the model expects.
    row = [[float(req.features[g]) for g in genes]]

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
