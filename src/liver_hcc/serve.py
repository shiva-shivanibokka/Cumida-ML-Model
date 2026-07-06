"""FastAPI serving layer for the trained liver HCC classifier.

Loads the model saved by ``train.py`` and exposes:

    GET  /         -> interactive demo landing page (gene-expression readout)
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


def load_examples() -> dict:
    """Real held-out test samples + per-gene stats baked into the image."""
    if config.EXAMPLES_PATH.exists():
        return json.loads(config.EXAMPLES_PATH.read_text())
    return {"samples": [], "genes": [], "stats": {}, "meta": {}, "model_type": ""}


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
    """Interactive demo landing page — a live gene-expression readout."""
    ex = load_examples()
    if not ex.get("model_type"):
        try:
            ex["model_type"] = load_bundle()["model_type"]
        except Exception:
            ex["model_type"] = ""
    return _LANDING_PAGE.replace("__DATA__", json.dumps(ex))


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


# --- Landing page ------------------------------------------------------------
# A single self-contained page. `__DATA__` is replaced at request time with the
# demo JSON (genes, per-gene stats, real samples, headline metrics).
_LANDING_PAGE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Liver HCC Classifier - gene-expression readout</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='88'>&#129516;</text></svg>">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root{
    --bg:#0A0E14; --panel:#111823; --panel2:#0D141D; --line:#1E2A38;
    --ink:#E7EEF6; --muted:#7C8CA0; --faint:#4C5B6E;
    --hcc:#FF5D73; --normal:#38C6E8;
    --sans:'IBM Plex Sans',system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
    --mono:'IBM Plex Mono',ui-monospace,SFMono-Regular,Menlo,monospace;
  }
  *{box-sizing:border-box}
  body{margin:0;background:
      radial-gradient(1200px 520px at 50% -220px,#13223300,#0A0E14 72%),var(--bg);
    color:var(--ink);font-family:var(--sans);line-height:1.55;
    -webkit-font-smoothing:antialiased;}
  .wrap{max-width:800px;margin:0 auto;padding:clamp(2rem,6vw,4.5rem) 1.25rem 4rem;}
  .eyebrow{font-family:var(--mono);font-size:.72rem;letter-spacing:.28em;
    text-transform:uppercase;color:var(--normal);margin:0 0 1rem;}
  h1{font-size:clamp(2rem,5.4vw,3.15rem);font-weight:600;line-height:1.04;
    letter-spacing:-.025em;margin:0 0 .9rem;}
  h1 .em{color:var(--hcc);}
  .lede{color:var(--muted);font-size:1.06rem;max-width:56ch;margin:0 0 1.9rem;}
  .stats{display:flex;flex-wrap:wrap;border:1px solid var(--line);
    border-radius:12px;overflow:hidden;margin-bottom:2.4rem;}
  .stat{flex:1 1 33%;min-width:120px;padding:.85rem 1.1rem;
    border-right:1px solid var(--line);}
  .stat:last-child{border-right:0;}
  .stat b{font-family:var(--mono);font-size:1.35rem;font-weight:500;display:block;
    letter-spacing:-.02em;}
  .stat span{font-size:.72rem;color:var(--muted);text-transform:uppercase;
    letter-spacing:.1em;}
  .panel{background:linear-gradient(180deg,var(--panel),var(--panel2));
    border:1px solid var(--line);border-radius:16px;padding:1.5rem;margin-bottom:1.6rem;}
  .k{font-family:var(--mono);font-size:.72rem;letter-spacing:.18em;
    text-transform:uppercase;color:var(--muted);margin:0 0 1rem;}
  .chips{display:grid;grid-template-columns:1fr 1fr;gap:.6rem;}
  .chip{cursor:pointer;text-align:left;padding:.8rem .95rem;border-radius:10px;
    border:1px solid var(--line);background:#0e1620;color:var(--ink);
    font-family:var(--sans);font-size:.95rem;
    transition:border-color .15s,background .15s,transform .06s;}
  .chip:hover{border-color:var(--faint);background:#101c28;}
  .chip:active{transform:translateY(1px);}
  .chip.on{border-color:currentColor;}
  .chip .dot{width:8px;height:8px;border-radius:50%;display:inline-block;
    margin-right:.5rem;vertical-align:middle;}
  .chip .lab{font-family:var(--mono);font-size:.72rem;color:var(--muted);
    display:block;margin-top:.15rem;letter-spacing:.03em;}
  .chip.hccc{color:var(--hcc);} .chip.normc{color:var(--normal);}
  .readout{margin-top:1.4rem;opacity:0;max-height:0;overflow:hidden;
    transition:opacity .45s ease;}
  .readout.show{opacity:1;max-height:none;}
  .heatwrap{margin:.2rem 0 1.4rem;}
  .heat{display:flex;gap:3px;}
  .cell{flex:1 1 0;height:38px;border-radius:3px;background:#16202c;
    opacity:0;transform:translateY(6px);
    transition:opacity .28s ease,transform .28s ease,background .28s ease;}
  .readout.show .cell{opacity:1;transform:none;}
  .heatlabels{display:flex;justify-content:space-between;font-family:var(--mono);
    font-size:.66rem;color:var(--muted);margin-top:.5rem;letter-spacing:.04em;}
  .swatch{display:inline-block;width:9px;height:9px;border-radius:2px;
    vertical-align:middle;margin:0 .25rem;}
  .verdict{display:flex;align-items:baseline;gap:.6rem;flex-wrap:wrap;margin:.2rem 0 .1rem;}
  .verdict .big{font-size:1.55rem;font-weight:600;letter-spacing:-.02em;}
  .verdict.hccv .big{color:var(--hcc);} .verdict.normv .big{color:var(--normal);}
  .verdict .sci{font-family:var(--mono);font-size:.8rem;color:var(--muted);}
  .match{font-family:var(--mono);font-size:.74rem;padding:.15rem .5rem;border-radius:20px;
    border:1px solid var(--line);}
  .match.ok{color:var(--normal);} .match.no{color:var(--hcc);}
  .meter{margin-top:1rem;}
  .meter .track{position:relative;height:12px;border-radius:6px;
    background:linear-gradient(90deg,var(--normal),#16202c 50%,var(--hcc));}
  .meter .needle{position:absolute;top:-5px;width:3px;height:22px;border-radius:2px;
    background:var(--ink);left:0;transition:left .6s cubic-bezier(.2,.8,.2,1);
    box-shadow:0 0 0 2px var(--bg);}
  .meter .ends{display:flex;justify-content:space-between;font-family:var(--mono);
    font-size:.68rem;color:var(--muted);margin-top:.45rem;letter-spacing:.08em;}
  .meter .pval{font-family:var(--mono);font-size:.82rem;color:var(--ink);
    text-align:center;margin-top:.5rem;}
  .steps{display:grid;margin:0 0 1.6rem;}
  .step{display:flex;gap:.9rem;padding:.78rem 0;border-top:1px solid var(--line);}
  .step .n{font-family:var(--mono);font-size:.75rem;color:var(--normal);
    min-width:2rem;padding-top:.15rem;letter-spacing:.05em;}
  .step p{margin:0;color:var(--muted);font-size:.95rem;}
  .step b{color:var(--ink);font-weight:600;}
  footer{border-top:1px solid var(--line);margin-top:2.2rem;padding-top:1.3rem;
    font-family:var(--mono);font-size:.8rem;color:var(--muted);
    display:flex;flex-wrap:wrap;gap:.4rem 1.1rem;align-items:center;}
  footer a{color:var(--muted);text-decoration:none;border-bottom:1px solid var(--line);}
  footer a:hover{color:var(--ink);}
  .hint{font-size:.8rem;color:var(--faint);margin:.9rem 0 0;font-family:var(--mono);}
  @media(max-width:520px){.chips{grid-template-columns:1fr}.cell{height:30px}}
  @media(prefers-reduced-motion:reduce){*{transition:none!important}}
</style></head>
<body>
<div class="wrap">
  <p class="eyebrow">Gene-expression classifier &middot; live demo</p>
  <h1>Read liver <span class="em">cancer</span><br>from the genes that give it away.</h1>
  <p class="lede">A model trained on 357 microarray biopsies (GEO GSE14520) decides
     whether liver tissue is hepatocellular carcinoma or healthy &mdash; from just 20
     gene probes. Pick a real biopsy it never saw and watch it read the expression.</p>

  <div class="stats">
    <div class="stat"><b id="s-auc">&mdash;</b><span>Test ROC-AUC</span></div>
    <div class="stat"><b id="s-genes">&mdash;</b><span>Genes used</span></div>
    <div class="stat"><b id="s-probes">&mdash;</b><span>Probes screened</span></div>
  </div>

  <div class="panel">
    <p class="k">Pick a biopsy sample</p>
    <div class="chips" id="chips"></div>

    <div class="readout" id="readout">
      <div class="heatwrap">
        <div class="heat" id="heat"></div>
        <div class="heatlabels">
          <span><span class="swatch" style="background:var(--normal)"></span>under-expressed</span>
          <span>expression readout &middot; 20 genes</span>
          <span>over-expressed<span class="swatch" style="background:var(--hcc)"></span></span>
        </div>
      </div>
      <div class="verdict" id="verdict"></div>
      <div class="meter">
        <div class="track"><div class="needle" id="needle"></div></div>
        <div class="ends"><span>Normal</span><span>HCC</span></div>
        <div class="pval" id="pval"></div>
      </div>
    </div>
    <p class="hint" id="hint">4 held-out samples &middot; the model never trained on these</p>
  </div>

  <div class="steps">
    <div class="step"><span class="n">01</span><p><b>Screen</b> 22,277 gene probes down to a
       small candidate set with label-free filters &mdash; no test data touched.</p></div>
    <div class="step"><span class="n">02</span><p><b>Select</b> the 20 most informative genes
       with recursive elimination run <b>inside</b> cross-validation &mdash; no selection leakage.</p></div>
    <div class="step"><span class="n">03</span><p><b>Classify</b> with a tuned, scaled logistic
       regression, served here on Cloud Run as a live API.</p></div>
  </div>

  <footer>
    <span id="f-model">model</span>
    <a href="/docs">API docs</a>
    <a href="/model">/model</a>
    <a href="/health">/health</a>
    <a href="https://github.com/shiva-shivanibokka/Cumida-ML-Model">GitHub</a>
  </footer>
</div>

<script>
const DATA = __DATA__;
const HCC='HCC', NORMAL='normal';
const C_NORMAL=[56,198,232], C_NEUT=[24,34,48], C_HCC=[255,93,115];
function mix(a,b,t){return 'rgb('+a.map((v,i)=>Math.round(v+(b[i]-v)*t)).join(',')+')';}
// Diverging colour, saturated by |z|=1.6 so real up/down-regulation is visible.
function divColor(z){const L=1.6,c=Math.max(-L,Math.min(L,z)),t=(c+L)/(2*L);
  return t<0.5?mix(C_NORMAL,C_NEUT,t*2):mix(C_NEUT,C_HCC,(t-0.5)*2);}

const meta=DATA.meta||{};
document.getElementById('s-auc').textContent=meta.roc_auc!=null?meta.roc_auc.toFixed(3):'--';
document.getElementById('s-genes').textContent=(DATA.genes||[]).length||'--';
document.getElementById('s-probes').textContent=meta.n_probes?meta.n_probes.toLocaleString():'--';
document.getElementById('f-model').textContent=(DATA.model_type||'model')+' - '+(meta.n_test||'?')+' test samples';

const chips=document.getElementById('chips');
(DATA.samples||[]).forEach((s,i)=>{
  const b=document.createElement('button');
  b.className='chip '+(s.label===HCC?'hccc':'normc');
  const n=DATA.samples.filter((x,j)=>j<=i&&x.label===s.label).length;
  b.innerHTML='<span class="dot" style="background:currentColor"></span>Biopsy '+
    (s.label===HCC?'T':'N')+n+
    '<span class="lab">confirmed: '+(s.label===HCC?'HCC tumour':'normal tissue')+'</span>';
  b.addEventListener('click',()=>predict(s,b));
  chips.appendChild(b);
});
if(!(DATA.samples||[]).length){
  chips.innerHTML='<p class="hint">No demo samples baked in. Run <code>python train.py</code>.</p>';
}

function buildHeat(features){
  const heat=document.getElementById('heat'); heat.innerHTML='';
  (DATA.genes||[]).forEach((g,i)=>{
    const st=(DATA.stats||{})[g]||{mean:0,std:1};
    const z=(features[g]-st.mean)/(st.std||1);
    const cell=document.createElement('div');
    cell.className='cell';
    cell.style.transitionDelay=(i*28)+'ms';
    requestAnimationFrame(()=>{cell.style.background=divColor(z);});
    cell.title=g+'  -  expr '+Number(features[g]).toFixed(2)+'  -  z '+z.toFixed(2);
    heat.appendChild(cell);
  });
}

let busy=false;
async function predict(sample,btn){
  if(busy)return; busy=true;
  document.querySelectorAll('.chip').forEach(c=>c.classList.remove('on'));
  btn.classList.add('on');
  document.getElementById('readout').classList.add('show');
  document.getElementById('hint').textContent='reading expression...';
  buildHeat(sample.features);
  try{
    const r=await fetch('/predict',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({features:sample.features})});
    const d=await r.json();
    const pct=d.probability_hcc*100, isHCC=d.prediction===HCC, ok=d.prediction===sample.label;
    const v=document.getElementById('verdict');
    v.className='verdict '+(isHCC?'hccv':'normv');
    v.innerHTML='<span class="big">'+(isHCC?'Hepatocellular carcinoma':'Normal liver tissue')+'</span>'+
      '<span class="sci">'+(isHCC?'malignant':'healthy')+'</span>'+
      '<span class="match '+(ok?'ok':'no')+'">'+(ok?'matches diagnosis':'vs confirmed '+sample.label)+'</span>';
    document.getElementById('needle').style.left=pct.toFixed(1)+'%';
    document.getElementById('pval').textContent='P(HCC) = '+pct.toFixed(1)+'%';
    document.getElementById('hint').textContent='20 gene values -> live model -> prediction';
  }catch(e){
    document.getElementById('hint').textContent='Error: '+e;
  }
  busy=false;
}
</script>
</body></html>"""
