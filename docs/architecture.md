# Architecture & Design Decisions

This document explains how the project fits together and records the decisions
that aren't obvious from the code alone. It doubles as a lightweight set of
Architecture Decision Records (ADRs).

## System overview

```
                    ┌──────────────────────────────────────────┐
                    │            src/liver_hcc/ (package)        │
                    │                                            │
 Liver_GSE14520 ──► │  data.py ──► features.py ──► models.py     │
 (22,277 probes)    │   load        label-free      leakage-free │
                    │   split       reduction       tuning       │
                    │                     │              │        │
                    │                     ▼              ▼        │
                    │               evaluate.py    train.py (CLI) │
                    │                                    │        │
                    └────────────────────────────────────┼───────┘
                                                          ▼
                              artifacts/model.joblib + metrics.json
                                                          │
                              ┌───────────────────────────┼───────┐
                              │                            ▼       │
   01–04 notebooks ──────────┤ import          serve.py (FastAPI) │
   (teaching narrative,      │ the same        /health /model     │
    import from the package) │ modules         /predict           │
                              └────────────────────────────────────┘
                                             │
                                    Dockerfile (container)
```

The four notebooks and the training CLI import the **same** modules, so there is
one implementation of every step. Notebooks are the readable narrative; the
package is the source of truth.

## Data flow

1. **Load & split** (`data.py`) — read the CSV, drop the sample-id column,
   stratified 80/20 train/test split with a fixed seed.
2. **Label-free reduction** (`features.py`) — zero-variance filter, high-null
   filter + median impute, and a **raw-scale** variance filter. None use the
   target, so they run once on the training set without leaking anything.
3. **Leakage-free supervised selection + tuning** (`models.py`) — a scikit-learn
   `Pipeline` (`scale → univariate prefilter → RFE → classifier`) is tuned with
   `GridSearchCV` (LR) / `BayesSearchCV` (GB). Because selection lives inside the
   pipeline, it is re-fit within every CV fold.
4. **Deploy** — the genes RFE chose are used to fit a compact
   `StandardScaler → classifier` model on the full training set. This is what's
   saved and served, so the API takes ~30 gene values instead of 22,277.
5. **Serve** (`serve.py`) — FastAPI loads the bundle and answers `/predict`.

---

## ADR-001: Move supervised feature selection inside the CV pipeline

**Context.** The original notebooks ran RFE once on the whole training set, then
cross-validated the model on the pre-selected features. Every CV fold's model
was therefore built on features chosen using that fold's own labels — *selection
leakage* — which inflates cross-validation scores relative to the honest test
score.

**Decision.** Wrap scaling + univariate prefilter + RFE + classifier in a single
`Pipeline` and tune it with the search cross-validator, so selection is re-fit
per fold. The number of selected genes (`rfe__n_features_to_select`) becomes a
tuned hyperparameter rather than a value chosen on all the data.

**Consequences.** CV scores drop to honest levels and track the test score.
Tuning is slower (RFE runs inside every fold), mitigated by a fast univariate
F-test prefilter (`SelectKBest`) that trims to a few hundred genes before RFE.

## ADR-002: Filter variance on raw data, not scaled data

**Context.** The original Step 3 applied `StandardScaler` and *then*
`VarianceThreshold(0.01)`. Standardizing forces every column to variance 1.0, so
the threshold removed nothing — a silent no-op that the README misattributed to
"the data is already clean."

**Decision.** Run the variance filter on raw expression values (it now removes
several thousand near-constant probes), and let `StandardScaler` live *inside*
the model pipeline where it belongs. A regression test
(`test_variance_threshold_after_scaling_is_a_noop`) locks this in.

## ADR-003: Serve a compact model on selected genes, not the full pipeline

**Context.** The tuned pipeline expects the full label-free-reduced feature
matrix (~18k columns). An API requiring 18k values per request is impractical.

**Decision.** After tuning identifies the best genes and hyperparameters, fit a
small `StandardScaler → classifier` on just those genes and save that. The
served contract is a handful of gene values. Reported metrics are for the model
that is actually deployed — you report what you ship.

**Trade-off.** Selection quality still comes from the leakage-free tuning; the
final refit is the standard "select via CV, then fit final model on all training
data" pattern. The test set is never touched by any of it.

## ADR-004: Config-driven Colab/local portability

**Context.** The original notebooks hard-coded a Google Drive path and imported
`google.colab`, so they ran only on Colab — and the README's documented path
didn't even match the code.

**Decision.** `config.py` auto-detects Colab vs local, resolves the dataset from
the repo root locally (or Drive on Colab), and allows env-var overrides
(`LIVER_HCC_DATA_DIR`). `mount_drive_if_colab()` is a no-op off Colab.

---

## What happens at 10× load?

The service is stateless — the model is loaded once and cached in-process, and a
prediction is a single scaled dot-product (LR) or a few hundred shallow-tree
evaluations (GB), i.e. sub-millisecond. Scaling out is therefore horizontal:
run N replicas of the container behind a load balancer; `/health` gives the
orchestrator its readiness signal. The model artifact is baked into the image at
build time, so a retrain is `python train.py` + a source redeploy (Cloud Build
rebuilds and rolls out the image); for fast *local* iteration you can instead
mount over the baked-in file (`-v "$PWD/artifacts:/app/artifacts"`, per the
Dockerfile) to swap models without a rebuild. The realistic bottleneck at high
load is per-request JSON (de)serialization, not inference — batching the
`/predict` endpoint would be the first optimization if needed.

## Testing strategy

- **`test_features.py`** — the variance-filter-on-raw behavior, the scaled no-op
  regression test, and a leakage guard asserting selection precedes the classifier.
- **`test_api.py`** — the `/health`, `/model`, and `/predict` contracts against a
  tiny synthetic model, so the suite runs without the dataset or a real train.

Run with `pytest`.
