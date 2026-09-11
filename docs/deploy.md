# Deploying

There are two artifacts here and only one of them is hosted.

## The demo page — hosted, static, free

`web/` is a Next.js static export. Both trained models are exported as parameters
(`scripts/export_web_artifacts.py`) and evaluated in the browser by `web/lib/model.ts`, so
the page has **no backend at all** — it is a folder of files.

- **Live:** <https://liver-hcc.vercel.app>
- **Host:** Vercel, git-connected, root directory `web`. Every push to `main` redeploys.

```bash
cd web
npm ci
npm run check:golden     # the browser must still reproduce scikit-learn
npm run build            # -> web/out/
```

### Why it is not a hosted API

An earlier version of this project ran the FastAPI service on Google Cloud Run, under a free
trial with an expiry date. That is the failure mode a portfolio link cannot afford: the trial
lapses, the service stops, and a recruiter clicking the link sees nothing. It also answered
cold requests in ~18 seconds.

Neither model needs a server. Logistic regression is a dot product behind a sigmoid; the
gradient booster is 180 depth-3 trees. Running them client-side is exact — CI holds the
TypeScript to scikit-learn's own answers on all 72 held-out biopsies, currently agreeing to
6e-13 — and removes the host, the cold start and the expiry in one move.

## The API — local and Docker

`serve.py` is the serving layer: typed request/response schemas, structured JSON logging of
every prediction, a `/health` probe, and OpenAPI docs. It runs locally or in a container and
is exercised by `tests/test_api.py` in CI. It is not publicly hosted.

```bash
uvicorn liver_hcc.serve:app --reload          # http://localhost:8000/docs

docker build -t liver-hcc .                   # model baked into the image
docker run -p 8000:8000 liver-hcc
```

The image installs only `.[serve]` — pandas, numpy, scikit-learn, joblib, fastapi, uvicorn —
and deliberately not the plotting or tuning libraries the runtime never imports. It listens
on `$PORT` when a platform provides one, so it will run on any container host if you ever
want it to.

```bash
curl localhost:8000/health
curl localhost:8000/model                      # the exact genes it expects
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \
     -d '{"features": {"200910_at": 8.9, "...": 0.0}}'
```

## Regenerating what is committed

The artifacts are small and committed so the repo is demo-able without the 128 MB dataset.
Each generator has a `--check` mode that fails instead of overwriting, and CI runs the one
that needs no dataset.

| Command | Writes | Needs the dataset? |
|---|---|---|
| `python scripts/build_patient_map.py` | `data/patients.csv` | yes (+ network once) |
| `python train.py` | `artifacts/model.joblib`, `metrics.json`, `examples.json`, `split.json` | yes |
| `python scripts/evaluate_honestly.py` | `artifacts/honesty.json` | yes |
| `python scripts/export_web_artifacts.py` | `web/public/data/*.json` | **no** — runs in CI |
