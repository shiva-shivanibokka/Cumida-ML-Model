# Deploying the API to Fly.io

The FastAPI service ships as a self-contained Docker image (the trained model is
baked in), so deployment is a handful of commands. This targets **Fly.io**, but
the same image runs on any container platform.

## Prerequisites

1. A trained model exists: `python train.py` (creates `artifacts/model.joblib`).
2. `flyctl` installed — https://fly.io/docs/flyctl/install/
   - macOS/Linux: `curl -L https://fly.io/install.sh | sh`
   - Windows (PowerShell): `iwr https://fly.io/install.ps1 -useb | iex`
3. A Fly.io account.

> In this Claude Code session, run the interactive login by typing it with a
> leading `!` so its output lands in the conversation:
> `! fly auth login`

## One-time setup

```bash
# From the repo root. Creates the Fly app from fly.toml WITHOUT deploying yet.
fly launch --no-deploy
```

If the name `liver-hcc` is taken, edit `app = "..."` in `fly.toml` to something
unique (e.g. `liver-hcc-<yourname>`), and pick a `primary_region` near you
(`fly platform regions` lists them).

## Deploy

```bash
fly deploy
```

Fly builds the Dockerfile, pushes the image, boots a machine, and waits for the
`/health` check to go green. When it finishes it prints your public URL, e.g.
`https://liver-hcc.fly.dev`.

## Verify the live service

```bash
APP=https://liver-hcc.fly.dev          # your URL

curl $APP/health
curl $APP/model                        # lists the 20 gene ids the model expects

# Predict (fill in all 20 gene values from /model)
curl -X POST $APP/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"200910_at": 8.9, "201293_x_at": 10.1, ...}}'
```

Interactive API docs are auto-generated at `$APP/docs`.

## Operations

- **Logs** (your structured JSON prediction events show up here):
  `fly logs`
- **Redeploy after retraining:** `python train.py && fly deploy`
  (the model is baked into the image, so a new model = a new image = versioned).
- **Scale / cost:** `fly.toml` sets `auto_stop_machines = "stop"` and
  `min_machines_running = 0`, so the machine sleeps when idle — friendly to the
  free allowance. First request after idle has a cold-start of a few seconds.
- **Roll back:** `fly releases` then `fly deploy --image <previous-image-ref>`,
  or `fly releases rollback`.

## Notes for the resume

Once live, you can link the public URL and `$APP/docs`. The service demonstrates:
containerization, a health-checked deployment, scale-to-zero cost awareness,
structured logging/observability, and a versioned model artifact — the
production-deployment gap most student portfolios miss.
