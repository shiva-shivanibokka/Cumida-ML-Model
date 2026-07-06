# Deploying the API

The FastAPI service ships as a self-contained Docker image (the trained model is
baked in), so it runs on any container platform. This project is deployed on
**Google Cloud Run**; a Fly.io path is included as an alternative.

**Live:** https://liver-hcc-579593244955.us-central1.run.app

## Prerequisites

1. A trained model exists: `python train.py` (creates `artifacts/model.joblib`,
   which is committed, so this is only needed if you retrain).
2. The image respects `$PORT` (Cloud Run injects `8080`); see the `Dockerfile`.

---

## Google Cloud Run (primary)

Cloud Run builds from source via **Cloud Build** — no local Docker required — and
gives a permanent `*.run.app` URL on a generous perpetual free tier (2M
requests/month, scales to zero).

### One-time project setup

```bash
gcloud auth login
gcloud projects create liver-hcc-portfolio --name "Liver HCC Classifier"
gcloud billing projects link liver-hcc-portfolio --billing-account <YOUR_BILLING_ID>
gcloud config set project liver-hcc-portfolio
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
```

(`gcloud billing accounts list` shows your billing account id.)

### Deploy

```bash
gcloud run deploy liver-hcc \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 512Mi --cpu 1
```

`.gcloudignore` keeps the 134 MB raw CSV out of the upload. The command builds the
Dockerfile, pushes to Artifact Registry, deploys, and prints the service URL.

### Verify

```bash
URL=https://liver-hcc-579593244955.us-central1.run.app   # your URL
curl $URL/health
curl $URL/model                                          # the 20 gene ids it expects
# interactive docs: $URL/docs
```

### Operations

- **Logs** (structured JSON prediction events): `gcloud run services logs read liver-hcc --region us-central1`
- **Redeploy after retraining:** `python train.py && gcloud run deploy liver-hcc --source . --region us-central1`
- **Cost:** `--min-instances 0` (Cloud Run default) means it sleeps when idle; first
  request after idle cold-starts for a few seconds.
- **Roll back:** `gcloud run services update-traffic liver-hcc --to-revisions <REV>=100 --region us-central1`

---

## Fly.io (alternative)

The repo includes a `fly.toml`. Note Fly's current new-account trial is time-limited.

```bash
# Windows install: iwr https://fly.io/install.ps1 -useb | iex
fly auth login
fly apps create <unique-name>       # then set app = "<unique-name>" in fly.toml
fly deploy --remote-only            # remote build, no local Docker
```

`fly.toml` sets scale-to-zero (`auto_stop_machines = "stop"`,
`min_machines_running = 0`) and a `/health` check. Logs: `fly logs`.

---

## For the resume

The live URL + `$URL/docs` demonstrate: containerization, source-to-Cloud-Run
build via Cloud Build, a health-checked deployment, scale-to-zero cost awareness,
structured logging/observability, and a versioned model artifact — the
production-deployment gap most student portfolios miss.
