# Serving image for the trained liver HCC classifier.
# Build:  docker build -t liver-hcc .
# Run:    docker run -p 8000:8000 liver-hcc
#
# The trained model (artifacts/model.joblib) is baked into the image at build
# time, so the container is self-contained and deploys anywhere (Fly.io, a cloud
# run service, etc.) with no volume mounts. Produce the model first with
# `python train.py`, then build. To iterate on the model without rebuilding,
# mount over it locally:  -v "$PWD/artifacts:/app/artifacts".
FROM python:3.11-slim

WORKDIR /app

# Package source + metadata, then install the package with ONLY its serving deps.
# `.[serve]` pulls the lean core (pandas/numpy/scikit-learn/joblib) plus
# fastapi/uvicorn/pydantic — and deliberately NOT the training/notebook libs
# (matplotlib, seaborn, scikit-optimize), which the runtime never imports. That
# keeps the image small and Cloud Run cold starts fast.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir ".[serve]"

# Bake in the trained model + demo samples (tiny — a few KB each). Build fails
# here if you haven't run `python train.py` yet, which is the desired safety check.
COPY artifacts/model.joblib ./artifacts/model.joblib
COPY artifacts/examples.json ./artifacts/examples.json

EXPOSE 8000

# Listen on $PORT when the platform provides one (Google Cloud Run injects
# PORT=8080), else default to 8000 for local runs and Fly.
ENV PORT=8000

# Simple container healthcheck hitting the app's own /health endpoint.
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\",\"8000\")}/health').status==200 else 1)"

CMD ["sh", "-c", "uvicorn liver_hcc.serve:app --host 0.0.0.0 --port ${PORT:-8000}"]
