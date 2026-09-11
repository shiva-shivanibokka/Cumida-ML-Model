# Serving image for the trained liver HCC classifier.
# Build:  docker build -t liver-hcc .
# Run:    docker run -p 8000:8000 liver-hcc
#
# The trained model (artifacts/model.joblib) is baked into the image at build
# time, so the container is self-contained and needs no volume mounts. Produce
# the model first with `python train.py`, then build. To iterate on the model
# without rebuilding, mount over it:  -v "$PWD/artifacts:/app/artifacts".
#
# This image is not deployed anywhere. The public demo is a static page that
# runs the same models in the browser (see docs/deploy.md); this exists so the
# API can be run locally, and so the serving layer is a real artifact rather
# than a description of one.
FROM python:3.11-slim

WORKDIR /app

# Package source + metadata, then install the package with ONLY its serving deps.
# `.[serve]` pulls the lean core (pandas/numpy/scikit-learn/joblib) plus
# fastapi/uvicorn/pydantic — and deliberately NOT the training/notebook libs
# (matplotlib, seaborn, scikit-optimize), which the runtime never imports. That
# keeps the image small and start-up fast.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir ".[serve]"

# Bake in the trained model + demo samples (tiny — a few KB each). Build fails
# here if you haven't run `python train.py` yet, which is the desired safety check.
COPY artifacts/model.joblib ./artifacts/model.joblib
COPY artifacts/examples.json ./artifacts/examples.json

EXPOSE 8000

# Listen on $PORT when a platform provides one, else default to 8000 locally.
ENV PORT=8000

# Simple container healthcheck hitting the app's own /health endpoint.
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\",\"8000\")}/health').status==200 else 1)"

CMD ["sh", "-c", "uvicorn liver_hcc.serve:app --host 0.0.0.0 --port ${PORT:-8000}"]
