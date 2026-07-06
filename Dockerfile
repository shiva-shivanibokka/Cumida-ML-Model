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

# Install dependencies first so Docker layer-caching skips this on code-only changes.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Package source + metadata, then install the package itself.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps -e .

# Bake in the trained model (tiny — a few KB). Build fails here if you haven't
# run `python train.py` yet, which is the desired safety check.
COPY artifacts/model.joblib ./artifacts/model.joblib

EXPOSE 8000

# Simple container healthcheck hitting the app's own /health endpoint.
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "liver_hcc.serve:app", "--host", "0.0.0.0", "--port", "8000"]
