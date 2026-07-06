"""Central configuration: paths, constants, and Colab/local auto-detection.

The whole project reads its paths from here so that the *same* notebooks and
scripts run unchanged whether you are on Google Colab or a local machine.

- On Colab: mounts Google Drive and points at a Drive folder.
- Locally: uses the repository root, so the dataset CSV that already lives in
  the repo is picked up automatically with no path editing.

You can always override any path with an environment variable (see below).
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Column / label constants ------------------------------------------------
TARGET_COL: str = "type"
SAMPLE_COL: str = "samples"
CLASS_POS: str = "HCC"       # positive class (cancer)
CLASS_NEG: str = "normal"    # negative class (healthy tissue)

# --- Reproducibility ---------------------------------------------------------
RANDOM_SEED: int = 42
TEST_SIZE: float = 0.20

# --- File names --------------------------------------------------------------
RAW_CSV_NAME: str = "Liver_GSE14520_U133A.csv"


def _running_in_colab() -> bool:
    """True if executing inside a Google Colab runtime."""
    try:
        import google.colab  # noqa: F401  (import is the probe)

        return True
    except ImportError:
        return False


IN_COLAB: bool = _running_in_colab()


def _default_data_dir() -> Path:
    """Directory that holds the raw dataset and receives generated artifacts.

    Precedence:
      1. LIVER_HCC_DATA_DIR env var (explicit override, works everywhere)
      2. Colab   -> the Drive folder below (mounted on demand)
      3. Local   -> the repository root (two levels up from this file)
    """
    override = os.environ.get("LIVER_HCC_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()

    if IN_COLAB:
        return Path("/content/drive/MyDrive/Colab Notebooks/Cumida Liver ML Project")

    # src/liver_hcc/config.py -> repo root is two parents up.
    return Path(__file__).resolve().parents[2]


def mount_drive_if_colab() -> None:
    """Mount Google Drive when on Colab; no-op locally.

    Call this once at the top of a notebook. It is safe to call repeatedly.
    """
    if IN_COLAB:
        from google.colab import drive  # imported lazily so local runs never need it

        drive.mount("/content/drive")


# --- Resolved paths ----------------------------------------------------------
DATA_DIR: Path = _default_data_dir()
RAW_CSV: Path = DATA_DIR / RAW_CSV_NAME

# Generated intermediates and model artifacts live under artifacts/ so they
# never clutter the repo root and are easy to .gitignore.
ARTIFACTS_DIR: Path = Path(
    os.environ.get("LIVER_HCC_ARTIFACTS_DIR", DATA_DIR / "artifacts")
)

CLEAN_CSV: Path = ARTIFACTS_DIR / "liver_clean.csv"
X_TRAIN_CSV: Path = ARTIFACTS_DIR / "X_train.csv"
X_TEST_CSV: Path = ARTIFACTS_DIR / "X_test.csv"
Y_TRAIN_CSV: Path = ARTIFACTS_DIR / "y_train.csv"
Y_TEST_CSV: Path = ARTIFACTS_DIR / "y_test.csv"

MODEL_PATH: Path = ARTIFACTS_DIR / "model.joblib"
METRICS_PATH: Path = ARTIFACTS_DIR / "metrics.json"


def ensure_artifacts_dir() -> Path:
    """Create the artifacts directory if needed and return it."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS_DIR


def describe() -> str:
    """Human-readable summary of the resolved environment — handy in notebooks."""
    return (
        f"Environment : {'Google Colab' if IN_COLAB else 'local'}\n"
        f"Data dir    : {DATA_DIR}\n"
        f"Raw CSV     : {RAW_CSV}  (exists: {RAW_CSV.exists()})\n"
        f"Artifacts   : {ARTIFACTS_DIR}"
    )
