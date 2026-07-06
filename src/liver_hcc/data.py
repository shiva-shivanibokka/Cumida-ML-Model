"""Data loading, cleaning, and the train/test split.

These functions are deliberately thin wrappers around pandas/sklearn so that
the notebooks can import them and stay focused on *narrative* rather than
plumbing. Nothing here uses the target label to transform features, so calling
any of it on the full dataset does not risk train/test leakage.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Load the raw GSE14520 CSV (357 samples x 22,279 columns)."""
    path = path or config.RAW_CSV
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. On a local machine, keep "
            f"'{config.RAW_CSV_NAME}' in the repo root; on Colab, set "
            "LIVER_HCC_DATA_DIR or place it in your Drive folder."
        )
    return pd.read_csv(path, low_memory=False)


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the GEO sample-id column, which is an identifier, not a feature."""
    if config.SAMPLE_COL in df.columns:
        df = df.drop(columns=[config.SAMPLE_COL])
    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X gene-probe matrix, y target series)."""
    gene_cols = [c for c in df.columns if c != config.TARGET_COL]
    return df[gene_cols].copy(), df[config.TARGET_COL].copy()


def make_split(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split with a fixed seed for reproducibility."""
    return train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_SEED,
    )


def y_to_binary(y: pd.Series) -> pd.Series:
    """Map string labels to 1 (positive/HCC) / 0 (negative/normal)."""
    return (y == config.CLASS_POS).astype(int)
