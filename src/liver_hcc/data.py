"""Data loading, cleaning, and the train/test split.

These functions are deliberately thin wrappers around pandas/sklearn so that
the notebooks can import them and stay focused on *narrative* rather than
plumbing. Nothing here uses the target label to transform features, so calling
any of it on the full dataset does not risk train/test leakage.

**The split is grouped by patient, and that is not cosmetic.** GSE14520 is a
paired study: for 165 of its 189 patients it holds both a tumour biopsy and a
matched non-tumour biopsy from the same liver. A plain stratified split puts one
of those two in train and the other in test, so the model has not seen that
*sample* but has seen that *liver* -- carrying the opposite label. On the
stratified split this module used to return, 48 of 72 held-out biopsies (67%)
had their own patient's other tissue in the training set.

Grouping costs about 0.008 F1 on this dataset, which is inside the seed-to-seed
spread -- the tumour/normal signal is large enough that patient identity adds
little. It is fixed anyway, because "biopsies the model never saw" has to be
true rather than nearly true, and because the honest number is the one worth
quoting.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from . import config


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Load the raw CuMiDa GSE14520 CSV (357 samples x 22,279 columns)."""
    path = path or config.RAW_CSV
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. On a local machine, keep "
            f"'{config.RAW_CSV_NAME}' in the repo root; on Colab, set "
            "LIVER_HCC_DATA_DIR or place it in your Drive folder. See the "
            "README section 'How to get the data'."
        )
    return pd.read_csv(path, low_memory=False)


def load_patients(df: pd.DataFrame, path: Path | None = None) -> pd.Series:
    """The patient behind every row of ``df``, aligned to its index.

    Raises rather than falling back to per-sample grouping: a split grouped on a
    partial map leaks exactly the samples the map failed to place, and would do
    it silently.
    """
    path = path or config.PATIENTS_CSV
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Patient map not found at {path}. Build it with: "
            "python scripts/build_patient_map.py --fetch"
        )
    mapping = pd.read_csv(path).set_index("sample")["patient"]
    if config.SAMPLE_COL not in df.columns:
        raise ValueError(
            f"load_patients needs the '{config.SAMPLE_COL}' column, so call it "
            "before basic_clean() drops it."
        )
    patients = df[config.SAMPLE_COL].map(mapping)
    missing = df.loc[patients.isna(), config.SAMPLE_COL].tolist()
    if missing:
        raise ValueError(
            f"{len(missing)} sample(s) are absent from the patient map, e.g. "
            f"{missing[:3]}. Regenerate it: python scripts/build_patient_map.py"
        )
    return patients


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
    X: pd.DataFrame, y: pd.Series, groups: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Patient-grouped, class-stratified train/test split with a fixed seed.

    ``StratifiedGroupKFold`` keeps every one of a patient's biopsies on the same
    side of the split while holding the class balance, which
    ``train_test_split(stratify=...)`` cannot do. The first of five folds is the
    test set, so the ratio matches the 80/20 this project has always used.
    """
    if groups is None:
        raise ValueError(
            "make_split needs patient groups. Pass data.load_patients(df) -- an "
            "ungrouped split puts a patient's tumour in train and their own "
            "matched normal in test."
        )
    splitter = StratifiedGroupKFold(
        n_splits=round(1 / config.TEST_SIZE),
        shuffle=True,
        random_state=config.RANDOM_SEED,
    )
    train_idx, test_idx = next(iter(splitter.split(X, y, groups)))
    return (
        X.iloc[train_idx],
        X.iloc[test_idx],
        y.iloc[train_idx],
        y.iloc[test_idx],
    )


def y_to_binary(y: pd.Series) -> pd.Series:
    """Map string labels to 1 (positive/HCC) / 0 (negative/normal)."""
    return (y == config.CLASS_POS).astype(int)
