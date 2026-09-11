"""Guards on the patient-grouped split.

These run without the 128 MB dataset, on purpose. The split assignment and the
patient map are both committed, so CI can re-derive who ended up where and check
it -- rather than trusting the training script's own report that it behaved.
"""

import json

import numpy as np
import pandas as pd
import pytest

from liver_hcc import config, data

PATIENTS = pd.read_csv(config.PATIENTS_CSV)
SPLIT = json.loads(config.SPLIT_PATH.read_text(encoding="utf-8"))


def _patients_of(samples):
    lookup = PATIENTS.set_index("sample")["patient"]
    return set(lookup.loc[list(samples)])


def test_patient_map_covers_every_biopsy():
    assert len(PATIENTS) == 357
    assert PATIENTS["sample"].is_unique
    assert set(PATIENTS["tissue"]) == {"A", "B"}          # tumour / non-tumour
    assert set(PATIENTS["label"]) == {config.CLASS_POS, config.CLASS_NEG}


def test_the_committed_split_shares_no_patient():
    """The claim the README makes, checked against the split it was made on."""
    train, test = SPLIT["train"], SPLIT["test"]
    assert len(train) + len(test) == len(PATIENTS)
    assert not set(train) & set(test)
    shared = _patients_of(train) & _patients_of(test)
    assert not shared, f"{len(shared)} patient(s) on both sides, e.g. {sorted(shared)[:3]}"


def test_a_stratified_split_would_have_leaked():
    """The guard above is only meaningful if the obvious alternative fails it.

    This is the split the project used to make -- stratified on the label, blind
    to the patient. It needs no expression data to reproduce: the leak is a
    property of *which rows* land where, and the committed map has that.
    """
    from sklearn.model_selection import train_test_split

    y = PATIENTS["label"]
    train_idx, test_idx = train_test_split(
        PATIENTS.index, test_size=config.TEST_SIZE, stratify=y,
        random_state=config.RANDOM_SEED,
    )
    train_pat = set(PATIENTS.loc[train_idx, "patient"])
    leaked = PATIENTS.loc[test_idx, "patient"].isin(train_pat)
    assert leaked.mean() > 0.5, (
        "expected the ungrouped split to leak most of its test set; if this "
        "ever stops being true the grouped split has stopped being necessary"
    )


def test_make_split_keeps_every_group_whole():
    rng = np.random.default_rng(0)
    n_patients = 60
    rows = []
    for p in range(n_patients):
        for label in (config.CLASS_POS, config.CLASS_NEG):
            rows.append({"patient": f"P{p}", "label": label,
                         "x0": rng.normal(), "x1": rng.normal()})
    frame = pd.DataFrame(rows)
    X = frame[["x0", "x1"]]
    y = frame["label"]
    groups = frame["patient"]

    X_train, X_test, _y_train, y_test = data.make_split(X, y, groups)

    assert len(X_train) + len(X_test) == len(frame)
    assert not set(groups.loc[X_train.index]) & set(groups.loc[X_test.index])
    assert set(y_test) == {config.CLASS_POS, config.CLASS_NEG}   # still stratified


def test_make_split_refuses_to_run_ungrouped():
    """Passing no groups must fail loudly; the old signature silently leaked."""
    X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    y = pd.Series([config.CLASS_POS, config.CLASS_NEG] * 2)
    with pytest.raises(ValueError, match="patient"):
        data.make_split(X, y, None)


def test_load_patients_rejects_an_incomplete_map(tmp_path):
    """A partial map would leak exactly the rows it failed to place."""
    partial = tmp_path / "patients.csv"
    partial.write_text(
        "sample,accession,patient,tissue,label\n"
        "GSM362958.CEL.gz,GSM362958,LCS-079,A,HCC\n",
        encoding="utf-8",
    )
    df = pd.DataFrame({
        config.SAMPLE_COL: ["GSM362958.CEL.gz", "GSM999999.CEL.gz"],
        config.TARGET_COL: [config.CLASS_POS, config.CLASS_NEG],
    })
    with pytest.raises(ValueError, match="absent from the patient map"):
        data.load_patients(df, path=partial)
