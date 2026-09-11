#!/usr/bin/env python
"""Export everything the browser demo needs, so it needs no backend.

The page runs the trained models in the visitor's browser. That is possible
because both of them are small and closed-form once fitted:

* **Logistic Regression** is a dot product behind a sigmoid. Ten coefficients,
  the scaler's mean and scale, an intercept.
* **Gradient Boosting** is 180 depth-3 regression trees. Each tree is a handful
  of ``feature <= threshold`` comparisons, so evaluating the ensemble is a walk
  down 180 short paths and a sum -- not something that needs Python.

Neither is an approximation of the fitted model; they are the fitted model's own
parameters, and ``golden.json`` pins scikit-learn's answers for all 72 held-out
biopsies so CI can prove the JavaScript still reproduces them.

Usage:
    python scripts/export_web_artifacts.py
    python scripts/export_web_artifacts.py --check   # fail if committed files are stale
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from liver_hcc import config

OUT = REPO / "web" / "public" / "data"

# Rounding is a correctness knob, not a formatting one: a scaler statistic
# rounded too hard moves predictions. 12 digits keeps every golden prediction
# inside 1e-9 while staying readable in the committed file.
DIGITS = 12


def r(x) -> float:
    return round(float(x), DIGITS)


def export_logistic(pipe, genes: list[str]) -> dict:
    scaler, clf = pipe.named_steps["scaler"], pipe.named_steps["clf"]
    return {
        "kind": "logistic",
        "genes": list(genes),
        "mean": [r(v) for v in scaler.mean_],
        "scale": [r(v) for v in scaler.scale_],
        "coef": [r(v) for v in clf.coef_[0]],
        "intercept": r(clf.intercept_[0]),
    }


def export_boosting(pipe, genes: list[str], X_probe: np.ndarray) -> dict:
    """Flatten the ensemble into arrays a tree walk can read.

    The constant the stages are added to (sklearn's prior log-odds) is recovered
    from the public API rather than reached for privately: subtract the summed
    stage output from ``decision_function``. It is verified constant across
    every biopsy before it is written, because if it were not, this export
    would be wrong in a way no single spot-check would reveal.
    """
    scaler, clf = pipe.named_steps["scaler"], pipe.named_steps["clf"]
    scaled = scaler.transform(X_probe)

    trees = []
    stage_total = np.zeros(len(scaled))
    for stage in clf.estimators_:
        tree = stage[0].tree_
        trees.append({
            "feature": [int(v) for v in tree.feature],
            "threshold": [r(v) for v in tree.threshold],
            "left": [int(v) for v in tree.children_left],
            "right": [int(v) for v in tree.children_right],
            "value": [r(v) for v in tree.value[:, 0, 0]],
        })
        stage_total += stage[0].predict(scaled)

    offsets = clf.decision_function(scaled) - clf.learning_rate * stage_total
    spread = float(offsets.max() - offsets.min())
    if spread > 1e-9:
        raise SystemExit(
            f"the ensemble's initial offset varies by {spread:g} across biopsies; "
            "it is treated as a constant here, so this export would be wrong"
        )

    return {
        "kind": "boosting",
        "genes": list(genes),
        "mean": [r(v) for v in scaler.mean_],
        "scale": [r(v) for v in scaler.scale_],
        "init": r(offsets[0]),
        "learning_rate": r(clf.learning_rate),
        "trees": trees,
    }


def build() -> dict[str, str]:
    bundle = joblib.load(config.MODEL_PATH)
    metrics = json.loads(config.METRICS_PATH.read_text(encoding="utf-8"))
    examples = json.loads(config.EXAMPLES_PATH.read_text(encoding="utf-8"))
    honesty = json.loads((REPO / "artifacts" / "honesty.json").read_text(encoding="utf-8"))
    patients = pd.read_csv(config.PATIENTS_CSV).set_index("sample")
    split = json.loads(config.SPLIT_PATH.read_text(encoding="utf-8"))

    # Every gene either model needs, so the page can hold one row per biopsy.
    all_genes: list[str] = []
    for spec in bundle["all_models"].values():
        all_genes += [g for g in spec["genes"] if g not in all_genes]

    biopsies = []
    for sample, row in zip(split["test"], examples["samples"], strict=True):
        biopsies.append({
            "id": sample.replace(".CEL.gz", ""),
            "patient": str(patients.loc[sample, "patient"]),
            "tissue": str(patients.loc[sample, "tissue"]),
            "label": row["label"],
            "values": {g: row["features"][g] for g in all_genes},
        })

    models = {}
    for name, spec in bundle["all_models"].items():
        frame = pd.DataFrame(
            [[b["values"][g] for g in spec["genes"]] for b in biopsies],
            columns=spec["genes"],
        )
        clf = spec["model"].named_steps["clf"]
        if hasattr(clf, "coef_"):
            models[name] = export_logistic(spec["model"], spec["genes"])
        else:
            models[name] = export_boosting(spec["model"], spec["genes"], frame)

    # scikit-learn's own answers, to be reproduced by the shipped TypeScript.
    golden = {}
    for name, spec in bundle["all_models"].items():
        frame = pd.DataFrame(
            [[b["values"][g] for g in spec["genes"]] for b in biopsies],
            columns=spec["genes"],
        )
        pipe = spec["model"]
        prob = pipe.predict_proba(frame)[:, list(pipe.classes_).index(1)]
        golden[name] = [r(p) for p in prob]

    # Where the two models actually disagree. Their confusion matrices are
    # byte-identical, which is not the same as agreeing -- they are each wrong
    # three times and share only two of those mistakes. Computed, because the
    # first version of this page asserted "the same 72 answers" and was wrong.
    verdicts, agreement = {}, {}
    for name, spec in bundle["all_models"].items():
        frame = pd.DataFrame(
            [[b["values"][g] for g in spec["genes"]] for b in biopsies],
            columns=spec["genes"],
        )
        pipe = spec["model"]
        prob = pipe.predict_proba(frame)[:, list(pipe.classes_).index(1)]
        verdicts[name] = (prob >= 0.5).astype(int)
    truth = np.array([1 if b["label"] == config.CLASS_POS else 0 for b in biopsies])
    (name_a, pred_a), (name_b, pred_b) = verdicts.items()
    disputed = [
        {
            "id": biopsies[i]["id"],
            "truth": biopsies[i]["label"],
            name_a: float(golden[name_a][i]),
            name_b: float(golden[name_b][i]),
        }
        for i in range(len(truth)) if pred_a[i] != pred_b[i]
    ]
    agreement = {
        "n": len(truth),
        "disagreements": len(disputed),
        "disputed": disputed,
        "errors": {n: [biopsies[i]["id"] for i in range(len(truth)) if p[i] != truth[i]]
                   for n, p in verdicts.items()},
        "shared_errors": sorted(
            {biopsies[i]["id"] for i in range(len(truth)) if pred_a[i] != truth[i]}
            & {biopsies[i]["id"] for i in range(len(truth)) if pred_b[i] != truth[i]}
        ),
    }

    # Which genes carry signal, and which were selected but contribute nothing.
    weights = {}
    for name, spec in bundle["all_models"].items():
        clf = spec["model"].named_steps["clf"]
        raw = clf.coef_[0] if hasattr(clf, "coef_") else clf.feature_importances_
        weights[name] = {
            "signed": bool(hasattr(clf, "coef_")),
            "genes": [{"gene": g, "weight": r(w)}
                      for g, w in zip(spec["genes"], raw, strict=True)],
        }

    payload = {
        "models.json": {"models": models, "winner": bundle["model_type"]},
        "biopsies.json": {"genes": all_genes, "biopsies": biopsies,
                          "stats": {g: examples["stats"][g] for g in all_genes
                                    if g in examples["stats"]}},
        "golden.json": {"probabilities": golden},
        "summary.json": {
            "metrics": metrics,
            "honesty": honesty,
            "agreement": agreement,
            "weights": weights,
            "dataset": {
                "name": "CuMiDa Liver_GSE14520_U133A",
                "source": "GEO GSE14520 (GPL571 / GPL3921), curated by CuMiDa",
                "samples": len(patients),
                "probes": metrics["n_probes"],
                "patients": metrics["split"]["n_patients_total"],
            },
        },
    }
    return {name: json.dumps(body, indent=1, allow_nan=False) + "\n"
            for name, body in payload.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="fail if the committed web data is out of date")
    args = parser.parse_args()

    written = build()

    if args.check:
        stale = [
            name for name, text in written.items()
            if not (OUT / name).exists()
            or (OUT / name).read_text(encoding="utf-8") != text
        ]
        for name in stale:
            print(f"FAIL  web/public/data/{name} is stale")
            current = ((OUT / name).read_text(encoding="utf-8").splitlines()
                       if (OUT / name).exists() else [])
            fresh = written[name].splitlines()
            for n, (a, b) in enumerate(zip(current, fresh, strict=False), 1):
                if a != b:
                    print(f"      first difference at line {n}")
                    print(f"      committed: {a.strip()[:110]}")
                    print(f"      rebuilt:   {b.strip()[:110]}")
                    break
            else:
                print(f"      {len(current)} vs {len(fresh)} lines long")
        if stale:
            print("      run: python scripts/export_web_artifacts.py")
            return 1
        print(f"{len(written)} web data files are up to date.")
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    for name, text in written.items():
        (OUT / name).write_text(text, encoding="utf-8", newline="\n")
        print(f"  wrote web/public/data/{name}  ({len(text) / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
