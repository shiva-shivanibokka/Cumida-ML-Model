#!/usr/bin/env python
"""Recover which patient each biopsy came from, and commit the answer.

CuMiDa's ``Liver_GSE14520_U133A.csv`` carries one row per biopsy with a GEO
sample accession and a label, and nothing else. What it does not carry is the
fact that GSE14520 is a **paired** study: for most patients it holds both a
tumour biopsy and a matched non-tumour biopsy from the same liver.

That matters because a plain stratified split then puts one of a patient's two
biopsies in train and the other in test. The model has not seen that *sample*,
but it has seen that *liver* -- with the opposite label. Measured on the split
this repo used to make, 48 of 72 held-out biopsies (67%) had their own
patient's other tissue sitting in the training set.

The pairing lives in GEO's series matrix, in ``!Sample_source_name_ch1``, as an
identifier like ``LCS-039A`` / ``LCS-039B`` -- same patient, A = tumour,
B = non-tumour. This script reads the two platform files GSE14520 spans,
recovers the patient behind every accession, and writes a small CSV that
``liver_hcc.data`` uses to group the split.

The map is committed, so training needs no network. Re-run this only to
regenerate it:

    python scripts/build_patient_map.py            # from the committed download
    python scripts/build_patient_map.py --fetch    # re-download from NCBI first
    python scripts/build_patient_map.py --check    # fail if the committed map is stale
"""

from __future__ import annotations

import argparse
import gzip
import re
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from liver_hcc import config

# GSE14520 was run on two Affymetrix platforms and CuMiDa's file draws on both,
# so both series matrices have to be read to cover all 357 biopsies.
SERIES = {
    "GPL571": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE14nnn/GSE14520/matrix/"
              "GSE14520-GPL571_series_matrix.txt.gz",
    "GPL3921": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE14nnn/GSE14520/matrix/"
               "GSE14520-GPL3921_series_matrix.txt.gz",
}

CACHE = REPO / "artifacts" / "geo"
MAP_PATH = REPO / "data" / "patients.csv"

# "LCS-039A" -> patient LCS-039, tissue A. A handful of accessions use a
# different prefix ("X02-342A") but the same trailing A/B convention.
SOURCE_ID = re.compile(r"^(?P<patient>.+?)(?P<tissue>[AB])$")


def download(force: bool = False) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    for platform, url in SERIES.items():
        target = CACHE / f"{platform}.txt.gz"
        if target.exists() and not force:
            print(f"  have {target.name}")
            continue
        print(f"  fetching {platform} ...", flush=True)
        urllib.request.urlretrieve(url, target)


def read_header(path: Path) -> dict[str, list[str]]:
    """Pull the sample-level header rows out of a GEO series matrix.

    The expression table underneath is tens of megabytes and irrelevant here,
    so reading stops at the row that introduces it.
    """
    wanted = ("!Sample_geo_accession", "!Sample_source_name_ch1")
    rows: dict[str, list[str]] = {}
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("!series_matrix_table_begin"):
                break
            for key in wanted:
                if line.startswith(key + "\t"):
                    values = [v.strip('"') for v in line.rstrip("\n").split("\t")[1:]]
                    rows.setdefault(key, []).extend(values)
    return rows


def build() -> list[dict[str, str]]:
    accession_to_source: dict[str, str] = {}
    for platform in SERIES:
        path = CACHE / f"{platform}.txt.gz"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing. Run: python scripts/build_patient_map.py --fetch"
            )
        header = read_header(path)
        pairs = zip(header["!Sample_geo_accession"],
                    header["!Sample_source_name_ch1"], strict=True)
        accession_to_source.update(pairs)

    import pandas as pd

    biopsies = pd.read_csv(config.RAW_CSV, usecols=[config.SAMPLE_COL, config.TARGET_COL])
    out, unmapped = [], []
    for accession_file, label in biopsies.itertuples(index=False):
        accession = accession_file.replace(".CEL.gz", "")
        source = accession_to_source.get(accession)
        match = SOURCE_ID.match(source) if source else None
        if match is None:
            unmapped.append((accession, source))
            continue
        out.append({
            "sample": accession_file,
            "accession": accession,
            "patient": match.group("patient"),
            "tissue": match.group("tissue"),
            "label": label,
        })

    if unmapped:
        raise SystemExit(
            f"{len(unmapped)} biopsies could not be traced to a patient, e.g. "
            f"{unmapped[:3]}. A split grouped on a partial map would silently "
            "leak exactly the samples it failed to place, so this is fatal."
        )
    return out


def write(rows: list[dict[str, str]]) -> str:
    lines = ["sample,accession,patient,tissue,label"]
    lines += [
        f"{r['sample']},{r['accession']},{r['patient']},{r['tissue']},{r['label']}"
        for r in rows
    ]
    return "\n".join(lines) + "\n"


def summarise(rows: list[dict[str, str]]) -> None:
    patients: dict[str, set[str]] = {}
    for r in rows:
        patients.setdefault(r["patient"], set()).add(r["label"])
    both = sum(1 for labels in patients.values() if len(labels) > 1)
    print(f"  {len(rows)} biopsies from {len(patients)} patients")
    print(f"  {both} patients contribute both a tumour and a matched normal")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true",
                        help="re-download the GEO series matrices first")
    parser.add_argument("--check", action="store_true",
                        help="fail if the committed map differs from a fresh build")
    args = parser.parse_args()

    if args.fetch or not (CACHE / "GPL571.txt.gz").exists():
        download(force=args.fetch)

    rows = build()
    summarise(rows)
    text = write(rows)

    if args.check:
        if not MAP_PATH.exists():
            print(f"FAIL  {MAP_PATH} does not exist")
            return 1
        if MAP_PATH.read_text(encoding="utf-8") != text:
            print(f"FAIL  {MAP_PATH} is stale; run: python scripts/build_patient_map.py")
            return 1
        print(f"{MAP_PATH.relative_to(REPO)} is up to date.")
        return 0

    MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    MAP_PATH.write_text(text, encoding="utf-8", newline="\n")
    print(f"  wrote {MAP_PATH.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
