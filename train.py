#!/usr/bin/env python
"""Shim: `python train.py` from a checkout, without installing anything.

The implementation lives in `liver_hcc.train` so that the `liver-hcc-train`
console script resolves after `pip install .` too.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from liver_hcc.train import main

if __name__ == "__main__":
    main()
