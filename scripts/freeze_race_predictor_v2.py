"""Freeze race predictor v2 (v1's position + DNF models, position model
extended with circuit-history features) and commit it as the frozen
snapshot every v2 consumer uses from here on. v1 stays frozen and running
unchanged -- this is a separate, additional artifact, not a replacement.

Run locally, deliberately, rarely -- NOT wired into any GitHub Action:

    python -m scripts.freeze_race_predictor_v2 --version 2.0.0 --notes "..."
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from app.config import RACE_DATASET_PATH
from app.models.race_predictor_v2 import freeze_model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--version", default="2.0.0", help="Version tag for this frozen snapshot.")
    p.add_argument("--notes", default="", help="Free-text note recorded in spec.json.")
    args = p.parse_args()

    if not RACE_DATASET_PATH.exists():
        raise SystemExit(f"No dataset at {RACE_DATASET_PATH} -- run scripts/build_race_dataset.py first.")
    df = pd.read_parquet(RACE_DATASET_PATH)
    spec = freeze_model(df, RACE_DATASET_PATH, version=args.version, notes=args.notes)
    print(json.dumps(spec, indent=2))
    print(f"\nFrozen race v{spec['version']} at {spec['frozen_at']}. "
         "Commit data/model/frozen/ to make this the live model.")


if __name__ == "__main__":
    main()
