"""Freeze the race predictor: train the final delta/DNF pipelines (weights
on the full committed dataset; Monte Carlo noise-by-grid-bucket + P(win)/
P(podium) isotonic calibration fit on 2022-2024 CV only -- see
app.models.race_predictor's module docstring) and commit them as the one
frozen snapshot every other consumer (the API route, the dashboard,
scripts/predict_next_race.py) uses from here on.

Run locally, deliberately, rarely -- NOT wired into any GitHub Action.
Freezing is a manual decision that starts a clean prospective track record
(see the dashboard/README's "Live track record since {frozen_at}"
section), not something that should silently re-happen on a schedule:

    python -m scripts.freeze_race_predictor --version 1.0.0 --notes "..."

Re-running this OVERWRITES the previous frozen snapshot and resets the live
track record's start date -- only do that for a deliberate new model
version, never as a routine retrain.
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from app.config import RACE_DATASET_PATH
from app.models.race_predictor import freeze_model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--version", default="1.0.0", help="Version tag for this frozen snapshot.")
    p.add_argument("--notes", default="", help="Free-text note recorded in spec.json.")
    args = p.parse_args()

    if not RACE_DATASET_PATH.exists():
        raise SystemExit(f"No dataset at {RACE_DATASET_PATH} -- run scripts/build_race_dataset.py first.")
    df = pd.read_parquet(RACE_DATASET_PATH)
    spec = freeze_model(df, RACE_DATASET_PATH, version=args.version, notes=args.notes)
    print(json.dumps(spec, indent=2))
    print(f"\nFrozen v{spec['version']} at {spec['frozen_at']}. "
         "Commit data/model/frozen/ to make this the live model.")


if __name__ == "__main__":
    main()
