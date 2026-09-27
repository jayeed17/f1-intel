"""Predict the next race's outcome from qualifying results. Run after
qualifying, before the race:

    python -m scripts.predict_next_race --year 2026 --gp "Las Vegas Grand Prix"
    python -m scripts.predict_next_race   # auto-detects the most recently
                                          # completed qualifying (for cron use)

Quali results come from app.data.get_session()'s existing fallback chain
(prebuilt -> OpenF1 -> live FastF1) rather than a reimplementation -- it's
already the source of truth for "how do we get a session from wherever it's
actually available" in this project. OpenF1 matters most here in practice:
this runs right after qualifying, before this week's race is even in
data/prebuilt/, and GitHub's hosted runner has confirmed trouble reaching
F1's own live timing API -- OpenF1's laps/results endpoints are what
actually work from the cloud in that case (its Position column is always
None, so quali_position falls back to ranking by best lap time instead).

Grid position isn't official until stewards finalise any penalties
(sometimes hours after qualifying) -- this assumes grid == qualifying
classification, usually right but occasionally wrong if a penalty lands
later. Appends one row per driver to predictions/{year}.csv with a
timestamp; scripts/score_predictions.py fills in the actual result once the
race has happened.

Auto-detect mode exits 0 with a message (not a failure) if no qualifying
session happened in the last 2 days -- a bye week is not an error. If a
qualifying session DID happen but its data can't be fetched from any
source, that's a real failure and this exits non-zero.
"""
from __future__ import annotations

import argparse

import fastf1
import pandas as pd

from app.config import PREDICTIONS_DIR, RACE_DATASET_PATH
from app.data import race_prediction_features
from app.models.race_predictor import ensure_trained, predict_race


def _auto_detect_next_race(now: pd.Timestamp) -> tuple[int, str, str | None] | None:
    """The event whose Qualifying session completed in the last 2 days --
    "the race to predict" for unattended cron use. None if no qualifying
    happened in that window (a bye week, not a failure)."""
    try:
        sch = fastf1.get_event_schedule(now.year, include_testing=False)
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"Could not fetch the {now.year} schedule to auto-detect the next race: {e}") from e

    for _, row in sch.iterrows():
        for i in range(1, 6):
            if row.get(f"Session{i}") != "Qualifying":
                continue
            q_date = row.get(f"Session{i}DateUtc")
            if pd.isna(q_date):
                continue
            q_date = pd.Timestamp(q_date)
            q_date = q_date.tz_localize("UTC") if q_date.tzinfo is None else q_date
            if pd.Timedelta(0) <= (now - q_date) <= pd.Timedelta(days=2):
                # FastF1's own schedule has no circuitId (that's Ergast-only);
                # circuit_type falls back to "mixed" here rather than adding
                # another network call just for a low-importance feature.
                return now.year, row["EventName"], None
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, default=None)
    p.add_argument("--gp", default=None)
    p.add_argument("--circuit-id", default=None)
    args = p.parse_args()

    if args.year and args.gp:
        year, gp, circuit_id = args.year, args.gp, args.circuit_id
    else:
        detected = _auto_detect_next_race(pd.Timestamp.now(tz="UTC"))
        if detected is None:
            print("No qualifying session completed in the last 2 days -- nothing to predict this weekend.")
            return
        year, gp, circuit_id = detected

    if not RACE_DATASET_PATH.exists():
        raise SystemExit(f"No dataset at {RACE_DATASET_PATH} -- run scripts/build_race_dataset.py first.")
    pos_pipe, pts_pipe = ensure_trained(RACE_DATASET_PATH)

    features = race_prediction_features(year, gp, circuit_id)
    preds = predict_race(pos_pipe, pts_pipe, features)

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = PREDICTIONS_DIR / f"{year}.csv"
    out = preds[["driver", "team_id", "grid", "predicted_position", "points_probability"]].copy()
    out.insert(0, "predicted_at", pd.Timestamp.now(tz="UTC").isoformat())
    out.insert(1, "gp", gp)
    out["actual_position"] = pd.NA
    out["scored_at"] = pd.NA

    if path.exists():
        out = pd.concat([pd.read_csv(path), out], ignore_index=True)
    out.to_csv(path, index=False)
    print(f"Logged {len(preds)} predictions for {year} {gp} to {path}")


if __name__ == "__main__":
    main()
