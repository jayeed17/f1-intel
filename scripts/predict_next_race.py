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
import numpy as np
import pandas as pd

from app.config import CIRCUIT_TYPE, PREDICTIONS_DIR, REG_CHANGE_SEASONS
from app.data import get_session
from app.models.race_predictor import ensure_trained, predict_race
from scripts.build_race_dataset import DATASET_PATH, _team_id


def _quali_features(s) -> pd.DataFrame:
    """One row per driver: driver, team_id, grid (== quali position),
    quali_position, quali_gap_to_pole_s, teammate_quali_gap_s. grid_pit_lane
    is always False here -- not knowable until the race actually starts."""
    laps = s.laps
    results = s.results
    best = laps.groupby("Driver")["LapTime"].min().dt.total_seconds()
    rows = [{
        "driver": r.get("Abbreviation"), "team_id": _team_id(r.get("TeamName")),
        "position_raw": r.get("Position"), "quali_best_s": best.get(r.get("Abbreviation"), np.nan),
    } for _, r in results.iterrows()]
    df = pd.DataFrame(rows)

    # OpenF1Session never has a real Position (see module docstring) -- rank
    # by best lap time instead, which is what a quali position fundamentally is.
    if df["position_raw"].isna().all():
        df["quali_position"] = df["quali_best_s"].rank(method="first")
    else:
        df["quali_position"] = df["position_raw"]

    pole = df["quali_best_s"].min()
    df["quali_gap_to_pole_s"] = df["quali_best_s"] - pole
    gap = {}
    for _, g in df.groupby("team_id"):
        if len(g) != 2:
            continue
        d1, d2 = g.iloc[0], g.iloc[1]
        gap[d1["driver"]] = d1["quali_best_s"] - d2["quali_best_s"]
        gap[d2["driver"]] = d2["quali_best_s"] - d1["quali_best_s"]
    df["teammate_quali_gap_s"] = df["driver"].map(gap)
    df["grid"] = df["quali_position"]
    df["grid_pit_lane"] = False
    return df


def _rolling_snapshot(dataset: pd.DataFrame, driver: str, team_id: str) -> dict:
    """Best-effort 'form entering the next race'. driver_* and
    team_rolling_avg_finish_3 are recomputed fresh from actual past results
    (target_finish_pos is saved in the dataset, so this correctly includes
    each driver/team's most recent race). team_rolling_pace_gap_3 reuses the
    driver's own most recent pre-race value as-is -- one race stale, since
    the raw per-race pace gap isn't persisted, only the already-rolled
    column -- a minor approximation for an inherently approximate exercise.
    """
    dh = dataset[dataset["driver"] == driver].sort_values(["season", "round"])
    th = dataset[dataset["team_id"] == team_id].groupby(["season", "round"])["target_finish_pos"].mean()
    return {
        "driver_rolling_avg_finish_3": dh["target_finish_pos"].tail(3).mean() if not dh.empty else np.nan,
        "driver_rolling_avg_finish_5": dh["target_finish_pos"].tail(5).mean() if not dh.empty else np.nan,
        "driver_dnf_rate_10": dh["dnf"].tail(10).mean() if not dh.empty else np.nan,
        "team_rolling_avg_finish_3": th.tail(3).mean() if not th.empty else np.nan,
        "team_rolling_pace_gap_3": dh["team_rolling_pace_gap_3"].iloc[-1] if not dh.empty else np.nan,
    }


def build_prediction_features(year: int, gp: str, circuit_id: str | None) -> pd.DataFrame:
    s = get_session(year, gp, "Q", telemetry=False)
    df = _quali_features(s)
    dataset = pd.read_parquet(DATASET_PATH)
    snaps = df.apply(lambda r: _rolling_snapshot(dataset, r["driver"], r["team_id"]),
                     axis=1, result_type="expand")
    df = pd.concat([df, snaps], axis=1)
    df["circuit_type"] = CIRCUIT_TYPE.get(circuit_id, "mixed")
    df["reg_change_flag"] = year in REG_CHANGE_SEASONS
    return df


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

    if not DATASET_PATH.exists():
        raise SystemExit(f"No dataset at {DATASET_PATH} -- run scripts/build_race_dataset.py first.")
    pos_pipe, pts_pipe = ensure_trained(DATASET_PATH)

    features = build_prediction_features(year, gp, circuit_id)
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
