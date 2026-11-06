"""Predict the next race's outcome from qualifying results. Run after
qualifying, before the race:

    python -m scripts.predict_next_race --year 2026 --gp "Las Vegas Grand Prix"
    python -m scripts.predict_next_race   # auto-detects a ready-to-predict
                                          # qualifying session (for cron use)

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

Auto-detect mode (cron use, hourly across Sat/Sun -- see update-data.yml)
scans the season schedule for an event whose Qualifying session ended at
least 90 minutes ago, whose Race hasn't started yet, and that doesn't
already have a logged prediction -- then runs for that one. Exits 0 with a
message (not a failure) if nothing currently matches -- a bye week, an
early-in-the-weekend tick, or a race that's already been predicted is not
an error. A fixed single-time cron ("quali's surely done by 20:00 UTC")
broke for the 2026 US GP (COTA), whose Qualifying runs 21:00-22:00 UTC --
this re-checks every hour instead of assuming one global offset. Never
backfills a prediction after the race has started, by design (see
app.data.race_has_started()) -- a race that was missed stays missed,
logged honestly via app.predictions_log.log_missed_prediction() rather
than silently predicted after the fact. If a qualifying session DID happen
but its data can't be fetched from any source, that's a real failure and
this exits non-zero.

Uses ONLY the frozen model (load_frozen_model(), data/model/frozen/) --
never a freshly self-trained one -- so every logged prediction comes from
the exact same model, making predictions/{year}.csv's post-freeze rows a
genuinely clean prospective track record (see the dashboard/README's "Live
track record since {frozen_at}" section). Exits non-zero if nothing has
been frozen yet (see scripts/freeze_race_predictor.py) rather than
silently falling back to a different model.
"""
from __future__ import annotations

import argparse

import fastf1
import pandas as pd

from app.config import PREDICTIONS_DIR
from app.data import race_has_started, race_prediction_features, session_readiness
from app.models.race_predictor import frozen_model_spec, load_frozen_model, predict_race
from app.predictions_log import log_missed_prediction, prediction_already_logged

_QUALI_BUFFER = pd.Timedelta(minutes=90)


def _quali_start(row: pd.Series) -> pd.Timestamp | None:
    """This schedule row's Qualifying session start (UTC), or None if it
    has no Qualifying date listed."""
    for i in range(1, 6):
        if row.get(f"Session{i}") != "Qualifying":
            continue
        d = row.get(f"Session{i}DateUtc")
        if pd.isna(d):
            return None
        d = pd.Timestamp(d)
        return d.tz_localize("UTC") if d.tzinfo is None else d.tz_convert("UTC")
    return None


def _auto_detect_next_race(now: pd.Timestamp, since: str) -> tuple[int, str, str | None] | None:
    """The event whose Qualifying ended >= 90 minutes ago, whose Race
    hasn't started, and that doesn't already have a logged prediction --
    "the race to predict" for unattended, hourly cron use. None if nothing
    currently matches (a bye week, an early-in-the-weekend tick, or a race
    that's already been predicted -- none of those are a failure).

    `since` is the frozen model's own frozen_at -- races whose Qualifying
    happened before the model even existed are skipped entirely (not
    considered, not logged as "missed"). Without this, every pre-freeze
    race in the season (which this model was never meant to predict) would
    get wrongly flagged as a pipeline failure the first time this ran."""
    try:
        sch = fastf1.get_event_schedule(now.year, include_testing=False)
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"Could not fetch the {now.year} schedule to auto-detect the next race: {e}") from e

    since_ts = pd.Timestamp(since)
    since_ts = since_ts.tz_localize("UTC") if since_ts.tzinfo is None else since_ts.tz_convert("UTC")
    log_path = PREDICTIONS_DIR / f"{now.year}.csv"
    for _, row in sch.iterrows():
        gp = row["EventName"]
        ready, _ = session_readiness(now, now.year, gp, ("Qualifying",), _QUALI_BUFFER)
        if not ready:
            continue
        quali_start = _quali_start(row)
        if quali_start is None or quali_start < since_ts:
            continue
        already_logged = prediction_already_logged(log_path, gp)
        if race_has_started(now, now.year, gp):
            if not already_logged:
                # Quali ended, the race already started, and we never logged
                # a prediction in between -- a genuine pipeline failure, not
                # a bye week. Record it honestly instead of silently
                # skipping it forever or backfilling a prediction now that
                # the outcome is no longer a real forecast.
                log_missed_prediction(
                    now.year, gp, "race_v1", "post_quali",
                    "quali ended and the race started before a prediction was ever logged "
                    "(pipeline failure) -- not backfilled")
            continue
        if already_logged:
            continue
        # FastF1's own schedule has no circuitId (that's Ergast-only);
        # circuit_type falls back to "mixed" here rather than adding
        # another network call just for a low-importance feature.
        return now.year, gp, None
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, default=None)
    p.add_argument("--gp", default=None)
    p.add_argument("--circuit-id", default=None)
    args = p.parse_args()

    try:
        delta_pipe, dnf_pipe, meta = load_frozen_model()
    except FileNotFoundError as e:
        raise SystemExit(str(e)) from e

    if args.year and args.gp:
        year, gp, circuit_id = args.year, args.gp, args.circuit_id
    else:
        detected = _auto_detect_next_race(pd.Timestamp.now(tz="UTC"), since=frozen_model_spec()["frozen_at"])
        if detected is None:
            print("Nothing ready to predict right now (no quali ended >= 90min ago with a "
                 "race still to come and no prediction logged yet) -- nothing to do.")
            return
        year, gp, circuit_id = detected

    features = race_prediction_features(year, gp, circuit_id)
    preds = predict_race(delta_pipe, dnf_pipe, features, meta)

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = PREDICTIONS_DIR / f"{year}.csv"
    out = preds[["driver", "team_id", "grid", "predicted_position",
                "win_probability", "podium_probability", "points_probability"]].copy()
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
