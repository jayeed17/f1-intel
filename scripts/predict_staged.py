"""Log pre-race qualifying + race v2 predictions at one of three stages,
each to its own predictions/{year}_{model}_{stage}.csv so each stage's
live track record can be scored independently of the others:

    python -m scripts.predict_staged --stage forecast        # Monday cron
    python -m scripts.predict_staged --stage post_practice    # hourly Sat/Sun cron
    python -m scripts.predict_staged --stage post_quali       # hourly Sat/Sun cron
    python -m scripts.predict_staged --stage forecast --year 2026 --gp "..."  # manual

"forecast" and "post_practice" use app.data.forecast_features() (no live
session -- grid/quali_position genuinely unknown yet) and have the race
model sample its grid from the quali model's own simulated distribution,
same as the dashboard's Forecast/Post-practice stages. "post_quali" uses
app.data.race_prediction_features() (the real grid, from the just-
finished Qualifying session) and skips quali logging entirely -- qualifying
has already happened, there's nothing left to predict there.

Only the quali model logs at forecast/post_practice. The race v2 model
logs at all three stages. v1 keeps running completely unchanged via
predict_next_race.py / predictions/{year}.csv -- this script never reads
or writes that file.

Auto-detect (no --year/--gp): "forecast" = the next event on the calendar
after now, in this year or next, regardless of session timing (a Monday
pre-weekend forecast for whatever's coming up). "post_practice" and
"post_quali" run on an hourly Sat/Sun cron (see update-data.yml) rather
than one fixed time, and each tick re-checks the schedule itself: an
event qualifies once its relevant session (Practice 2/3 for
post_practice, Qualifying for post_quali) ended at least 30/90 minutes
ago, its Race hasn't started yet, and this stage hasn't already been
logged for it -- same gate predict_next_race.py uses for v1, and for the
same reason (a fixed single-time cron broke for the 2026 US GP's
COTA-timed Qualifying). Each exits 0 with a message (not a failure) if
nothing currently matches -- a bye week, an early tick, or an
already-logged race is not an error. Never backfills a prediction after
the race has started; a stage that was missed that way gets logged
honestly via app.predictions_log.log_missed_prediction() instead.

Uses ONLY the frozen models (load_frozen_model() in both
app.models.quali_predictor and app.models.race_predictor_v2) -- never a
freshly self-trained one -- same discipline as predict_next_race.py.
Exits non-zero if either hasn't been frozen yet.
"""
from __future__ import annotations

import argparse

import fastf1
import pandas as pd

from app.config import PREDICTIONS_DIR
from app.data import forecast_features, race_has_started, race_prediction_features, session_readiness
from app.models.quali_predictor import frozen_model_spec as quali_frozen_spec
from app.models.quali_predictor import load_frozen_model as load_frozen_quali
from app.models.quali_predictor import predict_quali
from app.models.race_predictor_v2 import frozen_model_spec as race_v2_frozen_spec
from app.models.race_predictor_v2 import load_frozen_model as load_frozen_race_v2
from app.models.race_predictor_v2 import predict_race_v2
from app.predictions_log import log_missed_prediction, prediction_already_logged

_STAGES = ("forecast", "post_practice", "post_quali")

# (session names whose end gates this stage, buffer after that end, and the
# models that log a missed-prediction entry if this stage's window closes
# -- the race starts -- without ever having been run for a given race).
_STAGE_GATE = {
    "post_practice": (("Practice 2", "Practice 3"), pd.Timedelta(minutes=30), ("quali", "race_v2")),
    "post_quali": (("Qualifying",), pd.Timedelta(minutes=90), ("race_v2",)),
}


def _auto_detect_forecast(now: pd.Timestamp) -> tuple[int, str, str | None] | None:
    """The next event on the calendar after now -- "the race coming up",
    regardless of session timing."""
    for year in (now.year, now.year + 1):
        try:
            sch = fastf1.get_event_schedule(year, include_testing=False)
        except Exception as e:  # noqa: BLE001
            if year == now.year:
                raise SystemExit(f"Could not fetch the {year} schedule to auto-detect the next race: {e}") from e
            continue  # next year's calendar isn't published yet -- not an error
        dates = pd.to_datetime(sch["EventDate"])
        dates = dates.dt.tz_localize("UTC") if dates.dt.tz is None else dates
        upcoming = sch[dates >= now.normalize()]
        if not upcoming.empty:
            row = upcoming.iloc[0]
            return year, row["EventName"], None
    return None


def _gate_session_start(row: pd.Series, session_names: tuple[str, ...]) -> pd.Timestamp | None:
    """This schedule row's start time (UTC) for the latest-dated session
    among session_names, or None if it has none of them listed."""
    starts = []
    for i in range(1, 6):
        if row.get(f"Session{i}") not in session_names:
            continue
        d = row.get(f"Session{i}DateUtc")
        if pd.isna(d):
            continue
        d = pd.Timestamp(d)
        starts.append(d.tz_localize("UTC") if d.tzinfo is None else d.tz_convert("UTC"))
    return max(starts) if starts else None


def _auto_detect_staged_race(now: pd.Timestamp, stage: str, since: str) -> tuple[int, str, str | None] | None:
    """An event ready for `stage` right now: its gating session (see
    _STAGE_GATE) ended at least its buffer ago, its Race hasn't started,
    and this stage hasn't already been logged for it. None if nothing
    currently matches. If the gating session has ended and the Race has
    already started without this stage ever being logged, that's a
    pipeline failure -- recorded via log_missed_prediction() rather than
    silently skipped or backfilled now that it's too late to be a real
    forecast.

    `since` is the later of the two frozen models' own frozen_at -- races
    whose gating session happened before either model existed are skipped
    entirely (not considered, not logged as "missed"), same reasoning as
    predict_next_race.py's own `since` guard."""
    session_names, buffer, models = _STAGE_GATE[stage]
    try:
        sch = fastf1.get_event_schedule(now.year, include_testing=False)
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"Could not fetch the {now.year} schedule to auto-detect the race: {e}") from e

    since_ts = pd.Timestamp(since)
    since_ts = since_ts.tz_localize("UTC") if since_ts.tzinfo is None else since_ts.tz_convert("UTC")
    # race_v2 logs at every stage, so its file is the canonical "did this
    # stage already run for this race" signal even though post_practice
    # also logs a second (quali) file in the same run.
    dedup_path = PREDICTIONS_DIR / f"{now.year}_race_v2_{stage}.csv"
    for _, row in sch.iterrows():
        gp = row["EventName"]
        ready, _ = session_readiness(now, now.year, gp, session_names, buffer)
        if not ready:
            continue
        gate_start = _gate_session_start(row, session_names)
        if gate_start is None or gate_start < since_ts:
            continue
        already_logged = prediction_already_logged(dedup_path, gp)
        if race_has_started(now, now.year, gp):
            if not already_logged:
                for model in models:
                    log_missed_prediction(
                        now.year, gp, model, stage,
                        f"{'/'.join(session_names)} ended and the race started before a {stage} "
                        "prediction was ever logged (pipeline failure) -- not backfilled")
            continue
        if already_logged:
            continue
        return now.year, gp, None
    return None


def _log(path, out: pd.DataFrame) -> None:
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    if path.exists():
        out = pd.concat([pd.read_csv(path), out], ignore_index=True)
    out.to_csv(path, index=False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, choices=_STAGES)
    p.add_argument("--year", type=int, default=None)
    p.add_argument("--gp", default=None)
    p.add_argument("--circuit-id", default=None)
    args = p.parse_args()

    try:
        quali_pipe, quali_meta = load_frozen_quali()
        race_delta_pipe, race_dnf_pipe, race_meta = load_frozen_race_v2()
    except FileNotFoundError as e:
        raise SystemExit(str(e)) from e

    now = pd.Timestamp.now(tz="UTC")
    if args.year and args.gp:
        year, gp, circuit_id = args.year, args.gp, args.circuit_id
    else:
        if args.stage == "forecast":
            detected = _auto_detect_forecast(now)
        else:
            since = max(quali_frozen_spec()["frozen_at"], race_v2_frozen_spec()["frozen_at"])
            detected = _auto_detect_staged_race(now, args.stage, since=since)
        if detected is None:
            print(f"Nothing ready for the {args.stage} stage right now -- nothing to do.")
            return
        year, gp, circuit_id = detected

    if args.stage == "post_quali":
        race_feats = race_prediction_features(year, gp, circuit_id)
    else:
        race_feats = forecast_features(year, circuit_id)

    predicted_at = now.isoformat()

    if args.stage != "post_quali":
        quali_preds = predict_quali(quali_pipe, race_feats, quali_meta)
        q_out = quali_preds[["driver", "team_id", "predicted_quali_position", "pole_probability",
                             "top3_probability", "q3_probability", "expected_quali_position",
                             "driver_rolling_quali_position_5", "driver_circuit_last_quali"]].rename(
            columns={"driver_rolling_quali_position_5": "baseline_rolling_quali_pos",
                    "driver_circuit_last_quali": "baseline_last_year_quali_pos"}).copy()
        q_out.insert(0, "predicted_at", predicted_at)
        q_out.insert(1, "gp", gp)
        q_out["actual_quali_position"] = pd.NA
        q_out["scored_at"] = pd.NA
        _log(PREDICTIONS_DIR / f"{year}_quali_{args.stage}.csv", q_out)
        print(f"Logged {len(q_out)} quali predictions for {year} {gp} ({args.stage})")

        race_preds = predict_race_v2(race_delta_pipe, race_dnf_pipe, race_feats, race_meta,
                                     quali_pipe=quali_pipe, quali_meta=quali_meta, quali_features=race_feats)
    else:
        race_preds = predict_race_v2(race_delta_pipe, race_dnf_pipe, race_feats, race_meta)

    r_out = race_preds[["driver", "team_id", "grid", "predicted_position", "win_probability",
                        "podium_probability", "points_probability"]].copy()
    r_out.insert(0, "predicted_at", predicted_at)
    r_out.insert(1, "gp", gp)
    r_out["actual_position"] = pd.NA
    r_out["scored_at"] = pd.NA
    _log(PREDICTIONS_DIR / f"{year}_race_v2_{args.stage}.csv", r_out)
    print(f"Logged {len(r_out)} race v2 predictions for {year} {gp} ({args.stage})")


if __name__ == "__main__":
    main()
