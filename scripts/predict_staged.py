"""Log pre-race qualifying + race v2 predictions at one of three stages,
each to its own predictions/{year}_{model}_{stage}.csv so each stage's
live track record can be scored independently of the others:

    python -m scripts.predict_staged --stage forecast        # Monday cron
    python -m scripts.predict_staged --stage post_practice    # Saturday ~16:00 UTC cron
    python -m scripts.predict_staged --stage post_quali       # Saturday ~20:00 UTC cron
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
pre-weekend forecast for whatever's coming up); "post_practice" = an
event whose Practice 2 or Practice 3 happened in the last 18 hours;
"post_quali" = an event whose Qualifying happened in the last 2 days
(same window predict_next_race.py uses for v1). Each exits 0 with a
message (not a failure) if nothing matches that window -- a bye week, or
an off-cycle manual run, is not an error.

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
from app.data import forecast_features, race_prediction_features
from app.models.quali_predictor import load_frozen_model as load_frozen_quali
from app.models.quali_predictor import predict_quali
from app.models.race_predictor_v2 import load_frozen_model as load_frozen_race_v2
from app.models.race_predictor_v2 import predict_race_v2

_STAGES = ("forecast", "post_practice", "post_quali")


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


def _auto_detect_session(now: pd.Timestamp, session_names: tuple[str, ...],
                         window: pd.Timedelta) -> tuple[int, str, str | None] | None:
    """An event whose session (matching one of session_names, e.g.
    "Practice 3") completed within `window` before now."""
    try:
        sch = fastf1.get_event_schedule(now.year, include_testing=False)
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"Could not fetch the {now.year} schedule to auto-detect the race: {e}") from e

    for _, row in sch.iterrows():
        for i in range(1, 6):
            if row.get(f"Session{i}") not in session_names:
                continue
            s_date = row.get(f"Session{i}DateUtc")
            if pd.isna(s_date):
                continue
            s_date = pd.Timestamp(s_date)
            s_date = s_date.tz_localize("UTC") if s_date.tzinfo is None else s_date
            if pd.Timedelta(0) <= (now - s_date) <= window:
                return now.year, row["EventName"], None
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

    now = pd.Timestamp.now(tz="UTC")
    if args.year and args.gp:
        year, gp, circuit_id = args.year, args.gp, args.circuit_id
    else:
        if args.stage == "forecast":
            detected = _auto_detect_forecast(now)
        elif args.stage == "post_practice":
            detected = _auto_detect_session(now, ("Practice 2", "Practice 3"), pd.Timedelta(hours=18))
        else:
            detected = _auto_detect_session(now, ("Qualifying",), pd.Timedelta(days=2))
        if detected is None:
            print(f"No race matched the {args.stage} window -- nothing to log.")
            return
        year, gp, circuit_id = detected

    try:
        quali_pipe, quali_meta = load_frozen_quali()
        race_delta_pipe, race_dnf_pipe, race_meta = load_frozen_race_v2()
    except FileNotFoundError as e:
        raise SystemExit(str(e)) from e

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
