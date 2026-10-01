"""Build the race-outcome-predictor training dataset: results + qualifying
classifications only, no telemetry. Run locally:

    python -m scripts.build_race_dataset
    python -m scripts.build_race_dataset --seasons 2023 2024 2025 2026

Sources, per the roadmap item this implements:
- 2025-2026: data/prebuilt/ (already-built Race + Qualifying sessions, no
  network needed). The team rolling race-pace-gap feature only exists for
  these seasons -- it needs lap times, which 2022-2024 doesn't have here.
- 2022-2024: FastF1's Ergast/Jolpica client (fastf1.ergast.Ergast) --
  results + qualifying classifications only, no telemetry, no FastF1 session
  loads.

One row per driver per race.

DNF / DSQ / pit-lane start / grid penalty handling:
- Finish position is each source's own official classification (`Position`
  from FastF1 results, `position` from Ergast) used as-is. Both already
  reflect DSQ (demoted below every classified finisher) and retirees who
  covered enough of the race to be classified, since they ultimately trace
  back to the same FIA timing data.
- "DNF" (for the rolling driver_dnf_rate_10/team_dnf_rate_10 features, AND as
  the race outcome predictor's DNF classifier target -- see
  app/models/race_predictor.py) is derived uniformly from lap count:
  completed < 90% of the race's max lap count that race. This is a proxy for
  the official classification threshold, not the official ruling, but it's
  source-agnostic and consistent across 2022-2026. target_delta (finish -
  grid, the predictor's regression target) is only meaningful for dnf == 0
  rows -- a retiree's "positions gained" isn't the quantity of interest.
- Pit-lane starts report grid=0 in both sources. Replaced with
  (field_size + 1) -- numerically worse than every real grid slot -- plus a
  `grid_pit_lane` boolean flag column so the model can also learn it's not
  "just another back-of-grid start".
- Grid penalties are not specially handled: they're exactly why both `grid`
  (post-penalty, actual starting slot) and `quali_position` (pre-penalty
  qualifying classification) are kept as separate features. Their gap is
  the penalty's effect.
"""
from __future__ import annotations

import argparse
import time

import fastf1.ergast as ergast
import numpy as np
import pandas as pd
from fastf1.exceptions import ErgastInvalidRequestError, RateLimitExceededError

from app.config import (CIRCUIT_HISTORY_RESET, CIRCUIT_TYPE, MODEL_DATA_DIR, RACE_DATASET_PATH,
                        REG_CHANGE_ROUNDS, REG_CHANGE_SEASONS, TEAM_ID)
from app.data import PrebuiltSession, _resolve_prebuilt, clean_laps, prebuilt_races

DATASET_PATH = RACE_DATASET_PATH  # kept as a local alias; this is the name scripts/tests already use
_ERGAST = ergast.Ergast()

# jolpica's own server enforces a much tighter rate limit than FastF1's
# client-side 200/h counter ever sees -- it 429s well before that, which
# FastF1 surfaces as ErgastInvalidRequestError (a generic "bad request"
# class), not RateLimitExceededError. Confirmed live: a tight loop of
# get_race_results()/get_qualifying_results() calls starts 429ing after
# ~10-12 requests. Treat a 429-flavoured ErgastInvalidRequestError as a
# retry-worthy rate limit, and pace every call so we hit it less often.
_RATE_LIMIT_BACKOFFS = [30, 60, 120, 300, 600]
_CALL_DELAY_S = 1.5


def _is_rate_limited(e: Exception) -> bool:
    return isinstance(e, RateLimitExceededError) or (
        isinstance(e, ErgastInvalidRequestError) and "Too Many Requests" in str(e))


def _ergast_call(fn, *args, **kwargs):
    for attempt, wait in enumerate([0, *_RATE_LIMIT_BACKOFFS]):
        if wait:
            print(f"  ! ergast rate limited -- backing off {wait}s "
                 f"(retry {attempt}/{len(_RATE_LIMIT_BACKOFFS)})")
            time.sleep(wait)
        try:
            result = fn(*args, **kwargs)
            time.sleep(_CALL_DELAY_S)
            return result
        except Exception as e:  # noqa: BLE001
            if not _is_rate_limited(e):
                raise
    raise RateLimitExceededError("gave up after repeated rate limiting")

FINAL_COLUMNS = [
    "season", "round", "event_name", "circuit_id", "driver", "team_id",
    "grid", "grid_pit_lane", "quali_position", "quali_gap_to_pole_s", "teammate_quali_gap_s",
    "driver_rolling_avg_finish_3", "driver_rolling_avg_finish_5",
    "team_rolling_avg_finish_3", "team_rolling_pace_gap_3",
    "driver_dnf_rate_10", "team_dnf_rate_10",
    "circuit_type", "reg_change_flag", "dnf", "source",
    "target_finish_pos", "target_points_top10", "target_delta",
    # circuit history (last 3 prior editions of this exact circuit identity)
    "circuit_new_or_changed",
    "driver_circuit_avg_quali_3", "driver_circuit_avg_finish_3", "driver_circuit_last_quali",
    "driver_circuit_races_here", "driver_circuit_hist_pre_reg_change",
    "team_circuit_avg_quali_3", "team_circuit_avg_finish_3", "team_circuit_last_quali",
    "team_circuit_races_here", "team_circuit_hist_pre_reg_change",
    # rolling qualifying form
    "driver_rolling_quali_position_3", "driver_rolling_quali_position_5",
    "team_rolling_quali_position_3", "teammate_quali_gap_trend_3",
    # practice pace (FP2+FP3, prebuilt-laps-only, NaN wherever not prebuilt)
    "fp_best_gap_s", "fp_long_run_gap_s",
]


def _team_id(raw: str) -> str:
    return TEAM_ID.get(raw, raw)


# --------------------------------------------------------------------------
# Per-race extraction: normalise both sources into the same shape.
# --------------------------------------------------------------------------

def _quali_from_prebuilt(year: int, round_number: int) -> pd.DataFrame | None:
    resolved = _resolve_prebuilt(year, round_number, "Q")
    if resolved is None:
        return None
    s = PrebuiltSession(*resolved, "Q")
    if s.laps.empty:
        return None
    best = s.laps.groupby("Driver")["LapTime"].min().dt.total_seconds()
    rows = [{
        "driver": r["Abbreviation"], "team_id": _team_id(r["TeamName"]),
        "quali_position": r["Position"], "quali_best_s": best.get(r["Abbreviation"], np.nan),
    } for _, r in s.results.iterrows()]
    return pd.DataFrame(rows)


def _quali_from_ergast(year: int, round_number: int) -> pd.DataFrame | None:
    try:
        qr = _ergast_call(_ERGAST.get_qualifying_results, season=year, round=round_number)
    except Exception:  # noqa: BLE001 -- e.g. not published yet, network hiccup
        return None
    if not qr.content or qr.content[0].empty:
        return None

    def _best(row) -> float:
        times = [row.get(c) for c in ("Q1", "Q2", "Q3")]
        times = [t.total_seconds() for t in times if pd.notna(t)]
        return min(times) if times else np.nan

    df = qr.content[0]
    rows = [{
        "driver": r["driverCode"], "team_id": _team_id(r["constructorId"]),
        "quali_position": r["position"], "quali_best_s": _best(r),
    } for _, r in df.iterrows()]
    return pd.DataFrame(rows)


def _race_from_prebuilt(year: int, round_number: int) -> pd.DataFrame | None:
    resolved = _resolve_prebuilt(year, round_number, "R")
    if resolved is None:
        return None
    s = PrebuiltSession(*resolved, "R")
    if s.laps.empty:
        return None
    laps_done = s.laps.groupby("Driver")["LapNumber"].max()
    max_laps = float(laps_done.max())
    rows = [{
        "driver": r["Abbreviation"], "team_id": _team_id(r["TeamName"]),
        "grid": r["GridPosition"], "finish_position": r["Position"],
        "laps_completed": laps_done.get(r["Abbreviation"], np.nan), "race_max_laps": max_laps,
    } for _, r in s.results.iterrows()]
    return pd.DataFrame(rows)


def _race_from_ergast(year: int, round_number: int) -> pd.DataFrame | None:
    try:
        rr = _ergast_call(_ERGAST.get_race_results, season=year, round=round_number)
    except Exception:  # noqa: BLE001
        return None
    if not rr.content or rr.content[0].empty:
        return None
    df = rr.content[0]
    max_laps = float(df["laps"].max())
    rows = [{
        "driver": r["driverCode"], "team_id": _team_id(r["constructorId"]),
        "grid": r["grid"], "finish_position": r["position"],
        "laps_completed": r["laps"], "race_max_laps": max_laps,
    } for _, r in df.iterrows()]
    return pd.DataFrame(rows)


def _add_within_race_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pole gap, teammate gap, and the pit-lane-start fix -- everything that
    needs to compare drivers against others in the *same* race."""
    df = df.copy()
    pole = df["quali_best_s"].min()
    df["quali_gap_to_pole_s"] = df["quali_best_s"] - pole

    gap = {}
    for _, g in df.groupby("team_id"):
        if len(g) != 2:  # mid-season driver change, or a data gap -- can't pair reliably
            continue
        d1, d2 = g.iloc[0], g.iloc[1]
        gap[d1["driver"]] = d1["quali_best_s"] - d2["quali_best_s"]
        gap[d2["driver"]] = d2["quali_best_s"] - d1["quali_best_s"]
    df["teammate_quali_gap_s"] = df["driver"].map(gap)

    # Pit-lane start = back of grid, i.e. the number of cars that actually
    # started this race (== len(df) here, one row per starter). NOT
    # max(everyone else's raw grid number) + 1: that number can have gaps
    # (grid numbers aren't renumbered when someone else also starts from the
    # pit lane) and so can overshoot the real field size -- confirmed on a
    # real race (2022 round 5, 20 starters incl. 2 pit-lane: the old logic
    # produced grid=21 in a 20-car field).
    df["grid_pit_lane"] = df["grid"] == 0
    df["grid"] = df["grid"].where(~df["grid_pit_lane"], len(df))

    df["dnf"] = (df["laps_completed"] < 0.9 * df["race_max_laps"]).astype(float)
    return df


def build_race(year: int, round_number: int, event_name: str, circuit_id: str | None,
               source: str) -> pd.DataFrame | None:
    if source == "prebuilt":
        quali, race = _quali_from_prebuilt(year, round_number), _race_from_prebuilt(year, round_number)
    else:
        quali, race = _quali_from_ergast(year, round_number), _race_from_ergast(year, round_number)
    if race is None or race.empty:
        return None
    if quali is None:
        quali = pd.DataFrame(columns=["driver", "quali_position", "quali_best_s"])

    merged = race.merge(quali[["driver", "quali_position", "quali_best_s"]], on="driver", how="left")
    merged = _add_within_race_features(merged)
    merged["season"] = year
    merged["round"] = round_number
    merged["event_name"] = event_name
    merged["circuit_id"] = circuit_id
    merged["source"] = source
    return merged


# --------------------------------------------------------------------------
# Circuit-id crosswalk (for circuit_type only -- results/quali never use it).
# --------------------------------------------------------------------------

def _circuit_crosswalk(seasons: list[int]) -> dict[tuple[int, int], tuple[str, str]]:
    """(season, round) -> (event_name, circuitId), from Ergast's schedule.
    Only used to look up circuit_type; a season/round missing here (e.g. the
    still-provisional far end of the current season in Jolpica's database)
    just falls back to "mixed" rather than failing the whole build."""
    crosswalk = {}
    for year in seasons:
        try:
            sch = _ergast_call(_ERGAST.get_race_schedule, season=year)
        except Exception as e:  # noqa: BLE001
            print(f"! could not fetch the {year} schedule for circuit lookup: {e}")
            continue
        for _, row in sch.iterrows():
            crosswalk[(year, int(row["round"]))] = (row["raceName"], row["circuitId"])
    return crosswalk


# --------------------------------------------------------------------------
# Team race-pace-gap feature (2025-2026 only, from prebuilt lap times).
# --------------------------------------------------------------------------

def _team_pace_gap_table(seasons: tuple[int, ...] = (2025, 2026)) -> pd.DataFrame:
    """(season, round, team_id) -> this race's median green-flag race pace gap
    to the fastest team that race. Only exists where prebuilt has lap data;
    left NaN everywhere else by the caller's merge."""
    rows = []
    for r in prebuilt_races():
        if r["year"] not in seasons or "R" not in r["sessions"]:
            continue
        resolved = _resolve_prebuilt(r["year"], r["round"], "R")
        if resolved is None:
            continue
        s = PrebuiltSession(*resolved, "R")
        cl = clean_laps(s)
        if cl.empty:
            continue
        cl = cl.copy()
        cl["team_id"] = cl["Team"].map(_team_id)
        pace = cl.groupby("team_id")["LapTimeS"].median()
        if pace.empty:
            continue
        fastest = pace.min()
        for team_id, med in pace.items():
            rows.append({"season": r["year"], "round": r["round"], "team_id": team_id,
                        "team_pace_gap_s": med - fastest})
    return pd.DataFrame(rows)


def _rolling(s: pd.Series, window: int) -> pd.Series:
    """Shift(1) BEFORE rolling: race N's window is strictly races before N,
    never race N itself. This is the leakage guard for every rolling feature."""
    return s.shift(1).rolling(window, min_periods=1).mean()


def add_delta_and_team_dnf_features(df: pd.DataFrame) -> pd.DataFrame:
    """target_delta (positions gained: finish - grid, the race-outcome-predictor
    rework's regression target for classified finishers) and team_dnf_rate_10
    (team-level rolling DNF rate, the DNF classifier's team-level feature --
    mirrors driver_dnf_rate_10 but rolled over the team's last 10 team-races).
    Pulled out as its own function (not inlined in build_dataset) so the same
    logic can be applied to an already-built dataset.parquet without a full
    network rebuild -- see scripts/patch_race_dataset.py."""
    df = df.copy()
    df["target_delta"] = df["target_finish_pos"] - df["grid"]
    team_race = df.groupby(["season", "round", "team_id"], as_index=False).agg(team_dnf=("dnf", "mean"))
    team_race = team_race.sort_values(["team_id", "season", "round"])
    team_race["team_dnf_rate_10"] = team_race.groupby("team_id")["team_dnf"].transform(
        lambda s: _rolling(s, 10))
    return df.merge(team_race[["season", "round", "team_id", "team_dnf_rate_10"]],
                    on=["season", "round", "team_id"], how="left")


def _reg_era(season: int) -> int:
    """Which regulation era `season` belongs to -- the latest reg-change
    season at or before it, or 0 if there isn't one yet. Two seasons in the
    same era are "the same rules generation" for this feature's purposes."""
    era = 0
    for e in sorted(REG_CHANGE_SEASONS):
        if season >= e:
            era = e
    return era


def _history_key(circuit_id: str | None, season: int) -> str | None:
    """circuit_id, unless CIRCUIT_HISTORY_RESET says this circuit got a
    materially different layout starting some season -- then editions at or
    after that season use a distinct key, so they never pick up history
    from the old layout (and the old layout's rows never see the new one)."""
    if circuit_id is None:
        return None
    reset_season = CIRCUIT_HISTORY_RESET.get(circuit_id)
    if reset_season is not None and season >= reset_season:
        return f"{circuit_id}__since{reset_season}"
    return circuit_id


def add_circuit_history_and_quali_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Circuit history (driver/team avg quali position + avg finish at this
    exact circuit identity, over up to the last 3 PRIOR editions -- strictly
    earlier seasons, never this one) plus rolling qualifying-form features.
    Pulled out as its own function (same reason as
    add_delta_and_team_dnf_features): re-appliable to an already-built
    dataset.parquet without a full network rebuild.

    Circuit "identity" is circuit_id, except where CIRCUIT_HISTORY_RESET
    marks a layout change (_history_key) -- the reset season, like a
    genuinely brand-new venue, gets circuit_new_or_changed=True and NaN
    history (no borrowing from the old layout's rows, or from any other
    venue). circuit_new_or_changed is a property of the event itself (every
    driver/team there faces the same blank slate), not of any one driver's
    experience.
    """
    df = df.copy()
    df["_hist_key"] = [_history_key(c, s) for c, s in zip(df["circuit_id"], df["season"])]

    first_season = df.groupby("_hist_key")["season"].transform("min")
    df["circuit_new_or_changed"] = df["season"] == first_season

    def _circuit_history(entity_col: str, prefix: str, dedupe: bool = False) -> pd.DataFrame:
        cols = [entity_col, "_hist_key", "season", "round", "quali_position", "target_finish_pos"]
        d = df[cols].copy()
        if dedupe:
            # team_id has 2 rows per race (one per teammate) in df -- collapse
            # to one row per (team, circuit identity, race) BEFORE rolling, or
            # the later merge back onto df fans out 2x2 (confirmed live: every
            # row in the dataset silently doubled without this).
            d = d.groupby([entity_col, "_hist_key", "season", "round"], as_index=False).agg(
                quali_position=("quali_position", "mean"), target_finish_pos=("target_finish_pos", "mean"))
        d = d.sort_values([entity_col, "_hist_key", "season"])
        grp = d.groupby([entity_col, "_hist_key"])
        d[f"{prefix}_circuit_avg_quali_3"] = grp["quali_position"].transform(
            lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        d[f"{prefix}_circuit_avg_finish_3"] = grp["target_finish_pos"].transform(
            lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        d[f"{prefix}_circuit_last_quali"] = grp["quali_position"].transform(lambda s: s.shift(1))
        d[f"{prefix}_circuit_races_here"] = grp.cumcount()  # count of PRIOR editions, 0 for the first
        prior_min_season = grp["season"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).min())
        d[f"{prefix}_circuit_hist_pre_reg_change"] = [
            bool(pd.notna(pmin) and _reg_era(int(pmin)) < _reg_era(int(cur)))
            for pmin, cur in zip(prior_min_season, d["season"])
        ]
        return d[[entity_col, "_hist_key", "season", "round",
                 f"{prefix}_circuit_avg_quali_3", f"{prefix}_circuit_avg_finish_3",
                 f"{prefix}_circuit_last_quali",
                 f"{prefix}_circuit_races_here", f"{prefix}_circuit_hist_pre_reg_change"]]

    df = df.merge(_circuit_history("driver", "driver"), on=["driver", "_hist_key", "season", "round"], how="left")
    df = df.merge(_circuit_history("team_id", "team", dedupe=True),
                 on=["team_id", "_hist_key", "season", "round"], how="left")
    df = df.drop(columns=["_hist_key"])

    # --- rolling qualifying form (mirrors the finish-position rolling block) ---
    df = df.sort_values(["driver", "season", "round"])
    g = df.groupby("driver")
    df["driver_rolling_quali_position_3"] = g["quali_position"].transform(lambda s: _rolling(s, 3))
    df["driver_rolling_quali_position_5"] = g["quali_position"].transform(lambda s: _rolling(s, 5))
    df["teammate_quali_gap_trend_3"] = g["teammate_quali_gap_s"].transform(lambda s: _rolling(s, 3))

    team_quali = df.groupby(["season", "round", "team_id"], as_index=False).agg(
        team_quali=("quali_position", "mean"))
    team_quali = team_quali.sort_values(["team_id", "season", "round"])
    team_quali["team_rolling_quali_position_3"] = team_quali.groupby("team_id")["team_quali"].transform(
        lambda s: _rolling(s, 3))
    df = df.merge(team_quali[["season", "round", "team_id", "team_rolling_quali_position_3"]],
                 on=["season", "round", "team_id"], how="left")
    return df


# --------------------------------------------------------------------------
# Practice pace (FP2+FP3, prebuilt laps-only -- see build_prebuilt.py's
# LAPS_ONLY_SESSIONS). NaN wherever FP2/FP3 aren't prebuilt yet, or there
# aren't enough laps to form a long-run estimate -- "when available" by
# design, same pattern as team_rolling_pace_gap_3.
# --------------------------------------------------------------------------

def _fp_long_run_median(lap_times: pd.Series, min_laps: int = 6, drop_fastest: int = 3) -> float:
    """A cheap long-run-pace proxy without full stint detection: drop each
    driver's `drop_fastest` quickest laps (push/quali-sim laps, which don't
    represent fuel-corrected race pace) and take the median of the rest.
    NaN if there aren't enough laps to do that meaningfully."""
    s = lap_times.sort_values()
    if len(s) < min_laps:
        return np.nan
    rest = s.iloc[drop_fastest:]
    return float(rest.median()) if len(rest) >= 3 else np.nan


def _fp_pace_table(seasons: tuple[int, ...] = (2025, 2026)) -> pd.DataFrame:
    """(season, round, driver) -> fp_best_gap_s (best single lap across
    FP2+FP3, gap to the session's fastest driver) and fp_long_run_gap_s
    (gap between each driver's _fp_long_run_median and the fastest one)."""
    rows = []
    for r in prebuilt_races():
        if r["year"] not in seasons:
            continue
        frames = []
        for code in ("FP2", "FP3"):
            if code not in r["sessions"]:
                continue
            resolved = _resolve_prebuilt(r["year"], r["round"], code)
            if resolved is None:
                continue
            s = PrebuiltSession(*resolved, code)
            if s.laps.empty:
                continue
            laps = s.laps[s.laps["PitInTime"].isna() & s.laps["PitOutTime"].isna()].copy()
            if "TrackStatus" in laps.columns:
                laps = laps[laps["TrackStatus"] == "1"]
            frames.append(laps)
        if not frames:
            continue
        laps = pd.concat(frames, ignore_index=True).dropna(subset=["LapTime", "Driver"])
        if laps.empty:
            continue
        laps["LapTimeS"] = laps["LapTime"].dt.total_seconds()

        best = laps.groupby("Driver")["LapTimeS"].min()
        if best.empty:
            continue
        best_gap = best - best.min()

        long_run = laps.groupby("Driver")["LapTimeS"].apply(_fp_long_run_median)
        valid = long_run.dropna()
        long_run_gap = long_run - valid.min() if not valid.empty else long_run

        for drv in best.index:
            rows.append({"season": r["year"], "round": r["round"], "driver": drv,
                        "fp_best_gap_s": float(best_gap.get(drv, np.nan)),
                        "fp_long_run_gap_s": float(long_run_gap.get(drv, np.nan))})
    return pd.DataFrame(rows, columns=["season", "round", "driver", "fp_best_gap_s", "fp_long_run_gap_s"])


def collect_all(seasons_prebuilt: tuple[int, ...] = (2025, 2026),
                seasons_ergast: tuple[int, ...] = (2022, 2023, 2024)) -> pd.DataFrame:
    crosswalk = _circuit_crosswalk(list(seasons_prebuilt) + list(seasons_ergast))
    frames = []

    for r in prebuilt_races():
        if r["year"] not in seasons_prebuilt or "R" not in r["sessions"]:
            continue
        name, circuit_id = crosswalk.get((r["year"], r["round"]), (r["name"], None))
        row = build_race(r["year"], r["round"], name, circuit_id, "prebuilt")
        if row is not None:
            frames.append(row)
            print(f"  {r['year']} round {r['round']} {r['name']}: {len(row)} drivers [prebuilt]")
        else:
            print(f"  ! {r['year']} round {r['round']} {r['name']}: no data, skipped [prebuilt]")

    for year in seasons_ergast:
        try:
            sch = _ergast_call(_ERGAST.get_race_schedule, season=year)
        except Exception as e:  # noqa: BLE001
            print(f"! could not fetch the {year} schedule, skipping this year: {e}")
            continue
        for _, ev in sch.iterrows():
            rnd = int(ev["round"])
            row = build_race(year, rnd, ev["raceName"], ev["circuitId"], "ergast")
            if row is not None:
                frames.append(row)
                print(f"  {year} round {rnd} {ev['raceName']}: {len(row)} drivers [ergast]")
            else:
                print(f"  ! {year} round {rnd} {ev['raceName']}: no data, skipped [ergast]")

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_dataset(seasons_prebuilt: tuple[int, ...] = (2025, 2026),
                  seasons_ergast: tuple[int, ...] = (2022, 2023, 2024)) -> pd.DataFrame:
    df = collect_all(seasons_prebuilt, seasons_ergast)
    if df.empty:
        raise SystemExit("No race data collected")

    # --- team race-pace-gap (2025-2026 only; NaN everywhere else via the merge) ---
    pace = _team_pace_gap_table(seasons_prebuilt)
    df = df.merge(pace, on=["season", "round", "team_id"], how="left") if not pace.empty \
        else df.assign(team_pace_gap_s=np.nan)

    # --- driver rolling features ---
    df = df.sort_values(["driver", "season", "round"]).reset_index(drop=True)
    g = df.groupby("driver")
    df["driver_rolling_avg_finish_3"] = g["finish_position"].transform(lambda s: _rolling(s, 3))
    df["driver_rolling_avg_finish_5"] = g["finish_position"].transform(lambda s: _rolling(s, 5))
    df["driver_dnf_rate_10"] = g["dnf"].transform(lambda s: _rolling(s, 10))

    # --- team rolling features (one row per team per race first, then roll, then join back) ---
    team_race = df.groupby(["season", "round", "team_id"], as_index=False).agg(
        team_finish=("finish_position", "mean"), team_pace_gap_s=("team_pace_gap_s", "first"))
    team_race = team_race.sort_values(["team_id", "season", "round"])
    team_race["team_rolling_avg_finish_3"] = team_race.groupby("team_id")["team_finish"].transform(
        lambda s: _rolling(s, 3))
    team_race["team_rolling_pace_gap_3"] = team_race.groupby("team_id")["team_pace_gap_s"].transform(
        lambda s: _rolling(s, 3))
    df = df.merge(
        team_race[["season", "round", "team_id", "team_rolling_avg_finish_3", "team_rolling_pace_gap_3"]],
        on=["season", "round", "team_id"], how="left")

    # --- circuit type + regulation-reset flag ---
    df["circuit_type"] = df["circuit_id"].map(CIRCUIT_TYPE)
    unmapped = sorted(df.loc[df["circuit_type"].isna(), "circuit_id"].dropna().unique())
    if unmapped:
        print(f"! circuit_id(s) with no CIRCUIT_TYPE entry, defaulting to 'mixed': {unmapped}")
    df["circuit_type"] = df["circuit_type"].fillna("mixed")
    df["reg_change_flag"] = df["season"].isin(REG_CHANGE_SEASONS) & (df["round"] <= REG_CHANGE_ROUNDS)

    # --- targets ---
    df["target_finish_pos"] = df["finish_position"]
    df["target_points_top10"] = df["finish_position"] <= 10
    before = len(df)
    df = df.dropna(subset=["target_finish_pos"]).reset_index(drop=True)
    if before != len(df):
        print(f"! dropped {before - len(df)} rows with no classified finish position (DNS with no data)")

    df = add_delta_and_team_dnf_features(df)
    df = add_circuit_history_and_quali_rolling_features(df)

    fp_pace = _fp_pace_table(seasons_prebuilt)
    df = df.merge(fp_pace, on=["season", "round", "driver"], how="left") if not fp_pace.empty \
        else df.assign(fp_best_gap_s=np.nan, fp_long_run_gap_s=np.nan)

    return df.sort_values(["season", "round", "target_finish_pos"])[FINAL_COLUMNS].reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seasons", type=int, nargs="+", default=[2022, 2023, 2024, 2025, 2026])
    args = p.parse_args()
    seasons_prebuilt = tuple(y for y in args.seasons if y >= 2025)
    seasons_ergast = tuple(y for y in args.seasons if y < 2025)

    df = build_dataset(seasons_prebuilt, seasons_ergast)
    MODEL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATASET_PATH)
    size_kb = DATASET_PATH.stat().st_size / 1024
    print(f"\n{DATASET_PATH}: {len(df)} rows, {df['season'].nunique()} seasons, {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
