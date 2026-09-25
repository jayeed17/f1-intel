"""Per-team 'where can they improve' report for one session."""
from __future__ import annotations

import pandas as pd

from app.analysis.pits import pit_stops
from app.data import clean_laps


def _s(col: pd.Series) -> pd.Series:
    return col.dt.total_seconds() if pd.api.types.is_timedelta64_dtype(col) else col


def team_report(session) -> pd.DataFrame:
    laps = pd.DataFrame(session.laps).dropna(subset=["Team"])
    g = laps.groupby("Team")
    df = pd.DataFrame({
        "best_lap": _s(g["LapTime"].min()),
        "s1": _s(g["Sector1Time"].min()),
        "s2": _s(g["Sector2Time"].min()),
        "s3": _s(g["Sector3Time"].min()),
        "speed_trap": g["SpeedST"].max(),
    })

    clean = clean_laps(session)
    if not clean.empty:
        df["race_pace"] = clean.groupby("Team")["LapTimeS"].median()
    stops = pit_stops(laps)
    if not stops.empty:
        df["pit_lane"] = stops.groupby("Team")["PitLaneS"].median()

    for c in ["best_lap", "s1", "s2", "s3", "race_pace", "pit_lane"]:
        if c in df:
            df[f"{c}_gap"] = (df[c] - df[c].min()).round(3)
    df["speed_trap_deficit"] = (df["speed_trap"].max() - df["speed_trap"]).round(1)

    notes = []
    for _, r in df.iterrows():
        n = []
        sec_gaps = {k: r[f"{k}_gap"] for k in ("s1", "s2", "s3") if pd.notna(r.get(f"{k}_gap"))}
        if sec_gaps:
            worst = max(sec_gaps, key=sec_gaps.get)
            if sec_gaps[worst] > 0.05:
                n.append(f"weakest sector {worst.upper()} (+{sec_gaps[worst]:.3f}s vs best)")
        if r.get("speed_trap_deficit", 0) > 5:
            n.append(f"top speed -{r['speed_trap_deficit']:.0f} kph (drag/power)")
        if pd.notna(r.get("pit_lane_gap")) and r["pit_lane_gap"] > 1.0:
            n.append(f"pit lane +{r['pit_lane_gap']:.1f}s vs best team")
        if pd.notna(r.get("race_pace_gap")) and r["race_pace_gap"] > 0.5:
            n.append(f"race pace +{r['race_pace_gap']:.2f}s/lap (tyre mgmt / balance)")
        notes.append("; ".join(n) or "no standout weakness")
    df["improve"] = notes

    sort_col = "race_pace_gap" if "race_pace_gap" in df else "best_lap_gap"
    return df.sort_values(sort_col).reset_index()
