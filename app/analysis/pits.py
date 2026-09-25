"""Pit stop extraction and pit-loss estimation."""
from __future__ import annotations

import pandas as pd

from app.config import DEFAULT_PIT_LOSS


def pit_stops(laps: pd.DataFrame) -> pd.DataFrame:
    """One row per stop: pit lane time and total loss vs driver's median lap."""
    df = pd.DataFrame(laps).sort_values(["Driver", "LapNumber"])
    rows = []
    for drv, g in df.groupby("Driver"):
        g = g.reset_index(drop=True)
        normal = g[g["PitInTime"].isna() & g["PitOutTime"].isna()]["LapTime"].dt.total_seconds()
        med = normal.median()
        for i in range(len(g) - 1):
            if pd.notna(g.at[i, "PitInTime"]) and pd.notna(g.at[i + 1, "PitOutTime"]):
                lane = (g.at[i + 1, "PitOutTime"] - g.at[i, "PitInTime"]).total_seconds()
                in_l, out_l = g.at[i, "LapTime"], g.at[i + 1, "LapTime"]
                loss = (in_l.total_seconds() + out_l.total_seconds() - 2 * med
                        if pd.notna(in_l) and pd.notna(out_l) and pd.notna(med) else None)
                rows.append({
                    "Driver": drv, "Team": g.at[i, "Team"], "Lap": int(g.at[i, "LapNumber"]),
                    "PitLaneS": round(lane, 2), "PitLossS": round(loss, 2) if loss is not None else None,
                    "From": g.at[i, "Compound"], "To": g.at[i + 1, "Compound"],
                })
    return pd.DataFrame(rows)


def estimate_pit_loss(stops: pd.DataFrame) -> float:
    if stops.empty or "PitLossS" not in stops:
        return DEFAULT_PIT_LOSS
    s = stops["PitLossS"].dropna()
    s = s[(s > 10) & (s < 45)]  # drop SC/VSC stops and slow-stop outliers
    return round(float(s.median()), 2) if not s.empty else DEFAULT_PIT_LOSS
