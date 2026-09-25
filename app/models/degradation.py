"""Fuel-corrected tyre degradation: per stint and per compound."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.config import FUEL_EFFECT_PER_LAP


def fuel_correct(df: pd.DataFrame, k: float = FUEL_EFFECT_PER_LAP) -> pd.DataFrame:
    """Add CorrS: lap time normalised to lap-1 fuel load (later laps get k*(lap-1) added back)."""
    out = df.copy()
    out["CorrS"] = out["LapTimeS"] + k * (out["LapNumber"] - 1)
    return out


def _fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    slope, intercept = np.polyfit(x, y, 1)
    pred = intercept + slope * x
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ((y - pred) ** 2).sum() / ss_tot if ss_tot > 0 else 0.0
    return float(slope), float(intercept), float(r2)


def stint_degradation(clean: pd.DataFrame, min_laps: int = 5) -> pd.DataFrame:
    """Deg rate (s/lap) for every driver stint. Needs Driver, Team, Stint, Compound, TyreLife, LapNumber, LapTimeS."""
    df = fuel_correct(clean)
    rows = []
    for (drv, stint), g in df.groupby(["Driver", "Stint"]):
        if len(g) < min_laps:
            continue
        slope, intercept, r2 = _fit(g["TyreLife"].to_numpy(float), g["CorrS"].to_numpy(float))
        rows.append({
            "Driver": drv, "Team": g["Team"].iloc[0], "Stint": int(stint),
            "Compound": g["Compound"].iloc[0], "Laps": len(g),
            "StartLap": int(g["LapNumber"].min()), "EndLap": int(g["LapNumber"].max()),
            "DegPerLap": round(slope, 4), "R2": round(r2, 3),
            "MedianLapS": round(float(g["LapTimeS"].median()), 3),
        })
    return pd.DataFrame(rows)


def compound_model(clean: pd.DataFrame, min_laps: int = 8) -> dict[str, dict[str, float]]:
    """Field-wide compound pace model: rel_time = offset + deg * tyre_age.

    rel_time is lap time relative to each driver's own median, which removes car pace.
    """
    df = fuel_correct(clean)
    df["Rel"] = df["CorrS"] - df.groupby("Driver")["CorrS"].transform("median")
    model = {}
    for comp, g in df.groupby("Compound"):
        if len(g) < min_laps or comp not in ("SOFT", "MEDIUM", "HARD"):
            continue
        slope, intercept, r2 = _fit(g["TyreLife"].to_numpy(float), g["Rel"].to_numpy(float))
        model[comp] = {"offset": round(intercept, 4), "deg": round(max(slope, 0.0), 4),
                       "r2": round(r2, 3), "n": int(len(g))}
    return model
