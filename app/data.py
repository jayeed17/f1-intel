"""Session loading, lap selection and JSON serialisation helpers."""
from __future__ import annotations

import json
from functools import lru_cache

import fastf1
import numpy as np
import pandas as pd

from app.config import CACHE_DIR

CACHE_DIR.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))


class DataError(Exception):
    """Raised when requested data doesn't exist (bad driver, lap, session)."""


def _gp(gp: str | int) -> str | int:
    return int(gp) if isinstance(gp, str) and gp.isdigit() else gp


@lru_cache(maxsize=8)
def load_session(year: int, gp: str | int, session: str = "R", telemetry: bool = True):
    s = fastf1.get_session(year, _gp(gp), session)
    s.load(laps=True, telemetry=telemetry, weather=True, messages=False)
    return s


def get_lap(session, driver: str, lap: str | int = "fastest"):
    laps = session.laps.pick_drivers(driver.upper())
    if laps.empty:
        raise DataError(f"No laps for driver {driver}")
    if str(lap) == "fastest":
        out = laps.pick_fastest()
    else:
        match = laps[laps["LapNumber"] == int(lap)]
        out = match.iloc[0] if not match.empty else None
    if out is None or getattr(out, "empty", False):
        raise DataError(f"Lap {lap} not found for {driver}")
    return out


def lap_telemetry(lap) -> pd.DataFrame:
    """Car + position data for one lap with Distance (m), TimeS (s), Brake (bool)."""
    tel = lap.get_telemetry()
    if "Distance" not in tel.columns:
        tel = tel.add_distance()
    tel = pd.DataFrame(tel).copy()
    tel["TimeS"] = tel["Time"].dt.total_seconds()
    tel["Brake"] = tel["Brake"].astype(bool)
    return tel.reset_index(drop=True)


def corners(session) -> pd.DataFrame:
    c = session.get_circuit_info().corners.copy()
    c["Label"] = "T" + c["Number"].astype(str) + c["Letter"].fillna("").astype(str)
    return c[["Label", "Number", "Distance", "X", "Y"]].sort_values("Distance").reset_index(drop=True)


def clean_laps(session, with_weather: bool = False) -> pd.DataFrame:
    """Green-flag, non-pit, representative laps as a plain DataFrame (LapTimeS in seconds)."""
    laps = session.laps.pick_quicklaps().pick_wo_box().pick_track_status("1")
    laps = laps[laps["IsAccurate"] == True]  # noqa: E712
    df = pd.DataFrame(laps).reset_index(drop=True)
    if with_weather and not df.empty:
        w = laps.get_weather_data().reset_index(drop=True)
        for col in ("TrackTemp", "AirTemp", "Humidity", "Rainfall"):
            if col in w.columns:
                df[col] = w[col].values
    df["LapTimeS"] = df["LapTime"].dt.total_seconds()
    return df.dropna(subset=["LapTimeS", "TyreLife", "Compound"]).reset_index(drop=True)


def to_records(df: pd.DataFrame) -> list[dict]:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_timedelta64_dtype(out[c]):
            out[c] = out[c].dt.total_seconds()
    out = out.replace([np.inf, -np.inf], np.nan)
    return json.loads(out.to_json(orient="records", date_format="iso"))
