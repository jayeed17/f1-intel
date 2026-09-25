"""Session loading, lap selection and JSON serialisation helpers."""
from __future__ import annotations

import json
import tempfile
from functools import lru_cache
from pathlib import Path

import fastf1
import fastf1.mvapi as mvapi
import numpy as np
import pandas as pd

from app.config import CACHE_DIR, PROCESSED_DIR


def _writable_cache_dir(preferred: Path) -> Path:
    """Use the repo's cache/ dir if writable; fall back to a temp dir otherwise
    (e.g. a read-only deployment filesystem)."""
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        probe = preferred / ".write_test"
        probe.touch()
        probe.unlink()
        return preferred
    except OSError:
        fallback = Path(tempfile.gettempdir()) / "f1-intel-cache"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


CACHE_DIR = _writable_cache_dir(CACHE_DIR)
fastf1.Cache.enable_cache(str(CACHE_DIR))

# Mirrors fastf1.core.Laps.QUICKLAP_THRESHOLD so clean_laps' filtering logic
# can run on a plain DataFrame (e.g. loaded back from Parquet) without needing
# a real Laps object.
_QUICKLAP_THRESHOLD = 1.07


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


_CORNER_COLUMNS = ["Label", "Number", "Distance", "X", "Y"]


def corners(session) -> pd.DataFrame:
    """Corner locations for the session's circuit.

    FastF1 fetches circuit maps from the MultiViewer API by circuit_key, and
    that map may not exist yet for a freshly started season or a redesigned
    track (raises AttributeError instead of a clean None). Fall back to the
    previous year's map for the same key; if that's unavailable too, return
    an empty frame so callers can degrade instead of crashing.
    """
    try:
        info = session.get_circuit_info()
    except AttributeError:
        info = None
    if info is None:
        key = session.session_info["Meeting"]["Circuit"]["Key"]
        info = mvapi.get_circuit_info(year=session.event.year - 1, circuit_key=key)
        if info is not None:
            info.add_marker_distance(reference_lap=session.laps.pick_fastest())
    if info is None:
        return pd.DataFrame(columns=_CORNER_COLUMNS)
    c = info.corners.copy()
    c["Label"] = "T" + c["Number"].astype(str) + c["Letter"].fillna("").astype(str)
    return c[_CORNER_COLUMNS].sort_values("Distance").reset_index(drop=True)


def clean_laps(session_or_laps, with_weather: bool = False) -> pd.DataFrame:
    """Green-flag, non-pit, representative laps as a plain DataFrame (LapTimeS in seconds).

    Accepts either a live FastF1 session or a plain raw-laps DataFrame (e.g.
    from race_laps()/Parquet cache). with_weather requires a live session.
    """
    if isinstance(session_or_laps, pd.DataFrame):
        if with_weather:
            raise ValueError("with_weather requires a live FastF1 session, not cached laps")
        raw = session_or_laps
        time_threshold = raw["LapTime"].min() * _QUICKLAP_THRESHOLD
        mask = ((raw["LapTime"] < time_threshold) & raw["PitInTime"].isna()
                & raw["PitOutTime"].isna() & (raw["TrackStatus"] == "1")
                & (raw["IsAccurate"] == True))  # noqa: E712
        df = raw[mask].reset_index(drop=True)
    else:
        laps = session_or_laps.laps.pick_quicklaps().pick_wo_box().pick_track_status("1")
        laps = laps[laps["IsAccurate"] == True]  # noqa: E712
        df = pd.DataFrame(laps).reset_index(drop=True)
        if with_weather and not df.empty:
            w = laps.get_weather_data().reset_index(drop=True)
            for col in ("TrackTemp", "AirTemp", "Humidity", "Rainfall"):
                if col in w.columns:
                    df[col] = w[col].values
    df["LapTimeS"] = df["LapTime"].dt.total_seconds()
    return df.dropna(subset=["LapTimeS", "TyreLife", "Compound"]).reset_index(drop=True)


def _round_number(year: int, gp: str | int) -> int:
    gp = _gp(gp)
    return gp if isinstance(gp, int) else int(fastf1.get_event(year, gp)["RoundNumber"])


def _processed_path(year: int, round_number: int) -> Path:
    return PROCESSED_DIR / str(year) / f"{round_number}.parquet"


def race_laps(year: int, gp: str | int) -> pd.DataFrame:
    """Raw laps for a race, as a plain DataFrame. Reads data/processed/{year}/{round}.parquet
    if present; otherwise loads from FastF1 and writes it for next time."""
    path = _processed_path(year, _round_number(year, gp))
    if path.exists():
        return pd.read_parquet(path)
    session = load_session(year, gp, "R", telemetry=False)
    df = pd.DataFrame(session.laps)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df


def to_records(df: pd.DataFrame) -> list[dict]:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_timedelta64_dtype(out[c]):
            out[c] = out[c].dt.total_seconds()
    out = out.replace([np.inf, -np.inf], np.nan)
    return json.loads(out.to_json(orient="records", date_format="iso"))
