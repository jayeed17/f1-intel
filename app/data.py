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
from fastf1.exceptions import DataNotLoadedError

from app.config import CACHE_DIR, DEMO_DATA_DIR, PROCESSED_DIR, offline_mode


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


class SessionLoadError(DataError):
    """Raised when FastF1's Session.load() completes without an exception but
    silently left data unloaded (e.g. session.f1_api_support is False for that
    session) — FastF1 only logs a warning in that case, so we must verify
    explicitly rather than trust a clean return from load()."""


def _gp(gp: str | int) -> str | int:
    return int(gp) if isinstance(gp, str) and gp.isdigit() else gp


_LIVE_DATA_UNAVAILABLE = (
    "Live data isn't reachable from this server. Try one of the demo races, "
    "or run locally for every race."
)


@lru_cache(maxsize=1)
def _demo_manifest() -> list[dict]:
    path = DEMO_DATA_DIR / "manifest.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())["races"]


def demo_years() -> list[int]:
    """Years that have at least one bundled demo race."""
    return sorted({r["year"] for r in _demo_manifest()})


def demo_race_names(year: int) -> list[str]:
    """Bundled demo race names for a year, in manifest order."""
    return [r["name"] for r in _demo_manifest() if r["year"] == year]


def demo_sessions_for(year: int, gp: str) -> list[str]:
    """Session codes bundled for a demo race, without touching FastF1's schedule."""
    for r in _demo_manifest():
        if r["year"] == year and r["name"].lower() == str(gp).lower():
            return list(r["sessions"])
    return []


def _resolve_demo(year: int, gp: str | int, session: str) -> tuple[int, int] | None:
    gp_norm = _gp(gp)
    for r in _demo_manifest():
        if r["year"] != year or session not in r["sessions"]:
            continue
        if gp_norm == r["round"] or (isinstance(gp_norm, str) and gp_norm.lower() == r["name"].lower()):
            return year, r["round"]
    return None


def is_demo_race(year: int, gp: str | int) -> bool:
    gp_norm = _gp(gp)
    return any(
        r["year"] == year
        and (gp_norm == r["round"] or (isinstance(gp_norm, str) and gp_norm.lower() == r["name"].lower()))
        for r in _demo_manifest()
    )


class DemoSession:
    """Duck-typed stand-in for a FastF1 Session, backed by bundled demo_data/.

    Exposes .laps/.results as plain DataFrames and works with get_lap(),
    lap_telemetry(), and corners() without ever touching FastF1.
    """

    def __init__(self, year: int, round_number: int, session: str):
        base = DEMO_DATA_DIR / str(year) / str(round_number) / session
        self.year = year
        self.round_number = round_number
        self.session_code = session
        self.laps = pd.read_parquet(base / "laps.parquet")
        self.results = pd.read_parquet(base / "results.parquet")
        self._corners = pd.read_parquet(base / "corners.parquet")
        self._telemetry_dir = base / "telemetry"

    def telemetry_path(self, driver: str) -> Path:
        return self._telemetry_dir / f"{driver.upper()}.parquet"


@lru_cache(maxsize=2)
def load_session(year: int, gp: str | int, session: str = "R", telemetry: bool = True):
    """Load a session and verify the data actually came through.

    FastF1's Session.load() can return normally while leaving .laps (or
    telemetry) unloaded — e.g. when session.f1_api_support is False, it just
    logs a warning and skips loading instead of raising. Accessing .laps
    afterward then raises DataNotLoadedError. We check for that explicitly,
    retry once (covers transient network issues), and raise before ever
    returning so a broken session is never cached by lru_cache.
    """
    last_error: DataNotLoadedError | None = None
    for _attempt in range(2):
        s = fastf1.get_session(year, _gp(gp), session)
        s.load(laps=True, telemetry=telemetry, weather=True, messages=False)
        try:
            _ = s.laps
            if telemetry:
                _ = s.car_data
            return s
        except DataNotLoadedError as e:
            last_error = e
    raise SessionLoadError(
        f"Could not load {year} {gp} {session} from F1's timing service after 2 attempts: {last_error}"
    )


def get_session(year: int, gp: str | int, session: str = "R", telemetry: bool = True):
    """Session-like object for telemetry views: demo_data first, else a live
    FastF1 session via load_session(). In offline_mode(), raises immediately
    (without ever touching FastF1) if this race isn't bundled as demo data.
    """
    demo = _resolve_demo(year, gp, session)
    if demo is not None:
        return DemoSession(*demo, session)
    if offline_mode():
        raise SessionLoadError(_LIVE_DATA_UNAVAILABLE)
    return load_session(year, gp, session, telemetry=telemetry)


def get_lap(session, driver: str, lap: str | int = "fastest"):
    if isinstance(session, DemoSession):
        driver = driver.upper()
        rows = session.laps[session.laps["Driver"] == driver]
        if rows.empty:
            raise DataError(f"No laps for driver {driver}")
        if str(lap) != "fastest":
            raise DataError("Demo data only includes each driver's fastest lap")
        row = rows.loc[rows["LapTime"].idxmin()].copy()
        row["_demo_telemetry_path"] = str(session.telemetry_path(driver))
        return row
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
    """Car + position data for one lap with Distance (m), TimeS (s), Brake (bool).

    Accepts either a live FastF1 Lap (calls .get_telemetry()) or a lap Series
    returned by get_lap() for a DemoSession (reads the bundled parquet file).
    """
    demo_path = lap.get("_demo_telemetry_path") if hasattr(lap, "get") else None
    if demo_path:
        return pd.read_parquet(demo_path)
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
    if isinstance(session, DemoSession):
        return session._corners
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

    Accepts a live FastF1 session, a DemoSession, or a plain raw-laps
    DataFrame (e.g. from race_laps()/Parquet cache). with_weather requires a
    live session.
    """
    if isinstance(session_or_laps, DemoSession):
        session_or_laps = session_or_laps.laps
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
    """Raw laps for a race, as a plain DataFrame. Checks demo_data/ first, then
    data/processed/{year}/{round}.parquet, then falls back to live FastF1
    (which writes the Parquet cache for next time). In offline_mode(), raises
    before ever resolving a round number or touching FastF1 for a non-demo race.
    """
    demo = _resolve_demo(year, gp, "R")
    if demo is not None:
        return DemoSession(*demo, "R").laps
    if offline_mode():
        raise SessionLoadError(_LIVE_DATA_UNAVAILABLE)
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
