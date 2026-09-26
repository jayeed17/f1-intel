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
import requests
from fastf1.exceptions import DataNotLoadedError

from app.config import CACHE_DIR, PREBUILT_DIR, PROCESSED_DIR


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
    """Raised when no data source (prebuilt bundle, OpenF1, FastF1) could
    serve this session."""


_SESSION_NOT_YET_AVAILABLE = (
    "This session isn't available yet — it's added automatically a few "
    "hours after it ends."
)


def _gp(gp: str | int) -> str | int:
    return int(gp) if isinstance(gp, str) and gp.isdigit() else gp


# --------------------------------------------------------------------------
# Prebuilt bundle: data/prebuilt/{year}/{round}/{session}/, refreshed by
# scripts/build_prebuilt.py and the update-data GitHub Action.
# --------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _prebuilt_manifest() -> list[dict]:
    path = PREBUILT_DIR / "manifest.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())["races"]


def prebuilt_races() -> list[dict]:
    """All bundled races, newest first (by year then round, descending)."""
    return sorted(_prebuilt_manifest(), key=lambda r: (r["year"], r["round"]), reverse=True)


def prebuilt_sessions_for(year: int, gp: str) -> list[str]:
    """Session codes bundled for a race, without touching FastF1's schedule."""
    for r in _prebuilt_manifest():
        if r["year"] == year and r["name"].lower() == str(gp).lower():
            return list(r["sessions"])
    return []


def prebuilt_built_at(year: int, gp: str) -> str | None:
    for r in _prebuilt_manifest():
        if r["year"] == year and r["name"].lower() == str(gp).lower():
            return r.get("built_at")
    return None


def _resolve_prebuilt(year: int, gp: str | int, session: str) -> tuple[int, int] | None:
    gp_norm = _gp(gp)
    for r in _prebuilt_manifest():
        if r["year"] != year or session not in r["sessions"]:
            continue
        if gp_norm == r["round"] or (isinstance(gp_norm, str) and gp_norm.lower() == r["name"].lower()):
            return year, r["round"]
    return None


def is_prebuilt_race(year: int, gp: str | int) -> bool:
    gp_norm = _gp(gp)
    return any(
        r["year"] == year
        and (gp_norm == r["round"] or (isinstance(gp_norm, str) and gp_norm.lower() == r["name"].lower()))
        for r in _prebuilt_manifest()
    )


class PrebuiltSession:
    """Duck-typed stand-in for a FastF1 Session, backed by data/prebuilt/.

    Exposes .laps/.results as plain DataFrames and works with get_lap(),
    lap_telemetry(), and corners() without ever touching FastF1.
    """

    def __init__(self, year: int, round_number: int, session: str):
        base = PREBUILT_DIR / str(year) / str(round_number) / session
        self.year = year
        self.round_number = round_number
        self.session_code = session
        self.laps = pd.read_parquet(base / "laps.parquet")
        self.results = pd.read_parquet(base / "results.parquet")
        self._corners = pd.read_parquet(base / "corners.parquet")
        self._telemetry_dir = base / "telemetry"

    def telemetry_path(self, driver: str) -> Path:
        return self._telemetry_dir / f"{driver.upper()}.parquet"


# --------------------------------------------------------------------------
# OpenF1 fallback (api.openf1.org) — used for a race that isn't in the
# prebuilt bundle yet. No circuit-map data is available from OpenF1, so
# corners() returns empty; TrackStatus/IsAccurate are approximated (OpenF1
# doesn't expose flag history or FastF1's accuracy check).
# --------------------------------------------------------------------------

OPENF1_BASE = "https://api.openf1.org/v1"
_OPENF1_SESSION_NAME = {
    "FP1": "Practice 1", "FP2": "Practice 2", "FP3": "Practice 3",
    "SQ": "Sprint Qualifying", "S": "Sprint", "Q": "Qualifying", "R": "Race",
}


class OpenF1Error(Exception):
    """Raised when OpenF1 has no usable data for a request (network error,
    unknown session, or an empty result set)."""


def _openf1_get(path: str, timeout: float = 10, **params) -> list[dict]:
    try:
        r = requests.get(f"{OPENF1_BASE}/{path}", params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        raise OpenF1Error(str(e)) from e


def _openf1_find_session(year: int, gp: str | int, session: str) -> dict:
    name = _OPENF1_SESSION_NAME.get(session, session)
    rows = _openf1_get("sessions", year=year, session_name=name)
    if not rows:
        raise OpenF1Error(f"No OpenF1 session for {year} {gp} {session}")
    if isinstance(gp, str) and not gp.isdigit():
        needle = gp.lower().replace(" grand prix", "").strip()
        matches = [
            r for r in rows
            if needle in (r.get("location") or "").lower()
            or needle in (r.get("circuit_short_name") or "").lower()
            or needle in (r.get("country_name") or "").lower()
        ]
        if matches:
            rows = matches
    rows.sort(key=lambda r: r.get("date_start") or "")
    return rows[-1]


class OpenF1Session:
    """Duck-typed stand-in for a FastF1 Session, backed by the OpenF1 API.
    Used when a race isn't in the prebuilt bundle."""

    def __init__(self, year: int, gp: str | int, session: str):
        meta = _openf1_find_session(year, gp, session)
        self.session_key = meta["session_key"]
        self.year = year
        self.session_code = session
        self._corners = pd.DataFrame(columns=_CORNER_COLUMNS)

        drivers = _openf1_get("drivers", session_key=self.session_key)
        by_number = {d["driver_number"]: d for d in drivers}
        self.results = pd.DataFrame([
            {"Abbreviation": d.get("name_acronym"), "FullName": d.get("full_name"),
             "TeamName": d.get("team_name"), "Position": None, "GridPosition": None}
            for d in drivers
        ])

        laps_raw = _openf1_get("laps", session_key=self.session_key)
        if not laps_raw:
            raise OpenF1Error("OpenF1 returned no laps for this session")
        stints = _openf1_get("stints", session_key=self.session_key)
        pits = _openf1_get("pit", session_key=self.session_key)
        pit_keys = {(p["driver_number"], p["lap_number"]) for p in pits}

        def _td(seconds):
            return pd.Timedelta(seconds=seconds) if seconds is not None else pd.NaT

        rows = []
        for lap in laps_raw:
            drv = by_number.get(lap["driver_number"], {})
            stint = next(
                (s for s in stints if s["driver_number"] == lap["driver_number"]
                 and s["lap_start"] <= lap["lap_number"] <= (s.get("lap_end") or lap["lap_number"])),
                None,
            )
            rows.append({
                "Driver": drv.get("name_acronym"),
                "Team": drv.get("team_name"),
                "LapNumber": float(lap["lap_number"]),
                "Stint": float(stint["stint_number"]) if stint else 1.0,
                "Compound": stint["compound"].upper() if stint and stint.get("compound") else None,
                "TyreLife": (float(stint.get("tyre_age_at_start") or 0) + (lap["lap_number"] - stint["lap_start"]))
                            if stint else np.nan,
                "LapTime": _td(lap.get("lap_duration")),
                "PitInTime": _td(1) if (lap["driver_number"], lap["lap_number"]) in pit_keys else pd.NaT,
                "PitOutTime": _td(1) if lap.get("is_pit_out_lap") else pd.NaT,
                "TrackStatus": "1",
                "IsAccurate": lap.get("lap_duration") is not None,
                "Sector1Time": _td(lap.get("duration_sector_1")),
                "Sector2Time": _td(lap.get("duration_sector_2")),
                "Sector3Time": _td(lap.get("duration_sector_3")),
                "SpeedST": lap.get("st_speed"),
                "_driver_number": lap["driver_number"],
                "_date_start": lap.get("date_start"),
            })
        self.laps = pd.DataFrame(rows)

    def lap_telemetry(self, driver_number: int, date_start: str, duration: float) -> pd.DataFrame:
        start = pd.Timestamp(date_start)
        end = start + pd.Timedelta(seconds=(duration or 0) + 1)
        car = _openf1_get("car_data", session_key=self.session_key, driver_number=driver_number,
                          **{"date>=": start.isoformat(), "date<=": end.isoformat()})
        if not car:
            raise OpenF1Error("No OpenF1 car telemetry for this lap")
        car_df = pd.DataFrame(car)
        car_df["date"] = pd.to_datetime(car_df["date"])
        car_df = car_df.sort_values("date").reset_index(drop=True)
        car_df["TimeS"] = (car_df["date"] - start).dt.total_seconds()
        car_df["Brake"] = car_df["brake"].astype(float) > 0
        car_df = car_df.rename(columns={"speed": "Speed", "throttle": "Throttle"})
        # OpenF1 doesn't provide distance-along-lap; approximate by integrating
        # speed (km/h -> m/s) over time, same idea as FastF1's add_distance().
        dt = car_df["TimeS"].diff().fillna(0).clip(lower=0)
        car_df["Distance"] = (car_df["Speed"] / 3.6 * dt).cumsum()
        loc = _openf1_get("location", session_key=self.session_key, driver_number=driver_number,
                          **{"date>=": start.isoformat(), "date<=": end.isoformat()})
        if loc:
            loc_df = pd.DataFrame(loc)
            loc_df["date"] = pd.to_datetime(loc_df["date"])
            loc_df = loc_df.sort_values("date")
            car_df = pd.merge_asof(car_df, loc_df[["date", "x", "y"]], on="date", direction="nearest")
            car_df = car_df.rename(columns={"x": "X", "y": "Y"})
        else:
            car_df["X"] = np.nan
            car_df["Y"] = np.nan
        return car_df[["Distance", "Speed", "Throttle", "Brake", "TimeS", "X", "Y"]].reset_index(drop=True)


# --------------------------------------------------------------------------
# Live FastF1 (last resort — unreachable from a typical cloud deployment).
# --------------------------------------------------------------------------

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
    """Session-like object for telemetry views. Tries, in order: the
    prebuilt bundle, the OpenF1 API, then a live FastF1 session. Raises
    SessionLoadError with a friendly message if none of them work.
    """
    prebuilt = _resolve_prebuilt(year, gp, session)
    if prebuilt is not None:
        return PrebuiltSession(*prebuilt, session)
    try:
        return OpenF1Session(year, gp, session)
    except OpenF1Error:
        pass
    try:
        return load_session(year, gp, session, telemetry=telemetry)
    except Exception as e:
        raise SessionLoadError(_SESSION_NOT_YET_AVAILABLE) from e


def get_lap(session, driver: str, lap: str | int = "fastest"):
    if isinstance(session, PrebuiltSession):
        driver = driver.upper()
        rows = session.laps[session.laps["Driver"] == driver]
        if rows.empty:
            raise DataError(f"No laps for driver {driver}")
        if str(lap) != "fastest":
            raise DataError("Prebuilt data only includes each driver's fastest lap")
        row = rows.loc[rows["LapTime"].idxmin()].copy()
        tel_path = session.telemetry_path(driver)
        if not tel_path.exists():
            raise DataError(f"No telemetry available for {driver} in this session")
        row["_prebuilt_telemetry_path"] = str(tel_path)
        return row
    if isinstance(session, OpenF1Session):
        driver = driver.upper()
        rows = session.laps[(session.laps["Driver"] == driver) & session.laps["LapTime"].notna()]
        if rows.empty:
            raise DataError(f"No laps for driver {driver}")
        if str(lap) != "fastest":
            raise DataError("OpenF1 fallback only supports each driver's fastest lap")
        row = rows.loc[rows["LapTime"].idxmin()].copy()
        row["_openf1_session"] = session
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

    Accepts a live FastF1 Lap (calls .get_telemetry()), a lap Series from
    get_lap() for a PrebuiltSession (reads the bundled parquet file), or one
    from an OpenF1Session (fetches car_data/location live from OpenF1).
    """
    prebuilt_path = lap.get("_prebuilt_telemetry_path") if hasattr(lap, "get") else None
    if prebuilt_path:
        return pd.read_parquet(prebuilt_path)
    openf1_session = lap.get("_openf1_session") if hasattr(lap, "get") else None
    if openf1_session is not None:
        duration = lap["LapTime"].total_seconds() if pd.notna(lap["LapTime"]) else 0
        return openf1_session.lap_telemetry(lap["_driver_number"], lap["_date_start"], duration)
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
    track (raises AttributeError instead of a clean None). Separately,
    get_circuit_info() can raise KeyError if the session's fastest lap has
    malformed position data (missing a 'Date' column) — a real, reproducible
    upstream data quirk seen on e.g. 2026 Monaco Race, not a caching issue.
    Fall back to the previous year's map for the same key; if that's
    unavailable too, return an empty frame so callers can degrade instead of
    crashing. Prebuilt/OpenF1 sessions carry no circuit map either way
    (OpenF1 doesn't expose one).
    """
    if isinstance(session, (PrebuiltSession, OpenF1Session)):
        return session._corners
    try:
        info = session.get_circuit_info()
    except (AttributeError, KeyError):
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

    Accepts a live FastF1 session, a PrebuiltSession/OpenF1Session, or a plain
    raw-laps DataFrame (e.g. from race_laps()/Parquet cache). with_weather
    requires a live session.
    """
    if isinstance(session_or_laps, (PrebuiltSession, OpenF1Session)):
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
    """Raw laps for a race, as a plain DataFrame. Tries, in order: the
    prebuilt bundle, data/processed/{year}/{round}.parquet, the OpenF1 API,
    then a live FastF1 load (which writes the Parquet cache for next time).
    """
    prebuilt = _resolve_prebuilt(year, gp, "R")
    if prebuilt is not None:
        return PrebuiltSession(*prebuilt, "R").laps

    path = None
    try:
        path = _processed_path(year, _round_number(year, gp))
        if path.exists():
            return pd.read_parquet(path)
    except Exception:
        path = None  # round number needs FastF1's schedule; unreachable is fine, just skip the cache

    try:
        return OpenF1Session(year, gp, "R").laps
    except OpenF1Error:
        pass

    try:
        session = load_session(year, gp, "R", telemetry=False)
    except Exception as e:
        raise SessionLoadError(_SESSION_NOT_YET_AVAILABLE) from e
    df = pd.DataFrame(session.laps)
    if path is not None:
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
