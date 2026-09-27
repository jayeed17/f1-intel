"""Build the prebuilt data bundle so the dashboard works with zero FastF1
network access at runtime (e.g. Streamlit Cloud, where F1's live timing API
isn't reachable at all).

Run locally, where FastF1 works:
    python -m scripts.build_prebuilt --year 2025 --year 2026
    python -m scripts.build_prebuilt --year 2026 --only-missing
    python -m scripts.build_prebuilt --year 2025 --race "Monaco Grand Prix"
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import fastf1
import pandas as pd
from fastf1.exceptions import RateLimitExceededError

from app.config import PREBUILT_DIR
from app.data import OpenF1Error, OpenF1Session, SessionLoadError
from app.data import corners as compute_corners
from app.data import get_lap, lap_telemetry, load_session

RESULTS_COLUMNS = ["Abbreviation", "FullName", "TeamName", "Position", "GridPosition"]
TELEMETRY_COLUMNS = ["Distance", "Speed", "Throttle", "Brake", "TimeS", "X", "Y"]
SIZE_BUDGET_BYTES = 150 * 1024 * 1024
_SESSION_NAME_TO_CODE = {
    "Practice 1": "FP1", "Practice 2": "FP2", "Practice 3": "FP3",
    "Sprint Qualifying": "SQ", "Sprint Shootout": "SQ", "Sprint": "S",
    "Qualifying": "Q", "Race": "R",
}

# Seconds to wait before each retry on FastF1's hard rate limit (500 calls/h,
# or 200/h on ergast.com) — ~25 min worst case before giving up on one session.
_RATE_LIMIT_BACKOFFS = [60, 120, 240, 480, 600]

_FAILURE_LABELS = {
    "rate_limited": "RATE LIMITED",
    "download_failed": "DOWNLOAD FAILED",
    "not_published": "NOT PUBLISHED YET",
}


class _Fastf1WarningCapture(logging.Handler):
    """Captures FastF1's internal "Failed to load X data!" warnings.

    FastF1 catches exceptions (network errors, and — we've confirmed live —
    a rate limit mid-fetch) around several of its own sub-loads and just logs
    a warning instead of raising, which leaves .laps/.car_data unset without
    telling calling code why. That's the only way to tell "this genuinely
    isn't published yet" apart from "a download inside FastF1 silently
    failed" — both otherwise surface identically as SessionLoadError.
    """

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record):
        msg = record.getMessage()
        if "failed to load" in msg.lower() or "failed to add" in msg.lower():
            self.messages.append(msg)


def _load_with_diagnosis(year: int, gp: str, round_number: int, session: str):
    """Load a session, classifying *why* it failed instead of guessing.

    Returns (session_obj, None) on success, or (None, (category, detail)) on
    failure where category is "rate_limited", "download_failed", or
    "not_published". Retries with backoff on a hard rate limit rather than
    giving up immediately.
    """
    capture = _Fastf1WarningCapture()
    fastf1_logger = logging.getLogger("fastf1")
    fastf1_logger.addHandler(capture)
    try:
        attempt = 0
        while True:
            capture.messages.clear()
            try:
                return load_session(year, round_number, session, telemetry=True), None
            except RateLimitExceededError as e:
                if attempt >= len(_RATE_LIMIT_BACKOFFS):
                    return None, ("rate_limited", f"gave up after {attempt} retries: {e}")
                wait = _RATE_LIMIT_BACKOFFS[attempt]
                print(f"  ! rate limited loading {gp} {session} — backing off {wait}s "
                      f"(retry {attempt + 1}/{len(_RATE_LIMIT_BACKOFFS)})")
                time.sleep(wait)
                attempt += 1
            except SessionLoadError as e:
                # FastF1's Session.load() completed without raising but left data
                # unset. If it logged a "Failed to load" warning along the way,
                # that's a real (likely transient/rate-limit) download failure,
                # not evidence the session isn't published yet.
                if capture.messages:
                    return None, ("download_failed", "; ".join(capture.messages))
                return None, ("not_published", str(e))
            except Exception as e:  # noqa: BLE001
                return None, ("download_failed", f"{type(e).__name__}: {e}")
    finally:
        fastf1_logger.removeHandler(capture)


def _manifest_path():
    return PREBUILT_DIR / "manifest.json"


def load_manifest() -> dict[tuple[int, str], dict]:
    path = _manifest_path()
    if not path.exists():
        return {}
    races = json.loads(path.read_text())["races"]
    return {(r["year"], r["name"]): r for r in races}


def save_manifest(manifest: dict[tuple[int, str], dict]) -> None:
    races = sorted(manifest.values(), key=lambda r: (r["year"], r["round"]))
    _manifest_path().parent.mkdir(parents=True, exist_ok=True)
    _manifest_path().write_text(json.dumps({"races": races}, indent=2))


def event_sessions(row: pd.Series) -> list[str]:
    """Which session codes actually happened this weekend, per the schedule."""
    codes = []
    for i in range(1, 6):
        code = _SESSION_NAME_TO_CODE.get(row.get(f"Session{i}"))
        if code:
            codes.append(code)
    return codes


def _classify_missing(written: int, driver_count: int, corner_count: int) -> list[str]:
    """What a session is missing after a build attempt -- "telemetry" if no
    driver got any (and there were drivers to try), "corners" if the corner
    map came back empty. A session with anything missing is "partial" rather
    than "complete" in the manifest."""
    missing = []
    if driver_count > 0 and written == 0:
        missing.append("telemetry")
    if corner_count == 0:
        missing.append("corners")
    return missing


def _is_missing_or_partial(session_code: str, existing: dict) -> bool:
    """Whether a session should be (re)built under --only-missing: either
    it's not in the manifest at all, or it's there but marked partial (so
    every run keeps retrying it until a source recovers full data)."""
    if session_code not in existing.get("sessions", []):
        return True
    return existing.get("session_status", {}).get(session_code, {}).get("status") == "partial"


def build_session(year: int, gp: str, round_number: int,
                  session: str) -> tuple[int, str, list[str]] | tuple[None, None, None]:
    """Build one session's parquet files. Returns (driver count, source
    ("fastf1" or "openf1"), missing) on success -- missing is [] for a fully
    complete build, or e.g. ["telemetry", "corners"] for a partial one (see
    _classify_missing). Returns (None, None, None) if the session couldn't be
    loaded from either source at all (skipped, not fatal)."""
    out_dir = PREBUILT_DIR / str(year) / str(round_number) / session
    s, failure = _load_with_diagnosis(year, gp, round_number, session)
    source = "fastf1"
    if failure is not None:
        category, detail = failure
        print(f"  ! {_FAILURE_LABELS[category]}: {year} {gp} {session}: {detail}")
        print(f"  -> trying OpenF1 for {year} {gp} {session}")
        try:
            s = OpenF1Session(year, gp, session)
            source = "openf1"
        except OpenF1Error as e:
            print(f"  ! OPENF1 ALSO FAILED: {year} {gp} {session}: {e}")
            return None, None, None

    out_dir.mkdir(parents=True, exist_ok=True)
    laps = pd.DataFrame(s.laps)
    if source == "openf1":
        # internal OpenF1Session bookkeeping columns -- don't leak into the bundle
        laps = laps.drop(columns=[c for c in ("_driver_number", "_date_start") if c in laps.columns])
    laps.to_parquet(out_dir / "laps.parquet")

    try:
        results = pd.DataFrame(s.results)[RESULTS_COLUMNS]
    except Exception as e:  # noqa: BLE001
        print(f"  ! {gp} {session}: no results ({e})")
        results = pd.DataFrame(columns=RESULTS_COLUMNS)
    results.to_parquet(out_dir / "results.parquet")

    try:
        cn = compute_corners(s)
    except Exception as e:  # noqa: BLE001
        print(f"  ! {gp} {session}: no corner map ({e})")
        cn = pd.DataFrame(columns=["Label", "Number", "Distance", "X", "Y", "Estimated"])
    cn.to_parquet(out_dir / "corners.parquet")

    tel_dir = out_dir / "telemetry"
    tel_dir.mkdir(exist_ok=True)
    drivers = sorted(laps["Driver"].dropna().unique())
    written = 0
    for drv in drivers:
        try:
            lap = get_lap(s, drv, "fastest")
            tel = lap_telemetry(lap)[TELEMETRY_COLUMNS]
        except Exception as e:  # noqa: BLE001
            print(f"  ! {gp} {session}: skipping driver {drv} ({e})")
            continue
        tel.to_parquet(tel_dir / f"{drv}.parquet")
        written += 1

    missing = _classify_missing(written, len(drivers), len(cn))
    status = "partial" if missing else "complete"
    print(f"  {gp} {session} [{source}, {status}]: laps={len(laps)} results={len(results)} corners={len(cn)} "
          f"telemetry={written}/{len(drivers)} drivers")
    return written, source, missing


def downsample_telemetry() -> None:
    """Halve every telemetry file's row count in place (every 2nd sample)."""
    for f in PREBUILT_DIR.rglob("telemetry/*.parquet"):
        df = pd.read_parquet(f)
        df.iloc[::2].reset_index(drop=True).to_parquet(f)


def total_size_bytes() -> int:
    return sum(f.stat().st_size for f in PREBUILT_DIR.rglob("*") if f.is_file())


def missing_completed_sessions(manifest: dict[tuple[int, str], dict], years: list[int],
                               wanted_sessions: list[str], now: pd.Timestamp) -> list[tuple[int, str, str]]:
    """(year, gp, session_code) triples whose scheduled session time is more
    than 6h in the past (comfortably longer than any session itself takes, so
    it should certainly be over and published by then) but that are still
    missing from the manifest after this run -- i.e. neither FastF1 nor
    OpenF1 could build them. Used to make the Action fail loudly instead of
    silently leaving stale data, per race weekend."""
    missing = []
    for year in years:
        try:
            sch = fastf1.get_event_schedule(year, include_testing=False)
        except Exception:  # noqa: BLE001 -- schedule itself unreachable; can't judge, skip
            continue
        for _, row in sch.iterrows():
            gp = row["EventName"]
            built = manifest.get((year, gp), {}).get("sessions", [])
            for i in range(1, 6):
                code = _SESSION_NAME_TO_CODE.get(row.get(f"Session{i}"))
                if not code or code not in wanted_sessions or code in built:
                    continue
                session_time = row.get(f"Session{i}DateUtc")
                if pd.isna(session_time):
                    continue
                session_time = pd.Timestamp(session_time)
                if session_time.tzinfo is None:
                    session_time = session_time.tz_localize("UTC")
                if now - session_time > pd.Timedelta(hours=6):
                    missing.append((year, gp, code))
    return missing


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, action="append", required=True)
    p.add_argument("--race", default=None, help="Only build this event name (substring match)")
    p.add_argument("--sessions", default="Q,R,S,SQ")
    p.add_argument("--only-missing", action="store_true",
                   help="Skip races/sessions already present in the manifest")
    p.add_argument("--force", action="store_true",
                   help="Rebuild even if already present in the manifest (ignores --only-missing)")
    args = p.parse_args()
    wanted_sessions = [s.strip() for s in args.sessions.split(",") if s.strip()]

    manifest = load_manifest()
    now = pd.Timestamp.now(tz="UTC")

    for year in args.year:
        try:
            sch = fastf1.get_event_schedule(year, include_testing=False)
        except Exception as e:  # noqa: BLE001 — e.g. Ergast/F1 API rate limit; don't lose other years
            print(f"! could not fetch the {year} schedule, skipping this year: {e}")
            continue
        dates = pd.to_datetime(sch["EventDate"])
        dates = dates.dt.tz_localize("UTC") if dates.dt.tz is None else dates
        sch = sch[dates < now]
        if args.race:
            sch = sch[sch["EventName"].str.contains(args.race, case=False, na=False)]

        for _, row in sch.iterrows():
            gp = row["EventName"]
            round_number = int(row["RoundNumber"])
            available = [c for c in wanted_sessions if c in event_sessions(row)]
            if not available:
                continue

            key = (year, gp)
            existing = manifest.get(key, {"year": year, "round": round_number, "name": gp,
                                          "sessions": [], "session_status": {}, "built_at": None})
            existing.setdefault("session_status", {})
            to_build = available
            if args.only_missing and not args.force:
                to_build = [c for c in available if _is_missing_or_partial(c, existing)]
                if not to_build:
                    continue

            print(f"{year} {gp} (round {round_number}): building {to_build}")
            built_now = []
            for session in to_build:
                written, source, missing = build_session(year, gp, round_number, session)
                if written is not None:
                    built_now.append(session)
                    existing["session_status"][session] = {
                        "source": source, "status": "partial" if missing else "complete", "missing": missing,
                    }

            if built_now:
                existing["sessions"] = sorted(set(existing["sessions"]) | set(built_now))
                existing["built_at"] = now.isoformat()
                manifest[key] = existing
                save_manifest(manifest)  # persist incrementally so a crash mid-run isn't a total loss

    size = total_size_bytes()
    print(f"\ndata/prebuilt/ size: {size / 1024 / 1024:.2f} MB across {len(manifest)} races")
    if size > SIZE_BUDGET_BYTES:
        print(f"Over the ~{SIZE_BUDGET_BYTES / 1024 / 1024:.0f} MB budget — downsampling telemetry to every 2nd sample...")
        downsample_telemetry()
        new_size = total_size_bytes()
        print(f"New size after downsampling: {new_size / 1024 / 1024:.2f} MB")

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")

    partial = [
        (yr, race["name"], code, st["missing"])
        for (yr, _name), race in manifest.items() if yr in args.year
        for code, st in race.get("session_status", {}).items() if st.get("status") == "partial"
    ]
    if partial:
        lines = [f"- {y} {gp} {code}: missing {', '.join(miss)}" for y, gp, code, miss in sorted(partial)]
        warning = ("## Partial prebuilt sessions\n\nThese only have laps/results (built from a "
                  "degraded source) and will keep being retried on every future run until a "
                  "source recovers the rest:\n\n" + "\n".join(lines) + "\n")
        print("\n" + warning)
        if summary_path:
            with open(summary_path, "a") as f:
                f.write(warning)

    missing = missing_completed_sessions(manifest, args.year, wanted_sessions, now)
    if missing:
        lines = [f"- {year} {gp} {code}" for year, gp, code in missing]
        summary = ("## Missing prebuilt sessions\n\nThese sessions' scheduled time is "
                    "more than 6h in the past but they're still absent from the manifest "
                    "(FastF1 and OpenF1 both failed to build them):\n\n" + "\n".join(lines) + "\n")
        print("\n" + summary)
        if summary_path:
            with open(summary_path, "a") as f:
                f.write(summary)
        sys.exit(1)


if __name__ == "__main__":
    main()
