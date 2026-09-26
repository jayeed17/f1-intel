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

import fastf1
import pandas as pd

from app.config import PREBUILT_DIR
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


def build_session(year: int, gp: str, round_number: int, session: str) -> int | None:
    """Build one session's parquet files. Returns driver count on success,
    None if the session couldn't be loaded at all (skipped, not fatal)."""
    out_dir = PREBUILT_DIR / str(year) / str(round_number) / session
    try:
        s = load_session(year, round_number, session, telemetry=True)
    except Exception as e:  # noqa: BLE001 — one bad session shouldn't kill the whole build
        print(f"  ! skipping {year} {gp} {session}: could not load ({e})")
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    laps = pd.DataFrame(s.laps)
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
        cn = pd.DataFrame(columns=["Label", "Number", "Distance", "X", "Y"])
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

    print(f"  {gp} {session}: laps={len(laps)} results={len(results)} corners={len(cn)} "
          f"telemetry={written}/{len(drivers)} drivers")
    return written


def downsample_telemetry() -> None:
    """Halve every telemetry file's row count in place (every 2nd sample)."""
    for f in PREBUILT_DIR.rglob("telemetry/*.parquet"):
        df = pd.read_parquet(f)
        df.iloc[::2].reset_index(drop=True).to_parquet(f)


def total_size_bytes() -> int:
    return sum(f.stat().st_size for f in PREBUILT_DIR.rglob("*") if f.is_file())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, action="append", required=True)
    p.add_argument("--race", default=None, help="Only build this event name (substring match)")
    p.add_argument("--sessions", default="Q,R,S,SQ")
    p.add_argument("--only-missing", action="store_true",
                   help="Skip races/sessions already present in the manifest")
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
                                          "sessions": [], "built_at": None})
            to_build = available
            if args.only_missing:
                to_build = [c for c in available if c not in existing["sessions"]]
                if not to_build:
                    continue

            print(f"{year} {gp} (round {round_number}): building {to_build}")
            built_now = []
            for session in to_build:
                if build_session(year, gp, round_number, session) is not None:
                    built_now.append(session)

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


if __name__ == "__main__":
    main()
