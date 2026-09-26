"""Build bundled offline demo data so the dashboard works with zero FastF1
network access (e.g. Streamlit Cloud, where the F1 timing API isn't reachable).

Run locally, where FastF1 works: python -m scripts.build_demo_data
"""
from __future__ import annotations

import json

import pandas as pd

from app.config import DEMO_DATA_DIR
from app.data import corners as compute_corners
from app.data import get_lap, lap_telemetry, load_session

YEAR = 2025
RACES = ["Italian Grand Prix", "Monaco Grand Prix", "British Grand Prix"]
SESSIONS = ["Q", "R"]
RESULTS_COLUMNS = ["Abbreviation", "FullName", "TeamName", "Position", "GridPosition"]
TELEMETRY_COLUMNS = ["Distance", "Speed", "Throttle", "Brake", "TimeS", "X", "Y"]


def build_session(year: int, gp: str, session: str) -> dict:
    print(f"Loading {year} {gp} {session}...")
    s = load_session(year, gp, session, telemetry=True)
    round_number = int(s.event["RoundNumber"])
    out_dir = DEMO_DATA_DIR / str(year) / str(round_number) / session
    out_dir.mkdir(parents=True, exist_ok=True)

    laps = pd.DataFrame(s.laps)
    laps.to_parquet(out_dir / "laps.parquet")

    results = pd.DataFrame(s.results)[RESULTS_COLUMNS]
    results.to_parquet(out_dir / "results.parquet")

    cn = compute_corners(s)
    cn.to_parquet(out_dir / "corners.parquet")

    tel_dir = out_dir / "telemetry"
    tel_dir.mkdir(exist_ok=True)
    drivers = sorted(laps["Driver"].dropna().unique())
    written = 0
    for drv in drivers:
        try:
            lap = get_lap(s, drv, "fastest")
            tel = lap_telemetry(lap)[TELEMETRY_COLUMNS]
        except Exception as e:  # noqa: BLE001 — a driver with no clean lap shouldn't kill the build
            print(f"  ! skipping {drv}: {e}")
            continue
        tel.to_parquet(tel_dir / f"{drv}.parquet")
        written += 1

    print(f"  laps={len(laps)} results={len(results)} corners={len(cn)} telemetry_drivers={written}/{len(drivers)}")
    return {"round": round_number, "drivers": written}


def main() -> None:
    manifest_races = []
    for gp in RACES:
        sessions_built = []
        round_number = None
        for session in SESSIONS:
            info = build_session(YEAR, gp, session)
            round_number = info["round"]
            sessions_built.append(session)
        manifest_races.append({"year": YEAR, "round": round_number, "name": gp, "sessions": sessions_built})

    (DEMO_DATA_DIR / "manifest.json").write_text(json.dumps({"races": manifest_races}, indent=2))

    total = sum(f.stat().st_size for f in DEMO_DATA_DIR.rglob("*") if f.is_file())
    print(f"\ndemo_data/ manifest written for {len(manifest_races)} races.")
    print(f"Total demo_data/ size: {total / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
