"""Exercise every function behind every API route against a real session.

Not a pytest suite (network + slow). Run manually:
    python -m scripts.smoke_test --year 2025 --gp Monza
    python -m scripts.smoke_test --year 2026 --gp <round>
"""
from __future__ import annotations

import argparse
import traceback

import fastf1

from app.analysis.braking import assign_corners, braking_zones, compare_corners
from app.analysis.delta import lap_delta, minisector_dominance
from app.analysis.pits import estimate_pit_loss, pit_stops
from app.analysis.team_report import team_report
from app.data import clean_laps, corners, get_lap, lap_telemetry, load_session
from app.models.degradation import compound_model, stint_degradation
from app.models.strategy import compare_actual, simulate

DRIVERS = ["VER", "NOR", "LEC"]


def section(name: str):
    print(f"\n{'=' * 10} {name} {'=' * 10}")


def run(year: int, gp: str):
    ok, failed = [], []

    def guard(name, fn):
        try:
            fn()
            ok.append(name)
        except Exception as e:  # noqa: BLE001
            failed.append((name, e))
            print(f"  !! {name} FAILED: {e}")
            traceback.print_exc()

    section("schedule")
    guard("schedule", lambda: print(
        fastf1.get_event_schedule(year, include_testing=False)[["RoundNumber", "EventName"]].tail(5)))

    section("load sessions")
    race = quali = None

    def _load_race():
        nonlocal race
        race = load_session(year, gp, "R")
    guard("load race", _load_race)

    def _load_quali():
        nonlocal quali
        quali = load_session(year, gp, "Q")
    guard("load quali", _load_quali)

    if race is None or quali is None:
        print("\nCannot continue without both sessions loaded.")
        return ok, failed

    section("drivers")
    guard("drivers", lambda: print(
        race.results[["Abbreviation", "FullName", "TeamName", "Position", "GridPosition"]].head(5)))

    section("braking (race, VER, fastest lap)")

    def _braking():
        lap = get_lap(race, "VER", "fastest")
        tel = lap_telemetry(lap)
        zones = assign_corners(braking_zones(tel), corners(race))
        print(f"  lap {int(lap['LapNumber'])}, lap time {lap['LapTime'].total_seconds():.3f}s, "
              f"telemetry rows={len(tel)}, braking zones={len(zones)}")
        print(zones[["corner", "brake_start_m", "min_kph", "avg_decel_g"]].to_string(index=False))
        print("  circuit corners:", corners(race)["Label"].tolist())
    guard("braking", _braking)

    section("compare (quali, VER vs NOR)")

    def _compare():
        ta = lap_telemetry(get_lap(quali, "VER", "fastest"))
        tb = lap_telemetry(get_lap(quali, "NOR", "fastest"))
        d = lap_delta(ta, tb)
        c = compare_corners(ta, tb, corners(quali))
        print(f"  final gap NOR-VER: {d['delta_s'].iloc[-1]:.3f}s, delta rows={len(d)}, corner rows={len(c)}")
        print(c.head(5).to_string(index=False))
    guard("compare", _compare)

    section("dominance (quali, VER,NOR,LEC)")

    def _dominance():
        tels = {d: lap_telemetry(get_lap(quali, d)) for d in DRIVERS}
        pts = minisector_dominance(tels, n=25)
        share = pts.groupby("Winner")["Minisector"].nunique().to_dict()
        print(f"  points={len(pts)}, minisectors won={share}")
    guard("dominance", _dominance)

    section("degradation (race)")
    cl = None

    def _degradation():
        nonlocal cl
        cl = clean_laps(race)
        cm = compound_model(cl)
        sd = stint_degradation(cl)
        print(f"  clean laps={len(cl)}, compound model={cm}")
        print(f"  stints={len(sd)}")
        print(sd.head(8).to_string(index=False))
    guard("degradation", _degradation)

    section("pits (race)")
    stops = None
    loss = None

    def _pits():
        nonlocal stops, loss
        stops = pit_stops(race.laps)
        loss = estimate_pit_loss(stops)
        print(f"  stops={len(stops)}, estimated pit loss={loss}s")
        print(stops.head(8).to_string(index=False))
    guard("pits", _pits)

    section("strategy (race)")

    def _strategy():
        model = compound_model(clean_laps(race))
        total = int(getattr(race, "total_laps", None) or race.laps["LapNumber"].max())
        plen = loss if loss is not None else estimate_pit_loss(pit_stops(race.laps))
        sims = simulate(total, model, plen, max_stops=2, top=5)
        actual = compare_actual(race.laps, total, model, plen, float(sims["total_s"].iloc[0]))
        print(f"  total_laps={total}, pit_loss={plen}, model={model}")
        print(sims.to_string(index=False))
        print(actual.head(8).to_string(index=False))
    guard("strategy", _strategy)

    section("team-report (race)")
    guard("team-report", lambda: print(
        team_report(race)[["Team", "best_lap", "race_pace_gap", "improve"]].to_string(index=False)))

    print(f"\n{len(ok)} ok, {len(failed)} failed: {[n for n, _ in failed]}")
    return ok, failed


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, default=2025)
    p.add_argument("--gp", default="Monza")
    args = p.parse_args()
    run(args.year, args.gp)
