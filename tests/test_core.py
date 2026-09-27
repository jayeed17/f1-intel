"""Offline tests on synthetic data. No network, no FastF1 downloads."""
import numpy as np
import pandas as pd
import pytest

from app.analysis.braking import assign_corners, braking_zones, compare_corners
from app.analysis.delta import lap_delta, minisector_dominance
from app.analysis.pits import estimate_pit_loss, pit_stops
from app.analysis.team_report import team_report
from app.data import clean_laps
from app.models.degradation import compound_model, stint_degradation
from app.models.strategy import plan_time, simulate, stint_time


def fake_lap(brake_at=(1000, 3000), scale=1.0, n=2000, length=5000.0):
    d = np.linspace(0, length, n)
    speed = np.full(n, 300.0)
    brake = np.zeros(n, bool)
    for b in brake_at:
        z = (d >= b) & (d < b + 120)
        brake[z] = True
        slow = (d >= b) & (d < b + 300)
        speed[slow] = 300 - 200 * np.sin(np.pi * (d[slow] - b) / 300)
    dt = np.diff(d, prepend=0) / (speed / 3.6) * scale
    t = np.cumsum(dt)
    x, y = np.cos(d / length * 2 * np.pi) * 800, np.sin(d / length * 2 * np.pi) * 800
    return pd.DataFrame({"Distance": d, "Speed": speed, "Brake": brake, "TimeS": t,
                         "Throttle": np.where(brake, 0, 100), "X": x, "Y": y})


CORNERS = pd.DataFrame({"Label": ["T1", "T2"], "Number": [1, 2],
                        "Distance": [1150.0, 3150.0], "X": [0, 0], "Y": [0, 0]})


def test_braking_zones_found():
    z = braking_zones(fake_lap())
    assert len(z) == 2
    assert abs(z["brake_start_m"].iloc[0] - 1000) < 5
    assert (z["speed_drop_kph"] > 150).all()
    assert list(assign_corners(z, CORNERS)["corner"]) == ["T1", "T2"]


def test_compare_corners_brake_point_diff():
    a, b = fake_lap(), fake_lap(brake_at=(1020, 3000))
    c = compare_corners(a, b, CORNERS)
    assert c.loc[c.corner == "T1", "brake_point_diff_m"].iloc[0] > 15


def test_lap_delta_slower_lap_positive():
    d = lap_delta(fake_lap(), fake_lap(scale=1.01))
    assert d["delta_s"].iloc[-1] > 0
    assert d["delta_s"].is_monotonic_increasing


def test_dominance():
    pts = minisector_dominance({"AAA": fake_lap(), "BBB": fake_lap(brake_at=(1000, 3100))}, n=10)
    assert set(pts["Winner"]) <= {"AAA", "BBB"}
    assert pts["Minisector"].between(0, 9).all()


def test_has_position_data():
    from app.data import has_position_data

    assert has_position_data(fake_lap())
    no_pos = fake_lap().assign(X=np.nan, Y=np.nan)
    assert not has_position_data(no_pos)


def test_speed_trace_corners_finds_local_minima():
    """Corner-fallback tier 3: estimate apexes from the speed trace's local
    minima when no circuit map (real or reused) is available at all."""
    from app.data import _speed_trace_corners

    cn = _speed_trace_corners(fake_lap())
    assert list(cn["Label"]) == ["C1", "C2"]
    assert list(cn["Number"]) == [1, 2]
    assert cn["Estimated"].all()
    assert abs(cn["Distance"].iloc[0] - 1150) < 50
    assert abs(cn["Distance"].iloc[1] - 3150) < 50


def test_speed_trace_corners_works_without_position_data():
    """Must only need Distance/Speed -- works on the car-data-only telemetry
    fallback from lap_telemetry(), which has no X/Y."""
    from app.data import _speed_trace_corners

    tel = fake_lap().drop(columns=["X", "Y"])
    cn = _speed_trace_corners(tel)
    assert len(cn) == 2
    assert cn["X"].isna().all() and cn["Y"].isna().all()


def test_speed_trace_corners_ignores_shallow_dips():
    """A <15 kph dip (e.g. a kink, not a real corner) must not register."""
    from app.data import _speed_trace_corners

    tel = fake_lap(brake_at=())  # flat speed, no braking zones
    d = tel["Distance"].to_numpy()
    speed = tel["Speed"].to_numpy().copy()
    dip = (d >= 2000) & (d < 2100)
    speed[dip] -= 8.0  # shallow, well under the 15 kph threshold
    cn = _speed_trace_corners(tel.assign(Speed=speed))
    assert cn.empty


def fake_race(n_drivers=4, laps=50, pit=25):
    rows = []
    for i in range(n_drivers):
        for lap in range(1, laps + 1):
            stint = 1 if lap <= pit else 2
            comp = "MEDIUM" if stint == 1 else "HARD"
            life = lap if stint == 1 else lap - pit
            deg = 0.08 if comp == "MEDIUM" else 0.04
            base = 90 + i * 0.3 + (0 if comp == "MEDIUM" else 0.3)
            lt = base + deg * life - 0.035 * (lap - 1)
            rows.append({"Driver": f"D{i}", "Team": f"T{i}", "LapNumber": lap, "Stint": stint,
                         "Compound": comp, "TyreLife": life, "LapTimeS": lt,
                         "LapTime": pd.Timedelta(seconds=lt + (20 if lap in (pit, pit + 1) else 0)),
                         "PitInTime": pd.Timedelta(seconds=1000) if lap == pit else pd.NaT,
                         "PitOutTime": pd.Timedelta(seconds=1022) if lap == pit + 1 else pd.NaT})
    return pd.DataFrame(rows)


def test_degradation_recovers_slopes():
    race = fake_race()
    sd = stint_degradation(race)
    med = sd[sd.Compound == "MEDIUM"]["DegPerLap"]
    assert np.allclose(med, 0.08, atol=1e-3)
    m = compound_model(race)
    assert abs(m["MEDIUM"]["deg"] - 0.08) < 0.01 and abs(m["HARD"]["deg"] - 0.04) < 0.01


def test_stint_time_formula():
    assert stint_time(1.0, 0.5, 3) == 3 * 1.0 + 0.5 * (1 + 2 + 3)


def test_strategy_rules():
    model = {"SOFT": {"offset": -0.5, "deg": 0.15}, "MEDIUM": {"offset": 0.0, "deg": 0.08},
             "HARD": {"offset": 0.4, "deg": 0.04}}
    sims = simulate(50, model, pit_loss=22, max_stops=2, min_stint=5)
    assert sims["gap_s"].iloc[0] == 0
    for comps, lens in zip(sims["compounds"], sims["stint_lengths"]):
        assert len(set(comps.split("-"))) >= 2 and sum(lens) == 50 and min(lens) >= 5
    best = sims.iloc[0]
    plan = list(zip([{"S": "SOFT", "M": "MEDIUM", "H": "HARD"}[c] for c in best["compounds"].split("-")],
                    best["stint_lengths"]))
    assert abs(plan_time(plan, model, 22) - best["total_s"]) < 0.01


def test_pit_stops():
    st = pit_stops(fake_race())
    assert len(st) == 4 and (st["PitLaneS"] == 22).all()
    assert 30 < estimate_pit_loss(st) < 45


def test_tyre_ml_train_and_predict(tmp_path, monkeypatch):
    from app.models import tyre_ml
    monkeypatch.setattr(tyre_ml, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(tyre_ml, "MODEL_PATH", tmp_path / "m.joblib")
    monkeypatch.setattr(tyre_ml, "METRICS_PATH", tmp_path / "m.json")
    df = pd.concat([tyre_ml.build_features(fake_race().assign(TrackTemp=35.0, AirTemp=25.0), c)
                    for c in ["Monza", "Spa", "Silverstone"]], ignore_index=True)
    metrics = tyre_ml.train(df)
    assert metrics["circuits"] == 3 and "cv_mae_s" in metrics
    preds = tyre_ml.predict(tyre_ml.load_model(), [{"TyreLife": 20, "LapNumber": 20, "Compound": "MEDIUM"}])
    assert len(preds) == 1


def test_api_health():
    from fastapi.testclient import TestClient
    from app.main import app
    r = TestClient(app).get("/health")
    assert r.status_code == 200 and r.json()["ok"]


def test_clean_laps_from_dataframe():
    """clean_laps must accept a plain raw-laps DataFrame (Parquet cache path),
    not just a live FastF1 session, and reproduce Laps.pick_quicklaps()
    .pick_wo_box().pick_track_status("1") plus the IsAccurate filter."""
    rows = [
        dict(LapTime=82.0, PitInTime=pd.NaT, PitOutTime=pd.NaT, TrackStatus="1",
             IsAccurate=True, TyreLife=5, Compound="MEDIUM"),
        dict(LapTime=83.0, PitInTime=pd.NaT, PitOutTime=pd.NaT, TrackStatus="1",
             IsAccurate=True, TyreLife=6, Compound="MEDIUM"),
        dict(LapTime=84.0, PitInTime=pd.Timedelta(seconds=1000), PitOutTime=pd.NaT,
             TrackStatus="1", IsAccurate=True, TyreLife=7, Compound="MEDIUM"),  # pit lap
        dict(LapTime=84.5, PitInTime=pd.NaT, PitOutTime=pd.NaT, TrackStatus="4",
             IsAccurate=True, TyreLife=8, Compound="MEDIUM"),  # not green flag
        dict(LapTime=83.5, PitInTime=pd.NaT, PitOutTime=pd.NaT, TrackStatus="1",
             IsAccurate=False, TyreLife=9, Compound="MEDIUM"),  # inaccurate
        dict(LapTime=95.0, PitInTime=pd.NaT, PitOutTime=pd.NaT, TrackStatus="1",
             IsAccurate=True, TyreLife=10, Compound="MEDIUM"),  # slower than 107%
    ]
    df = pd.DataFrame(rows)
    df["LapTime"] = df["LapTime"].apply(lambda s: pd.Timedelta(seconds=s))
    clean = clean_laps(df)
    assert sorted(clean["TyreLife"].tolist()) == [5, 6]
    assert set(clean["LapTimeS"].round(1)) == {82.0, 83.0}


def test_race_laps_cache_roundtrip(tmp_path, monkeypatch):
    """First call loads via FastF1 (mocked) and writes Parquet; second call is
    served straight from data/processed/{year}/{round}.parquet without touching
    FastF1 again."""
    from app import data as data_mod

    monkeypatch.setattr(data_mod, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(data_mod, "_resolve_prebuilt", lambda *a, **k: None)

    def _openf1_unavailable(*a, **k):
        raise data_mod.OpenF1Error("not available in this test")

    monkeypatch.setattr(data_mod, "OpenF1Session", _openf1_unavailable)
    calls = {"n": 0}

    class FakeSession:
        laps = fake_race()

    def fake_load_session(year, gp, session, telemetry=True):
        calls["n"] += 1
        return FakeSession()

    monkeypatch.setattr(data_mod, "load_session", fake_load_session)

    df1 = data_mod.race_laps(2099, 1)
    assert calls["n"] == 1
    assert (tmp_path / "2099" / "1.parquet").exists()

    df2 = data_mod.race_laps(2099, 1)
    assert calls["n"] == 1  # served from Parquet, FastF1 not touched again
    pd.testing.assert_frame_equal(df1, df2)


def test_team_report_from_dataframe():
    """team_report must accept a plain raw-laps DataFrame (Parquet cache path),
    not just a live FastF1 session."""
    df = fake_race(n_drivers=2, laps=20, pit=10)
    df["TrackStatus"] = "1"
    df["IsAccurate"] = True
    df["Sector1Time"] = df["LapTime"] * 0.3
    df["Sector2Time"] = df["LapTime"] * 0.4
    df["Sector3Time"] = df["LapTime"] * 0.3
    df["SpeedST"] = 300 - df["Driver"].str[1:].astype(int) * 5

    rep = team_report(df)
    assert len(rep) == 2
    assert "improve" in rep and "race_pace_gap" in rep
    assert (rep["best_lap_gap"] >= 0).all() and rep["best_lap_gap"].min() == 0


def test_load_session_raises_and_does_not_cache(monkeypatch):
    """A session whose load() completes without raising but leaves .laps
    unloaded (e.g. f1_api_support=False) must surface as SessionLoadError,
    retry once, and never be cached by lru_cache."""
    from app import data as data_mod

    load_calls = {"n": 0}

    class FakeSession:
        def load(self, **kwargs):
            load_calls["n"] += 1

        @property
        def laps(self):
            raise data_mod.DataNotLoadedError(
                "The data you are trying to access has not been loaded yet. See `Session.load`"
            )

    monkeypatch.setattr(data_mod.fastf1, "get_session", lambda *a, **k: FakeSession())
    data_mod.load_session.cache_clear()

    with pytest.raises(data_mod.SessionLoadError):
        data_mod.load_session(2099, "Fake Grand Prix", "R", telemetry=False)

    assert load_calls["n"] == 2  # initial attempt + one retry
    assert data_mod.load_session.cache_info().currsize == 0  # never cached
    data_mod.load_session.cache_clear()


def test_prebuilt_manifest_structure():
    """Every manifest entry must point at files that actually exist on disk."""
    from app import data as data_mod

    manifest = data_mod._prebuilt_manifest()
    assert manifest, "data/prebuilt/manifest.json is empty — run scripts/build_prebuilt.py"
    for race in manifest:
        assert {"year", "round", "name", "sessions", "built_at"} <= set(race)
        assert race["sessions"], f"{race['name']} has no sessions listed"
        for session in race["sessions"]:
            base = data_mod.PREBUILT_DIR / str(race["year"]) / str(race["round"]) / session
            assert (base / "laps.parquet").exists()
            assert (base / "results.parquet").exists()
            assert (base / "corners.parquet").exists()


def test_prebuilt_races_render_all_views_offline(monkeypatch):
    """A sample of bundled races/sessions (oldest + newest, to keep this test
    fast against a large bundle) must serve all 6 dashboard views through the
    data layer with zero FastF1 network calls."""
    from app import data as data_mod

    def _boom(*a, **k):
        raise AssertionError("FastF1 network call attempted while serving prebuilt data")

    monkeypatch.setattr(data_mod.fastf1, "get_session", _boom)

    manifest = data_mod._prebuilt_manifest()
    assert manifest, "data/prebuilt/manifest.json is empty — run scripts/build_prebuilt.py"
    ordered = sorted(manifest, key=lambda r: (r["year"], r["round"]))
    sample = [ordered[0], ordered[-1]] if len(ordered) > 1 else ordered

    telemetry_cols = {"Distance", "Speed", "Throttle", "Brake", "TimeS", "X", "Y"}

    for race in sample:
        year, gp = race["year"], race["name"]
        for session in race["sessions"]:
            # A session built from a degraded source (e.g. OpenF1 without car
            # data) is legitimately telemetry-less -- that's what the
            # dashboard's session_missing() check is for; skip the
            # telemetry-dependent views here the same way it does.
            if "telemetry" in data_mod.session_missing(year, gp, session):
                continue

            s = data_mod.get_session(year, gp, session)
            drivers = sorted(s.laps["Driver"].dropna().unique())
            assert len(drivers) >= 2

            tels = {}
            for drv in drivers[:3]:
                lap = data_mod.get_lap(s, drv)
                tel = data_mod.lap_telemetry(lap)
                assert not tel.empty
                assert telemetry_cols <= set(tel.columns)
                tels[drv] = tel
            # A real circuit can legitimately have no MultiViewer map yet (e.g. a
            # redesigned track early in its first season) — corners() degrades to
            # empty rather than crashing; assert the downstream views still run.
            cn = data_mod.corners(s)

            a, b = drivers[0], drivers[1]
            assert isinstance(assign_corners(braking_zones(tels[a]), cn), pd.DataFrame)  # Braking view
            assert not lap_delta(tels[a], tels[b]).empty  # Head to head view
            compare_corners(tels[a], tels[b], cn)
            assert not minisector_dominance(tels, n=10).empty  # Track dominance view

        if "R" not in race["sessions"]:
            continue
        # Degradation / Strategy / Team report views always use the Race session.
        # A wet/short race can legitimately have few or no clean dry-compound laps
        # (same case the dashboard shows a warning for) — assert these run without
        # crashing, not that they're non-empty.
        laps = data_mod.race_laps(year, gp)
        cl = clean_laps(laps)
        compound_model(cl)
        stint_degradation(cl)
        stops = pit_stops(laps)
        estimate_pit_loss(stops)
        assert not team_report(laps).empty


def test_openf1_session_maps_to_expected_shapes(monkeypatch):
    """OpenF1Session must map raw OpenF1 JSON (laps/drivers/stints/pit/car_data/
    location) into the same DataFrame shapes get_lap()/lap_telemetry() expect,
    with no real HTTP calls."""
    from app import data as data_mod

    def fake_openf1_get(path, timeout=10, **params):
        if path == "sessions":
            return [{"session_key": 9999, "date_start": "2026-01-01T13:00:00+00:00",
                     "location": "Testville", "circuit_short_name": "testville",
                     "country_name": "Testland", "session_name": "Race"}]
        if path == "drivers":
            return [{"driver_number": 1, "name_acronym": "VER", "full_name": "Max Verstappen",
                     "team_name": "Red Bull Racing"},
                    {"driver_number": 44, "name_acronym": "HAM", "full_name": "Lewis Hamilton",
                     "team_name": "Ferrari"}]
        if path == "laps":
            return [
                {"driver_number": 1, "lap_number": 1, "lap_duration": 90.5, "duration_sector_1": 30.1,
                 "duration_sector_2": 30.2, "duration_sector_3": 30.2, "st_speed": 320.0,
                 "is_pit_out_lap": False, "date_start": "2026-01-01T13:01:00+00:00"},
                {"driver_number": 44, "lap_number": 1, "lap_duration": 91.0, "duration_sector_1": 30.5,
                 "duration_sector_2": 30.3, "duration_sector_3": 30.2, "st_speed": 315.0,
                 "is_pit_out_lap": False, "date_start": "2026-01-01T13:01:05+00:00"},
            ]
        if path == "stints":
            return [{"driver_number": 1, "stint_number": 1, "lap_start": 1, "lap_end": 1,
                     "compound": "medium", "tyre_age_at_start": 0},
                    {"driver_number": 44, "stint_number": 1, "lap_start": 1, "lap_end": 1,
                     "compound": "soft", "tyre_age_at_start": 2}]
        if path == "pit":
            return []
        if path == "car_data":
            return [
                {"date": "2026-01-01T13:01:00+00:00", "speed": 300.0, "throttle": 100.0, "brake": 0},
                {"date": "2026-01-01T13:01:01+00:00", "speed": 310.0, "throttle": 100.0, "brake": 0},
                {"date": "2026-01-01T13:01:02+00:00", "speed": 250.0, "throttle": 0.0, "brake": 100},
            ]
        if path == "location":
            return [
                {"date": "2026-01-01T13:01:00+00:00", "x": 0.0, "y": 0.0},
                {"date": "2026-01-01T13:01:01+00:00", "x": 10.0, "y": 5.0},
                {"date": "2026-01-01T13:01:02+00:00", "x": 20.0, "y": 8.0},
            ]
        raise AssertionError(f"unexpected OpenF1 path {path!r}")

    monkeypatch.setattr(data_mod, "_openf1_get", fake_openf1_get)

    s = data_mod.OpenF1Session(2026, "Testland Grand Prix", "R")
    assert set(s.laps["Driver"]) == {"VER", "HAM"}
    assert s.laps.loc[s.laps["Driver"] == "VER", "Team"].iloc[0] == "Red Bull Racing"
    assert s.laps.loc[s.laps["Driver"] == "VER", "Compound"].iloc[0] == "MEDIUM"
    assert s.laps.loc[s.laps["Driver"] == "HAM", "TyreLife"].iloc[0] == 2.0
    assert s._corners.empty  # OpenF1 has no circuit-map endpoint

    lap = data_mod.get_lap(s, "VER")
    tel = data_mod.lap_telemetry(lap)
    assert list(tel.columns) == ["Distance", "Speed", "Throttle", "Brake", "TimeS", "X", "Y"]
    assert tel["Brake"].dtype == bool
    assert bool(tel["Brake"].iloc[-1]) is True
    assert tel["Distance"].is_monotonic_increasing
    assert tel["X"].iloc[1] == 10.0


def test_get_session_source_order(monkeypatch):
    """get_session() must try prebuilt -> OpenF1 -> FastF1 in that order, stop
    at the first success, and raise SessionLoadError only if all three fail."""
    from app import data as data_mod

    monkeypatch.setattr(data_mod, "_resolve_prebuilt", lambda *a, **k: None)
    calls = []

    def fake_openf1_fails(year, gp, session):
        calls.append("openf1")
        raise data_mod.OpenF1Error("not found")

    def fake_load_session(year, gp, session, telemetry=True):
        calls.append("fastf1")
        return "FASTF1_SESSION"

    monkeypatch.setattr(data_mod, "OpenF1Session", fake_openf1_fails)
    monkeypatch.setattr(data_mod, "load_session", fake_load_session)
    assert data_mod.get_session(2026, "Nowhere Grand Prix", "R") == "FASTF1_SESSION"
    assert calls == ["openf1", "fastf1"]

    calls.clear()

    class FakeOpenF1:
        def __init__(self, year, gp, session):
            calls.append("openf1")

    monkeypatch.setattr(data_mod, "OpenF1Session", FakeOpenF1)
    result = data_mod.get_session(2026, "Nowhere Grand Prix", "R")
    assert isinstance(result, FakeOpenF1)
    assert calls == ["openf1"]  # OpenF1 succeeding must short-circuit FastF1

    def fake_load_session_fails(year, gp, session, telemetry=True):
        raise RuntimeError("network unreachable")

    monkeypatch.setattr(data_mod, "OpenF1Session", fake_openf1_fails)
    monkeypatch.setattr(data_mod, "load_session", fake_load_session_fails)
    with pytest.raises(data_mod.SessionLoadError):
        data_mod.get_session(2026, "Nowhere Grand Prix", "R")


class _FakeCarData:
    """Stand-in for fastf1.core.Telemetry: only needs add_distance()."""

    def __init__(self, df: pd.DataFrame):
        self._df = df

    def add_distance(self):
        df = self._df.copy()
        df["Distance"] = np.arange(len(df)) * 10.0
        return df


class _FakeLapBrokenTelemetry:
    """A live-FastF1-like Lap whose get_telemetry() fails the way 2026 Monaco
    Race's does (malformed position data), but get_car_data() still works."""

    def get(self, key, default=None):
        return default

    def get_telemetry(self):
        raise KeyError("None of ['Date'] are in the columns")

    def get_car_data(self):
        return _FakeCarData(pd.DataFrame({
            "Time": pd.to_timedelta([0, 1, 2, 3, 4], unit="s"),
            "Speed": [200.0, 150.0, 100.0, 150.0, 200.0],
            "Throttle": [100.0, 50.0, 0.0, 50.0, 100.0],
            "Brake": [False, True, True, False, False],
        }))


def test_lap_telemetry_falls_back_to_car_data_when_get_telemetry_fails():
    """Telemetry fallback: get_telemetry() failing (e.g. Monaco 2026 R's
    KeyError) must fall back to get_car_data().add_distance() -- Distance/
    Speed/Throttle/Brake/TimeS recovered, X/Y NaN (no position data)."""
    from app.data import has_position_data, lap_telemetry

    tel = lap_telemetry(_FakeLapBrokenTelemetry())
    assert {"Distance", "Speed", "Throttle", "Brake", "TimeS", "X", "Y"} <= set(tel.columns)
    assert tel["Brake"].dtype == bool
    assert tel["Distance"].is_monotonic_increasing
    assert not has_position_data(tel)


def test_corners_falls_back_through_all_tiers_to_speed_trace(monkeypatch):
    """corners() on a live session: this session's map fails, no previous
    season is available (prebuilt or live), so it must land on the speed-trace
    estimate rather than crash or return empty."""
    from app import data as data_mod

    class FakeEvent:
        year = 2099

        def __getitem__(self, key):
            return "Fake Grand Prix"

    class FakeLaps:
        def pick_fastest(self):
            return _FakeLapBrokenTelemetry()

    class FakeSession:
        event = FakeEvent()
        name = "Race"
        laps = FakeLaps()

        def get_circuit_info(self):
            raise AttributeError("no map published for this circuit_key yet")

    monkeypatch.setattr(data_mod, "_resolve_prebuilt", lambda *a, **k: None)
    calls = []

    def fake_get_session(*a, **k):
        calls.append(a)
        raise RuntimeError("simulated: no such session last year either")

    monkeypatch.setattr(data_mod.fastf1, "get_session", fake_get_session)

    cn = data_mod.corners(FakeSession())
    assert calls, "should have attempted the previous-season live lookup before giving up"
    assert not cn.empty
    assert cn["Estimated"].all()


class _FakeLiveLaps:
    """Stand-in for fastf1.core.Laps: only the bits get_lap() touches."""

    def __init__(self, df: pd.DataFrame):
        self._df = df

    @property
    def empty(self):
        return self._df.empty

    def pick_fastest(self, only_by_time: bool = False):
        df = self._df if only_by_time else self._df[self._df["IsPersonalBest"] == True]  # noqa: E712
        valid = df[df["LapTime"].notna()]
        if valid.empty:
            return None
        return valid.loc[valid["LapTime"].idxmin()]

    def pick_drivers(self, driver):
        return _FakeLiveLaps(self._df[self._df["Driver"] == driver])


def test_get_lap_falls_back_to_lowest_laptime_when_no_personal_best():
    """No lap flagged personal-best (e.g. all deleted for track limits, the
    real VER-at-Monaco-2026 case) must fall back to the lowest non-null
    LapTime instead of raising."""
    from app.data import get_lap

    class FakeSession:
        laps = _FakeLiveLaps(pd.DataFrame({
            "Driver": ["VER", "VER"],
            "LapNumber": [1, 2],
            "LapTime": [pd.Timedelta(seconds=95.0), pd.Timedelta(seconds=90.0)],
            "IsPersonalBest": [False, False],
        }))

    lap = get_lap(FakeSession(), "VER", "fastest")
    assert lap["LapNumber"] == 2


def test_session_missing_reads_manifest_status(monkeypatch):
    """Dashboard gating: session_missing() must surface a partial session's
    missing fields so the UI can show a friendly message instead of crashing
    on lap_telemetry(), and return [] for a complete or absent session."""
    from app import data as data_mod

    fake_manifest = [
        {"year": 2026, "name": "Azerbaijan Grand Prix", "sessions": ["R"],
         "session_status": {"R": {"source": "openf1", "status": "partial",
                                   "missing": ["telemetry", "corners"]}}},
        {"year": 2026, "name": "Italian Grand Prix", "sessions": ["R"],
         "session_status": {"R": {"source": "fastf1", "status": "complete", "missing": []}}},
    ]
    monkeypatch.setattr(data_mod, "_prebuilt_manifest", lambda: fake_manifest)

    assert data_mod.session_missing(2026, "Azerbaijan Grand Prix", "R") == ["telemetry", "corners"]
    assert data_mod.session_missing(2026, "Italian Grand Prix", "R") == []
    assert data_mod.session_missing(2026, "Nonexistent Grand Prix", "R") == []


def test_build_prebuilt_classify_missing_and_upgrade():
    """build_prebuilt.py's partial/complete classification and the
    --only-missing retry logic that upgrades a partial session once a source
    recovers full data."""
    from scripts.build_prebuilt import _classify_missing, _is_missing_or_partial

    assert _classify_missing(written=0, driver_count=22, corner_count=0) == ["telemetry", "corners"]
    assert _classify_missing(written=22, driver_count=22, corner_count=19) == []
    assert _classify_missing(written=0, driver_count=0, corner_count=19) == []  # no drivers to try at all

    absent = {"sessions": [], "session_status": {}}
    partial = {"sessions": ["R"], "session_status": {"R": {"status": "partial", "missing": ["telemetry"]}}}
    complete = {"sessions": ["R"], "session_status": {"R": {"status": "complete", "missing": []}}}

    assert _is_missing_or_partial("R", absent) is True
    assert _is_missing_or_partial("R", partial) is True
    assert _is_missing_or_partial("R", complete) is False  # upgraded -> no longer rebuilt
