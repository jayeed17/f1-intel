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
    calls = {"n": 0}

    class FakeSession:
        laps = fake_race()

    def fake_load_session(year, gp, session, telemetry=True):
        calls["n"] += 1
        return FakeSession()

    monkeypatch.setattr(data_mod, "load_session", fake_load_session)

    df1 = data_mod.race_laps(2025, 1)
    assert calls["n"] == 1
    assert (tmp_path / "2025" / "1.parquet").exists()

    df2 = data_mod.race_laps(2025, 1)
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
