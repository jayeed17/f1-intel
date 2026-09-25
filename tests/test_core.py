"""Offline tests on synthetic data. No network, no FastF1 downloads."""
import numpy as np
import pandas as pd

from app.analysis.braking import assign_corners, braking_zones, compare_corners
from app.analysis.delta import lap_delta, minisector_dominance
from app.analysis.pits import estimate_pit_loss, pit_stops
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
