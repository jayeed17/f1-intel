"""F1 Intel API. Run: uvicorn app.main:app --reload"""
from __future__ import annotations

import fastf1
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.analysis.braking import assign_corners, braking_zones, compare_corners
from app.analysis.delta import lap_delta, minisector_dominance
from app.analysis.pits import estimate_pit_loss, pit_stops
from app.analysis.team_report import team_report
from app.config import RACE_DATASET_PATH
from app.data import (DataError, clean_laps, corners, get_lap, has_position_data,
                      lap_telemetry, load_session, race_laps, race_prediction_features,
                      to_records)
from app.models import tyre_ml
from app.models.degradation import compound_model, stint_degradation
from app.models.race_predictor import ensure_trained, predict_race
from app.models.strategy import compare_actual, simulate

app = FastAPI(title="F1 Intel", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.exception_handler(DataError)
async def data_error(_, exc: DataError):
    return JSONResponse(status_code=404, content={"detail": str(exc)})


def _session(year: int, gp: str, session: str, telemetry: bool = True):
    try:
        return load_session(year, gp, session, telemetry)
    except DataError:
        raise
    except Exception as e:  # fastf1 raises ValueError etc. for unknown events
        raise HTTPException(404, f"Could not load {year} {gp} {session}: {e}") from e


def _race_laps(year: int, gp: str, session: str = "R") -> pd.DataFrame:
    """Raw laps for the given session. Reads from the Parquet cache for "R" sessions
    (see app.data.race_laps); other sessions always load live from FastF1."""
    try:
        if session == "R":
            return race_laps(year, gp)
        return pd.DataFrame(_session(year, gp, session, telemetry=False).laps)
    except DataError:
        raise
    except Exception as e:
        raise HTTPException(404, f"Could not load {year} {gp} {session}: {e}") from e


@app.get("/health")
def health():
    return {"ok": True, "tyre_model": tyre_ml.MODEL_PATH.exists()}


@app.get("/schedule/{year}")
def schedule(year: int):
    sch = fastf1.get_event_schedule(year, include_testing=False)
    return to_records(sch[["RoundNumber", "EventName", "Country", "Location", "EventDate"]])


@app.get("/{year}/{gp}/drivers")
def drivers(year: int, gp: str, session: str = "R"):
    s = _session(year, gp, session, telemetry=False)
    return to_records(s.results[["Abbreviation", "FullName", "TeamName", "Position", "GridPosition"]])


@app.get("/{year}/{gp}/braking/{driver}")
def braking(year: int, gp: str, driver: str, session: str = "R", lap: str = "fastest"):
    s = _session(year, gp, session)
    l = get_lap(s, driver, lap)
    zones = assign_corners(braking_zones(lap_telemetry(l)), corners(s))
    return {"driver": driver.upper(), "lap": int(l["LapNumber"]),
            "lap_time_s": l["LapTime"].total_seconds(), "zones": to_records(zones)}


@app.get("/{year}/{gp}/compare")
def compare(year: int, gp: str, a: str, b: str, session: str = "Q",
            lap_a: str = "fastest", lap_b: str = "fastest"):
    s = _session(year, gp, session)
    ta, tb = lap_telemetry(get_lap(s, a, lap_a)), lap_telemetry(get_lap(s, b, lap_b))
    d = lap_delta(ta, tb)
    return {"a": a.upper(), "b": b.upper(),
            "final_gap_s": float(d["delta_s"].iloc[-1]),
            "corners": to_records(compare_corners(ta, tb, corners(s))),
            "delta": to_records(d.iloc[::4])}  # thin for payload size


@app.get("/{year}/{gp}/dominance")
def dominance(year: int, gp: str, drivers: str = Query(..., description="comma list e.g. VER,LEC"),
              session: str = "Q", n: int = 25):
    s = _session(year, gp, session)
    tels = {d.upper(): lap_telemetry(get_lap(s, d)) for d in drivers.split(",")}
    if not all(has_position_data(t) for t in tels.values()):
        return {"track_map_available": False, "message": "Track map unavailable for this session",
                "minisectors_won": {}, "points": []}
    pts = minisector_dominance(tels, n)
    share = pts.groupby("Winner")["Minisector"].nunique().to_dict()
    return {"track_map_available": True, "minisectors_won": share, "points": to_records(pts.iloc[::3])}


@app.get("/{year}/{gp}/degradation")
def degradation(year: int, gp: str, session: str = "R"):
    laps = _race_laps(year, gp, session)
    cl = clean_laps(laps)
    return {"compound_model": compound_model(cl), "stints": to_records(stint_degradation(cl))}


@app.get("/{year}/{gp}/pits")
def pits(year: int, gp: str):
    laps = _race_laps(year, gp)
    st = pit_stops(laps)
    return {"estimated_pit_loss_s": estimate_pit_loss(st), "stops": to_records(st)}


@app.get("/{year}/{gp}/strategy")
def strategy(year: int, gp: str, max_stops: int = Query(2, ge=1, le=3), top: int = 10,
             pit_loss: float | None = None):
    laps = _race_laps(year, gp)
    model = compound_model(clean_laps(laps))
    if len(model) < 2:
        raise HTTPException(422, "Need at least two dry compounds with enough clean laps")
    loss = pit_loss if pit_loss is not None else estimate_pit_loss(pit_stops(laps))
    total = int(laps["LapNumber"].max())
    sims = simulate(total, model, loss, max_stops=max_stops, top=top)
    actual = compare_actual(laps, total, model, loss, float(sims["total_s"].iloc[0]))
    return {"total_laps": total, "pit_loss_s": loss, "compound_model": model,
            "optimal": to_records(sims), "actual_vs_optimal": to_records(actual)}


@app.get("/{year}/{gp}/team-report")
def team(year: int, gp: str, session: str = "R"):
    laps = _race_laps(year, gp, session)
    return to_records(team_report(laps))


@app.get("/predict/race/{year}/{gp}")
def predict_race_route(year: int, gp: str, circuit_id: str | None = None):
    if not RACE_DATASET_PATH.exists():
        raise HTTPException(503, "Race dataset not built yet -- run scripts/build_race_dataset.py")
    pos_pipe, pts_pipe = ensure_trained(RACE_DATASET_PATH)
    try:
        features = race_prediction_features(year, gp, circuit_id)
    except DataError:
        raise
    except Exception as e:  # noqa: BLE001 -- e.g. gp/year not found by any source
        raise HTTPException(404, f"Could not load qualifying for {year} {gp}: {e}") from e

    preds = predict_race(pos_pipe, pts_pipe, features)
    predicted_order = preds.sort_values("predicted_position")[
        ["driver", "team_id", "predicted_position", "points_probability"]]
    grid_order = preds.sort_values("grid")[["driver", "team_id", "grid"]]
    return {"predicted_order": to_records(predicted_order), "grid_order": to_records(grid_order)}


class TyreQuery(BaseModel):
    TyreLife: float
    LapNumber: float
    Compound: str
    Stint: float = 1
    TrackTemp: float | None = None
    AirTemp: float | None = None
    Circuit: str | None = None


@app.post("/predict/tyre")
def predict_tyre(rows: list[TyreQuery]):
    pipe = tyre_ml.load_model()
    if pipe is None:
        raise HTTPException(503, "Model not trained. Run: python -m scripts.train_tyre_model --year 2025")
    preds = tyre_ml.predict(pipe, [r.model_dump() for r in rows])
    return {"rel_lap_time_s": preds}
