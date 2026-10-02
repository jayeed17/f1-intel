"""Offline tests for the qualifying predictor / race predictor v2 feature
plumbing added alongside them: live-prediction circuit history (new/
changed-circuit handling) and the dashboard's stage selection. No
network, no FastF1 -- synthetic DataFrames and monkeypatched manifest
lookups only."""
import pandas as pd

import app.data as data_mod
from app.data import _circuit_history_snapshot_for_prediction, race_stage_for


def _dataset_frame(rows: list[dict]) -> pd.DataFrame:
    out = []
    for r in rows:
        out.append({"team_id": r.get("team_id", f"T_{r['driver']}"), **r})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
# New/changed-circuit handling (app.data._circuit_history_snapshot_for_prediction)
# --------------------------------------------------------------------------

def test_unresolved_circuit_id_gives_nan_history_not_a_crash():
    """circuit_id=None (can't be resolved live) -> NaN history + the
    new/changed flag, not an exception and not a borrowed value from
    some other venue."""
    dataset = _dataset_frame([
        {"driver": "D1", "circuit_id": "somewhere", "season": 2024, "round": 1,
        "quali_position": 3.0, "target_finish_pos": 3.0},
    ])
    snap = _circuit_history_snapshot_for_prediction(dataset, "D1", "T_D1", None, 2025)
    assert snap["circuit_new_or_changed"] is True
    assert pd.isna(snap["driver_circuit_avg_quali_3"])
    assert pd.isna(snap["team_circuit_avg_quali_3"])
    assert snap["driver_circuit_races_here"] == 0
    assert snap["team_circuit_races_here"] == 0


def test_brand_new_circuit_not_in_dataset_at_all():
    """circuit_id resolves, but no row anywhere in the dataset has ever
    used it (a genuinely new venue) -> same NaN-history, new/changed
    treatment, not a KeyError."""
    dataset = _dataset_frame([
        {"driver": "D1", "circuit_id": "somewhere_else", "season": 2024, "round": 1,
        "quali_position": 1.0, "target_finish_pos": 1.0},
    ])
    snap = _circuit_history_snapshot_for_prediction(dataset, "D1", "T_D1", "brand_new_venue", 2025)
    assert snap["circuit_new_or_changed"] is True
    assert pd.isna(snap["driver_circuit_avg_quali_3"])
    assert snap["driver_circuit_races_here"] == 0


def test_circuit_with_real_history_resolves_correctly():
    """Sanity check the live snapshot actually finds and averages real
    prior editions (strictly before `season`) when they exist, same
    values build_race_dataset.py's own leakage test expects."""
    dataset = _dataset_frame([
        {"driver": "D1", "circuit_id": "foo", "season": s, "round": 1,
        "quali_position": float(s - 2021), "target_finish_pos": float(s - 2021)}
        for s in (2022, 2023, 2024)
    ])
    snap = _circuit_history_snapshot_for_prediction(dataset, "D1", "T_D1", "foo", 2025)
    assert snap["circuit_new_or_changed"] is False
    assert snap["driver_circuit_races_here"] == 3
    assert snap["driver_circuit_last_quali"] == 3.0  # 2024's value
    assert snap["driver_circuit_avg_quali_3"] == (1.0 + 2.0 + 3.0) / 3


def test_circuit_history_reset_excludes_pre_reset_editions(monkeypatch):
    """Live snapshot honors CIRCUIT_HISTORY_RESET the same way the
    dataset builder does: asking about season 2026 at a circuit reset in
    2026 must not see 2022-2024's pre-reset rows."""
    monkeypatch.setitem(data_mod.CIRCUIT_HISTORY_RESET, "reset_circuit", 2026)
    dataset = _dataset_frame([
        {"driver": "D1", "circuit_id": "reset_circuit", "season": s, "round": 1,
        "quali_position": 1.0, "target_finish_pos": 1.0}
        for s in (2022, 2023, 2024)
    ])
    snap = _circuit_history_snapshot_for_prediction(dataset, "D1", "T_D1", "reset_circuit", 2026)
    assert snap["circuit_new_or_changed"] is True
    assert snap["driver_circuit_races_here"] == 0
    assert pd.isna(snap["driver_circuit_avg_quali_3"])


# --------------------------------------------------------------------------
# Dashboard stage selection (app.data.race_stage_for)
# --------------------------------------------------------------------------

def test_stage_forecast_when_nothing_built(monkeypatch):
    monkeypatch.setattr(data_mod, "prebuilt_sessions_for", lambda year, gp: [])
    assert race_stage_for(2026, "Nowhere Grand Prix") == "Forecast"


def test_stage_post_practice_when_only_fp_sessions_built(monkeypatch):
    monkeypatch.setattr(data_mod, "prebuilt_sessions_for", lambda year, gp: ["FP2", "FP3"])
    assert race_stage_for(2026, "Somewhere Grand Prix") == "Post-practice"


def test_stage_post_quali_when_q_built_but_not_r(monkeypatch):
    monkeypatch.setattr(data_mod, "prebuilt_sessions_for", lambda year, gp: ["FP2", "FP3", "Q"])
    assert race_stage_for(2026, "Somewhere Grand Prix") == "Post-quali"


def test_stage_result_when_race_built(monkeypatch):
    monkeypatch.setattr(data_mod, "prebuilt_sessions_for", lambda year, gp: ["FP2", "FP3", "Q", "R"])
    assert race_stage_for(2026, "Somewhere Grand Prix") == "Result"


def test_stage_result_takes_priority_even_with_partial_sessions(monkeypatch):
    """R present is always "Result", regardless of which other sessions
    also happen to be bundled (order of the checks in race_stage_for
    matters: R beats Q beats FP)."""
    monkeypatch.setattr(data_mod, "prebuilt_sessions_for", lambda year, gp: ["R"])
    assert race_stage_for(2026, "Somewhere Grand Prix") == "Result"
