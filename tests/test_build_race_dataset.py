"""Offline tests for scripts/build_race_dataset.py's pure-pandas helpers.
No network, no FastF1 -- these operate on plain per-race DataFrames."""
import numpy as np
import pandas as pd

from scripts.build_race_dataset import _add_within_race_features


def _race_frame(grids: list[float]) -> pd.DataFrame:
    n = len(grids)
    return pd.DataFrame({
        "driver": [f"D{i}" for i in range(n)],
        "team_id": [f"T{i}" for i in range(n)],  # distinct teams: no teammate-gap pairing noise
        "grid": grids,
        "quali_best_s": np.linspace(80.0, 82.0, n),
        "laps_completed": [50] * n,
        "race_max_laps": [50] * n,
    })


def test_pit_lane_start_uses_actual_field_size_not_max_plus_one():
    """Regression test: two pit-lane starters (grid==0) in a 5-car field
    whose other three drivers have non-contiguous raw grid numbers (1, 2, 5
    -- simulating real data, where grid numbers aren't renumbered just
    because someone else also starts from the pit lane). The old
    max(other grids) + 1 logic produced grid=6 in this 5-car field; the
    fix must produce grid=5 (the real number of starters)."""
    df = _add_within_race_features(_race_frame([1, 2, 5, 0, 0]))
    assert df["grid_pit_lane"].tolist() == [False, False, False, True, True]
    n_starters = len(df)
    assert (df.loc[df["grid_pit_lane"], "grid"] == n_starters).all()
    assert n_starters == 5  # sanity: this is exactly the overshoot scenario


def test_pit_lane_single_starter():
    df = _add_within_race_features(_race_frame([1, 2, 3, 4, 0]))
    assert df.loc[df["grid_pit_lane"], "grid"].iloc[0] == 5


def test_no_pit_lane_starters_is_a_no_op():
    df = _add_within_race_features(_race_frame([1, 2, 3, 4, 5]))
    assert not df["grid_pit_lane"].any()
    assert df["grid"].tolist() == [1, 2, 3, 4, 5]
