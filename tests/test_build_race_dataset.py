"""Offline tests for scripts/build_race_dataset.py's pure-pandas helpers.
No network, no FastF1 -- these operate on plain per-race DataFrames."""
import numpy as np
import pandas as pd

from scripts.build_race_dataset import (_add_within_race_features,
                                        add_circuit_history_and_quali_rolling_features)


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


def _circuit_history_frame(rows: list[dict]) -> pd.DataFrame:
    """One row per driver per race, the columns
    add_circuit_history_and_quali_rolling_features() needs. `rows` entries
    may omit team_id (defaults to one teammate-less team per driver) --
    keeps leakage-test fixtures terse."""
    out = []
    for r in rows:
        out.append({"team_id": r.get("team_id", f"T_{r['driver']}"),
                   "teammate_quali_gap_s": r.get("teammate_quali_gap_s", 0.0), **r})
    return pd.DataFrame(out)


def test_circuit_history_never_uses_the_same_or_a_later_season():
    """D1 races at circuit "foo" every season 2022-2025 with strictly
    increasing quali positions (2022->1, 2023->2, 2024->3, 2025->4). The
    2025 row's history must be built ONLY from 2022-2024 -- never 2025's
    own value, and never anything that would require the future."""
    df = _circuit_history_frame([
        {"driver": "D1", "circuit_id": "foo", "season": s, "round": 1,
        "quali_position": s - 2021, "target_finish_pos": s - 2021}
        for s in (2022, 2023, 2024, 2025)
    ])
    out = add_circuit_history_and_quali_rolling_features(df)
    row_2025 = out[out["season"] == 2025].iloc[0]
    row_2024 = out[out["season"] == 2024].iloc[0]
    row_2023 = out[out["season"] == 2023].iloc[0]
    row_2022 = out[out["season"] == 2022].iloc[0]

    # 2025's "last" and "avg of last 3" must come from 2022-2024 only.
    assert row_2025["driver_circuit_last_quali"] == 3  # 2024's value, not 2025's own 4
    assert row_2025["driver_circuit_avg_quali_3"] == _mean([1, 2, 3])
    assert row_2025["driver_circuit_races_here"] == 3  # 3 prior editions

    # The very first edition has nothing before it: NaN history, not a
    # leaked peek at its own (or a future) value.
    assert pd.isna(row_2022["driver_circuit_last_quali"])
    assert pd.isna(row_2022["driver_circuit_avg_quali_3"])
    assert row_2022["driver_circuit_races_here"] == 0
    assert row_2022["circuit_new_or_changed"]

    # 2023's history must be exactly {2022} -- not 2023 itself, not 2024/2025.
    assert row_2023["driver_circuit_last_quali"] == 1
    assert row_2023["driver_circuit_races_here"] == 1


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def test_circuit_history_reset_blocks_old_layout_from_counting(monkeypatch):
    """CIRCUIT_HISTORY_RESET says circuit "foo" got a new layout in 2025 --
    the 2026 edition's history must NOT include the pre-reset 2022-2024
    rows (old layout), even though circuit_id is unchanged. The reset
    season itself (2025) gets circuit_new_or_changed=True and NaN history,
    same as a genuinely brand-new venue."""
    import scripts.build_race_dataset as brd
    monkeypatch.setitem(brd.CIRCUIT_HISTORY_RESET, "foo", 2025)

    df = _circuit_history_frame([
        {"driver": "D1", "circuit_id": "foo", "season": s, "round": 1,
        "quali_position": s - 2021, "target_finish_pos": s - 2021}
        for s in (2022, 2023, 2024, 2025, 2026)
    ])
    out = add_circuit_history_and_quali_rolling_features(df)
    row_2025 = out[out["season"] == 2025].iloc[0]
    row_2026 = out[out["season"] == 2026].iloc[0]

    assert row_2025["circuit_new_or_changed"]  # the reset season is a blank slate
    assert pd.isna(row_2025["driver_circuit_last_quali"])
    assert row_2025["driver_circuit_races_here"] == 0

    assert not row_2026["circuit_new_or_changed"]  # 2026 has one prior (post-reset) edition
    assert row_2026["driver_circuit_races_here"] == 1
    assert row_2026["driver_circuit_last_quali"] == 2025 - 2021  # only 2025 counts, not 2022-2024


def test_brand_new_circuit_gets_nan_history_not_borrowed_from_elsewhere():
    """A circuit appearing for the first time in the whole dataset: every
    driver racing there gets circuit_new_or_changed=True and NaN history,
    regardless of how much history they have at OTHER circuits."""
    df = _circuit_history_frame([
        {"driver": "D1", "circuit_id": "established", "season": 2022, "round": 1,
        "quali_position": 1, "target_finish_pos": 1},
        {"driver": "D1", "circuit_id": "established", "season": 2023, "round": 1,
        "quali_position": 2, "target_finish_pos": 2},
        {"driver": "D1", "circuit_id": "brand_new", "season": 2024, "round": 5,
        "quali_position": 5, "target_finish_pos": 5},
    ])
    out = add_circuit_history_and_quali_rolling_features(df)
    new_row = out[out["circuit_id"] == "brand_new"].iloc[0]
    assert new_row["circuit_new_or_changed"]
    assert pd.isna(new_row["driver_circuit_last_quali"])
    assert pd.isna(new_row["driver_circuit_avg_quali_3"])
    assert new_row["driver_circuit_races_here"] == 0
