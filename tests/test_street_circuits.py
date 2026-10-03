"""Offline tests for app.analysis.street_circuits -- synthetic DataFrames
only, no FastF1, no network."""
import pandas as pd

from app.analysis.street_circuits import (braking_summary_by_class, circuit_history_table,
                                          classify_circuit, sc_vsc_frequency_by_class,
                                          street_vs_permanent_metrics)


def test_classify_circuit():
    assert classify_circuit("monaco") == "street"
    assert classify_circuit("baku") == "street"
    assert classify_circuit("albert_park") == "hybrid_street"
    assert classify_circuit("miami") == "hybrid_street"
    assert classify_circuit("monza") == "permanent"  # not listed -> permanent default
    assert classify_circuit("nonexistent_circuit") == "permanent"
    assert classify_circuit(None) == "permanent"


def _race_rows(circuit_id: str, season: int, round_: int, grids_finishes: list[tuple[int, int, int]],
               dnf: list[int] | None = None) -> list[dict]:
    """grids_finishes: list of (driver_num, grid, finish). dnf: driver_nums that DNF'd."""
    dnf = dnf or []
    rows = []
    for i, (drv, grid, finish) in enumerate(grids_finishes):
        rows.append({
            "season": season, "round": round_, "circuit_id": circuit_id,
            "driver": f"D{drv}", "team_id": f"T{drv % 3}",
            "grid": grid, "target_finish_pos": finish, "dnf": 1 if drv in dnf else 0,
        })
    return rows


def test_street_vs_permanent_metrics_separates_classes_and_computes_real_numbers():
    rows = []
    # Monaco (street): grid order preserved exactly -> perfect Spearman, 0 avg gain, pole always converts.
    rows += _race_rows("monaco", 2024, 1, [(1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 4, 4)])
    rows += _race_rows("monaco", 2025, 1, [(1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 4, 4)])
    # Monza (permanent): lots of overtaking -> grid order fully reversed.
    rows += _race_rows("monza", 2024, 2, [(1, 1, 4), (2, 2, 3), (3, 3, 2), (4, 4, 1)])
    rows += _race_rows("monza", 2025, 2, [(1, 1, 4), (2, 2, 3), (3, 3, 2), (4, 4, 1)], dnf=[2])

    df = pd.DataFrame(rows)
    out = street_vs_permanent_metrics(df).set_index("circuit_class")

    assert out.loc["street", "n_races"] == 2
    assert out.loc["street", "grid_finish_spearman"] == 1.0
    assert out.loc["street", "avg_positions_gained"] == 0.0
    assert out.loc["street", "pole_conversion_rate"] == 1.0
    assert out.loc["street", "dnf_rate"] == 0.0

    assert out.loc["permanent", "n_races"] == 2
    assert out.loc["permanent", "grid_finish_spearman"] == -1.0
    assert out.loc["permanent", "pole_conversion_rate"] == 0.0  # pole-sitter always finishes last here
    assert out.loc["permanent", "dnf_rate"] == 1 / 8  # 1 DNF across 8 rows


def test_circuit_history_table_limits_to_last_n_editions_and_counts_podiums():
    rows = []
    # 4 editions at a street circuit; only the last 3 should count.
    rows += _race_rows("baku", 2022, 1, [(9, 1, 1), (2, 2, 2), (3, 3, 3)])            # excluded (oldest of 4)
    rows += _race_rows("baku", 2023, 1, [(1, 1, 1), (2, 2, 2), (3, 3, 3)])
    rows += _race_rows("baku", 2024, 1, [(1, 2, 1), (2, 1, 2), (3, 3, 3)], dnf=[3])    # pole (D2) didn't win
    rows += _race_rows("baku", 2025, 1, [(1, 1, 1), (2, 2, 2), (3, 3, 3)])
    # A permanent circuit, deliberately excluded from the default street-only table.
    rows += _race_rows("monza", 2025, 2, [(1, 1, 1), (2, 2, 2), (3, 3, 3)])

    df = pd.DataFrame(rows)
    out = circuit_history_table(df, n_editions=3).set_index("circuit_id")

    assert list(out.index) == ["baku"]  # monza (permanent) excluded by default
    assert out.loc["baku", "editions"] == 3
    assert "D9 (2022)" not in out.loc["baku", "winners"]  # the 4th-oldest edition is dropped
    assert out.loc["baku", "winners"] == "D1 (2023), D1 (2024), D1 (2025)"
    assert out.loc["baku", "pole_to_win_rate"] == round(2 / 3, 3)  # pole won 2023 and 2025, not 2024
    assert out.loc["baku", "avg_dnfs"] == round(1 / 3, 2)
    assert out.loc["baku", "top_driver"] == "D1"  # podium in all 3 counted editions


def test_sc_vsc_frequency_by_class():
    df = pd.DataFrame([
        {"circuit_id": "monaco", "had_sc_or_vsc": True},
        {"circuit_id": "monaco", "had_sc_or_vsc": False},
        {"circuit_id": "monza", "had_sc_or_vsc": False},
        {"circuit_id": "monza", "had_sc_or_vsc": False},
    ])
    out = sc_vsc_frequency_by_class(df).set_index("circuit_class")
    assert out.loc["street", "n_races"] == 2
    assert out.loc["street", "sc_vsc_rate"] == 0.5
    assert out.loc["permanent", "sc_vsc_rate"] == 0.0


def test_braking_summary_by_class():
    df = pd.DataFrame([
        {"circuit_id": "monaco", "n_zones": 14, "avg_decel_g": 4.5},
        {"circuit_id": "monaco", "n_zones": 16, "avg_decel_g": 5.5},
        {"circuit_id": "monza", "n_zones": 6, "avg_decel_g": 5.0},
    ])
    out = braking_summary_by_class(df).set_index("circuit_class")
    assert out.loc["street", "avg_zones_per_lap"] == 15.0
    assert out.loc["street", "avg_decel_g"] == 5.0
    assert out.loc["permanent", "avg_zones_per_lap"] == 6.0
