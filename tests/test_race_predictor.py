"""Offline tests for the race predictor, using synthetic data. No network."""
import numpy as np
import pandas as pd
import pytest

from app.models.race_predictor import (
    DEV_SEASON_MAX, GROUP_COLS, TARGET_DELTA, TARGET_DNF, _prep_delta, _prep_dnf,
    dev_holdout_split, empirical_grid_probs, evaluate, fit_final, make_delta_pipeline,
    make_dnf_pipeline, predict_race, rank_within_race, race_sequence, run_full_evaluation,
    select_dnf_hyperparams, select_hyperparams, simulate_positions, time_based_splits,
)


def fake_race_dataset(n_seasons=3, n_rounds=8, n_drivers=14, seed=0, dnf_rate=0.12) -> pd.DataFrame:
    """Synthetic one-row-per-driver-per-race dataset matching
    build_race_dataset.py's schema, with a real (noisy) relationship between
    grid and finish position so the model has something to learn, plus a
    randomised DNF flag (biased toward nothing in particular -- just enough
    class variety for the DNF classifier to train on)."""
    rng = np.random.default_rng(seed)
    rows = []
    drivers = [f"D{i}" for i in range(n_drivers)]
    teams = {d: f"T{i // 2}" for i, d in enumerate(drivers)}
    for season in range(2022, 2022 + n_seasons):
        for rnd in range(1, n_rounds + 1):
            grid = rng.permutation(np.arange(1, n_drivers + 1))
            finish_raw = grid + rng.normal(0, 1.5, n_drivers)
            finish = pd.Series(finish_raw).rank(method="first").astype(int).to_numpy()
            dnf = (rng.random(n_drivers) < dnf_rate).astype(float)
            for i, drv in enumerate(drivers):
                rows.append({
                    "season": season, "round": rnd, "event_name": f"Race {rnd}",
                    "circuit_id": "test", "driver": drv, "team_id": teams[drv],
                    "grid": int(grid[i]), "grid_pit_lane": False,
                    "quali_position": int(grid[i]), "quali_gap_to_pole_s": float(grid[i]) * 0.1,
                    "teammate_quali_gap_s": 0.0,
                    "driver_rolling_avg_finish_3": np.nan, "driver_rolling_avg_finish_5": np.nan,
                    "team_rolling_avg_finish_3": np.nan, "team_rolling_pace_gap_3": np.nan,
                    "driver_dnf_rate_10": 0.0, "team_dnf_rate_10": 0.0,
                    "circuit_type": "mixed", "reg_change_flag": False,
                    "dnf": float(dnf[i]), "source": "test",
                    "target_finish_pos": int(finish[i]), "target_points_top10": bool(finish[i] <= 10),
                    "target_delta": int(finish[i]) - int(grid[i]),
                })
    return pd.DataFrame(rows)


def test_time_based_splits_never_trains_on_the_future():
    """The leakage test: every training row's race must be strictly before
    every test row's race -- this must fail if that ever stops holding."""
    df = fake_race_dataset()
    seq = pd.Series(race_sequence(df), index=df.index)
    checked = 0
    for train_idx, test_idx in time_based_splits(df, min_train_races=5):
        assert seq.loc[train_idx].max() < seq.loc[test_idx].min()
        checked += 1
    assert checked > 0


def test_time_based_splits_expanding_window():
    """The training set must only ever grow -- an expanding window, not a
    sliding one -- across successive folds."""
    df = fake_race_dataset()
    sizes = [len(tr) for tr, _ in time_based_splits(df, min_train_races=5)]
    assert sizes == sorted(sizes)


def test_rank_within_race_unique_and_handles_nan():
    df = pd.DataFrame({
        "season": [1, 1, 1, 2, 2],
        "round": [1, 1, 1, 1, 1],
        "score": [3.0, 1.0, np.nan, 2.0, 2.0],  # race 1 has a NaN; race 2 a tie
    })
    out = rank_within_race(df, "score", "rank")
    for _, g in out.groupby(["season", "round"]):
        assert sorted(g["rank"]) == list(range(1, len(g) + 1))
    race1 = out[(out["season"] == 1) & (out["round"] == 1)]
    assert race1.loc[race1["score"].isna(), "rank"].iloc[0] == race1["rank"].max()


def test_dev_holdout_split():
    df = fake_race_dataset(n_seasons=5, n_rounds=6, n_drivers=14)  # 2022-2026
    dev, holdout = dev_holdout_split(df)
    assert dev["season"].max() <= DEV_SEASON_MAX
    assert holdout["season"].min() > DEV_SEASON_MAX
    assert len(dev) + len(holdout) == len(df)
    assert not dev.empty and not holdout.empty


def test_select_hyperparams_refuses_holdout_rows():
    """The required regression test: hyperparameter selection must never see
    a holdout-season (2025+) row. select_hyperparams/select_dnf_hyperparams
    raise immediately if handed one -- this is a structural guarantee, not
    just a documented convention, so it can't be silently bypassed by a
    future caller forgetting to filter first."""
    df = fake_race_dataset(n_seasons=5, n_rounds=6, n_drivers=14)  # spans into 2025-2026
    dev, holdout = dev_holdout_split(df)
    assert not holdout.empty  # sanity: the fixture actually spans into holdout

    with pytest.raises(ValueError, match="dev seasons"):
        select_hyperparams(df, min_train_races=6, n_sims=50)
    with pytest.raises(ValueError, match="dev seasons"):
        select_hyperparams(holdout, min_train_races=2, n_sims=50)
    with pytest.raises(ValueError, match="dev seasons"):
        select_dnf_hyperparams(df, min_train_races=6, n_sims=50)

    # dev-only calls must succeed and return one of the grid's candidates.
    grid = [{"max_leaf_nodes": 7, "min_samples_leaf": 20, "learning_rate": 0.1,
            "max_iter": 100, "l2_regularization": 1.0}]
    chosen = select_hyperparams(dev, grid=grid, min_train_races=6, n_sims=50)
    assert chosen in grid
    chosen_dnf = select_dnf_hyperparams(dev, grid=grid, min_train_races=6, n_sims=50)
    assert chosen_dnf in grid


def test_empirical_grid_probs_bounded_and_present():
    df = fake_race_dataset(n_seasons=3, n_rounds=8, n_drivers=14)
    dev, _ = dev_holdout_split(df)
    table = empirical_grid_probs(dev)
    assert not table.empty
    for col in ("p_win", "p_podium", "p_points"):
        assert table[col].between(0, 1).all()
    # grid correlates with finish in the fixture -- pole should win more
    # often than the back of the grid.
    pole_row = table.loc[table["grid_bucket"] == table["grid_bucket"].min()].iloc[0]
    back_row = table.loc[table["grid_bucket"] == table["grid_bucket"].max()].iloc[0]
    assert pole_row["p_win"] >= back_row["p_win"]


def test_simulate_positions_produces_valid_probabilities():
    n = 10
    grid = pd.Series(np.arange(1, n + 1, dtype=float))
    delta_pred = pd.Series(np.zeros(n))
    p_dnf = pd.Series(np.full(n, 0.1))
    dnf_samples = np.array([n + 1.0])
    sim = simulate_positions(grid, delta_pred, p_dnf, dnf_samples, residual_std=2.0,
                             n_sims=5000, seed=1)
    for key in ("p_win", "p_podium", "p_points"):
        assert (sim[key] >= 0).all() and (sim[key] <= 1).all()
    # exactly one winner/podium-of-3/points-of-min(10,n) per simulated run.
    assert sim["p_win"].sum() == pytest.approx(1.0, abs=1e-9)
    assert sim["p_podium"].sum() == pytest.approx(3.0, abs=1e-9)
    assert sim["p_points"].sum() == pytest.approx(min(10, n), abs=1e-9)
    assert (sim["expected_position"] >= 1).all() and (sim["expected_position"] <= n).all()
    # pole (grid=1, zero delta) should win more often than the back marker.
    assert sim["p_win"][0] > sim["p_win"][-1]


def test_evaluate_runs_and_produces_valid_metrics():
    """Not asserting the model beats the baselines -- see race_predictor.py's
    module docstring / the project's honest write-up for that. Just that
    evaluate() runs end to end and every metric is a plausible number."""
    df = fake_race_dataset(n_seasons=2, n_rounds=7)
    preds, meta = evaluate(df, min_train_races=6, n_sims=300)
    assert meta["residual_std"] > 0
    for _, race in preds.groupby(GROUP_COLS):
        assert sorted(race["model_pos"]) == list(range(1, len(race) + 1))
        assert race["p_win"].sum() == pytest.approx(1.0, abs=1e-6)

    grid_prob_table = empirical_grid_probs(df)
    from app.models.race_predictor import summarise
    summary = summarise(preds, grid_prob_table)
    for method in ("model", "baseline_grid", "baseline_quali"):
        m = summary["point_metrics"][method]
        assert m["position_mae"] > 0
        assert -1 <= m["spearman"] <= 1
        assert 0 <= m["top3_hit_rate"] <= 1
        assert 0 <= m["winner_accuracy"] <= 1
    for method in ("model", "baseline_grid_prob"):
        for outcome in ("win", "podium", "points"):
            pm = summary["prob_metrics"][method][outcome]
            assert pm["brier"] is None or 0 <= pm["brier"] <= 1
            assert pm["logloss"] is None or pm["logloss"] >= 0


def test_run_full_evaluation_splits_dev_and_holdout():
    # n_rounds=8 > min_train_races=6 so season 2022 itself has evaluated
    # races too, not just consumed entirely by the initial training window.
    df = fake_race_dataset(n_seasons=5, n_rounds=8, n_drivers=14)
    report = run_full_evaluation(df, min_train_races=6, n_sims=200)
    assert report["dev"]["n_races"] > 0
    assert report["holdout"]["n_races"] > 0
    assert set(report["by_season"].keys()) == {"2022", "2023", "2024", "2025", "2026"}
    assert report["residual_std"] > 0


def test_prep_handles_all_nan_numeric_column():
    """Regression test: an entirely-NaN numeric feature (e.g.
    team_rolling_pace_gap_3 / team_dnf_rate_10 in an early walk-forward fold
    trained only on seasons that predate it) must not crash
    HistGradientBoosting's binning. Confirmed live on sklearn 1.9.1: fitting
    on a column with 0 distinct non-null values raises "window shape cannot
    be larger than input array shape" inside sklearn's own
    _find_binning_thresholds. _prep() fills a fully-missing numeric column
    with 0.0 (zero variance either way, so this changes nothing
    predictively) specifically to avoid that crash."""
    df = fake_race_dataset(n_seasons=1, n_rounds=6, n_drivers=14)
    df["team_rolling_pace_gap_3"] = np.nan
    df["team_dnf_rate_10"] = np.nan

    X_delta = _prep_delta(df)
    assert X_delta["team_rolling_pace_gap_3"].notna().all()
    delta_pipe = make_delta_pipeline().fit(X_delta, df[TARGET_DELTA])
    delta_pipe.predict(X_delta)  # must not raise

    X_dnf = _prep_dnf(df)
    assert X_dnf["team_dnf_rate_10"].notna().all()
    dnf_pipe = make_dnf_pipeline().fit(X_dnf, df[TARGET_DNF])
    dnf_pipe.predict_proba(X_dnf)  # must not raise


def test_predict_race_ranks_single_race_and_probabilities():
    df = fake_race_dataset(n_seasons=4, n_rounds=10)
    delta_pipe, dnf_pipe, meta = fit_final(df)
    next_race = df[df["round"] == 1].drop_duplicates("driver").copy()

    out = predict_race(delta_pipe, dnf_pipe, next_race, meta, n_sims=3000, seed=7)
    assert sorted(out["predicted_position"]) == list(range(1, len(out) + 1))
    for col in ("win_probability", "podium_probability", "points_probability"):
        assert out[col].between(0, 1).all()
    assert out["win_probability"].sum() == pytest.approx(1.0, abs=1e-6)
    assert list(out["predicted_position"]) == list(out["predicted_position"].sort_values())


def test_train_writes_model_and_metrics(tmp_path, monkeypatch):
    from app.models import race_predictor as rp

    monkeypatch.setattr(rp, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(rp, "MODEL_DELTA_PATH", tmp_path / "delta.joblib")
    monkeypatch.setattr(rp, "MODEL_DNF_PATH", tmp_path / "dnf.joblib")
    monkeypatch.setattr(rp, "META_PATH", tmp_path / "meta.json")
    monkeypatch.setattr(rp, "METRICS_PATH", tmp_path / "metrics.json")

    df = fake_race_dataset(n_seasons=3, n_rounds=8)  # all dev seasons -- holdout will be empty
    report = rp.train(df)

    assert (tmp_path / "delta.joblib").exists()
    assert (tmp_path / "dnf.joblib").exists()
    assert (tmp_path / "meta.json").exists()
    assert (tmp_path / "metrics.json").exists()
    assert report["n_races"] > 0
    delta_pipe, dnf_pipe, meta = rp.load_models()
    assert delta_pipe is not None and dnf_pipe is not None
    assert "residual_std" in meta and len(meta["dnf_position_samples"]) > 0
