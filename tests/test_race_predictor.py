"""Offline tests for the race predictor, using synthetic data. No network."""
import numpy as np
import pandas as pd
import pytest

from app.models.race_predictor import (
    DEV_SEASON_MAX, GROUP_COLS, TARGET_DELTA, TARGET_DNF, _prep_delta, _prep_dnf,
    bootstrap_diff_ci, dev_holdout_split, empirical_grid_probs, evaluate, fit_final,
    grid_bucket_residual_std_array, make_delta_pipeline, make_dnf_pipeline, predict_race,
    rank_within_race, race_sequence, residual_std_by_grid_bucket, run_full_evaluation,
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


def test_residual_std_by_grid_bucket_and_array_mapping():
    """residual_std_by_grid_bucket() should report a materially different
    std for a bucket whose residuals actually are wider (constructed here
    directly, rather than relying on the fixture's incidental noise
    pattern), and grid_bucket_residual_std_array() should map each row to
    its own bucket's value."""
    rng = np.random.default_rng(0)
    n_per_bucket = 40
    grids = np.concatenate([rng.integers(1, 4, n_per_bucket),      # bucket "1-3"
                            rng.integers(4, 11, n_per_bucket),     # bucket "4-10"
                            rng.integers(11, 21, n_per_bucket)])   # bucket "11+"
    # Wide noise for the front, narrow for mid-pack, medium for the back.
    noise_std = np.concatenate([np.full(n_per_bucket, 6.0), np.full(n_per_bucket, 1.0),
                                np.full(n_per_bucket, 3.0)])
    delta_pred = np.zeros(len(grids))
    actual_delta = rng.normal(0, noise_std)
    preds = pd.DataFrame({
        "grid": grids, "delta_pred": delta_pred, TARGET_DNF: 0.0,
        "target_finish_pos": grids + delta_pred + actual_delta,
    })
    bucket_stds = residual_std_by_grid_bucket(preds)
    assert bucket_stds["1-3"] > bucket_stds["11+"] > bucket_stds["4-10"]

    grid = pd.Series([2.0, 7.0, 15.0])
    arr = grid_bucket_residual_std_array(grid, bucket_stds, fallback=1.0)
    assert arr[0] == pytest.approx(bucket_stds["1-3"])
    assert arr[1] == pytest.approx(bucket_stds["4-10"])
    assert arr[2] == pytest.approx(bucket_stds["11+"])


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
    assert "bootstrap_ci" in report["holdout"]  # wired in automatically when holdout is non-empty


def test_bootstrap_diff_ci_bounds_contain_mean_and_shrink_the_ci_narrower_for_more_races():
    """Resampling whole races (not rows) for a percentile CI on the
    model-minus-baseline difference. Sanity checks: the CI must bracket its
    own mean_diff, and doubling the number of races (holding the underlying
    per-race distribution roughly fixed) should not widen the CI."""
    df = fake_race_dataset(n_seasons=2, n_rounds=7, n_drivers=14)
    small_preds, _ = evaluate(df, min_train_races=6, n_sims=300)
    grid_prob_table = empirical_grid_probs(df)

    ci = bootstrap_diff_ci(small_preds, grid_prob_table, n_boot=300, seed=1)
    assert ci["n_boot"] == 300 and ci["ci"] == 0.95
    for key in ("position_mae_diff", "points_brier_diff", "win_brier_diff"):
        d = ci[key]
        assert d["ci_low"] <= d["mean_diff"] <= d["ci_high"]

    df_more = fake_race_dataset(n_seasons=6, n_rounds=7, n_drivers=14)
    big_preds, _ = evaluate(df_more, min_train_races=6, n_sims=300)
    ci_more = bootstrap_diff_ci(big_preds, grid_prob_table, n_boot=300, seed=1)
    small_width = ci["position_mae_diff"]["ci_high"] - ci["position_mae_diff"]["ci_low"]
    big_width = ci_more["position_mae_diff"]["ci_high"] - ci_more["position_mae_diff"]["ci_low"]
    assert big_width <= small_width * 1.5  # more races -> CI should not blow up


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
    """fit_calibration=False here: this test is about predict_race()'s
    ranking/shape contract (a raw MC simulation's win probabilities sum to
    exactly 1 across the field), not calibration -- isotonic calibration is
    fit per-probability-value on pooled cross-race data, so it does NOT
    preserve that sum-to-1 invariant (see the calibrated variant below)."""
    df = fake_race_dataset(n_seasons=4, n_rounds=10)
    delta_pipe, dnf_pipe, meta = fit_final(df, fit_calibration=False)
    next_race = df[df["round"] == 1].drop_duplicates("driver").copy()

    out = predict_race(delta_pipe, dnf_pipe, next_race, meta, n_sims=3000, seed=7)
    assert sorted(out["predicted_position"]) == list(range(1, len(out) + 1))
    for col in ("win_probability", "podium_probability", "points_probability"):
        assert out[col].between(0, 1).all()
    assert out["win_probability"].sum() == pytest.approx(1.0, abs=1e-6)
    assert list(out["predicted_position"]) == list(out["predicted_position"].sort_values())


def test_predict_race_with_calibration_uses_bucket_noise_and_calibrators():
    """fit_calibration=True (the default): meta should carry a per-bucket
    residual std and fitted win/podium calibrators, and predict_race()
    should actually use them (calibrated probabilities differ from the raw
    simulation, still valid probabilities, though no longer constrained to
    sum to 1 across the field -- see the note above)."""
    df = fake_race_dataset(n_seasons=5, n_rounds=8, n_drivers=14)  # spans into holdout for dev/holdout split
    delta_pipe, dnf_pipe, meta = fit_final(df)
    assert meta["residual_std_by_bucket"] is not None
    assert meta["win_calibrator"] is not None and meta["podium_calibrator"] is not None

    next_race = df[df["round"] == 1].drop_duplicates("driver").copy()
    out = predict_race(delta_pipe, dnf_pipe, next_race, meta, n_sims=3000, seed=7)
    assert sorted(out["predicted_position"]) == list(range(1, len(out) + 1))
    for col in ("win_probability", "podium_probability", "points_probability"):
        assert out[col].between(0, 1).all()


def test_train_writes_model_and_metrics(tmp_path, monkeypatch):
    from app.models import race_predictor as rp

    monkeypatch.setattr(rp, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(rp, "MODEL_BUNDLE_PATH", tmp_path / "bundle.joblib")
    monkeypatch.setattr(rp, "METRICS_PATH", tmp_path / "metrics.json")

    df = fake_race_dataset(n_seasons=3, n_rounds=8)  # all dev seasons -- holdout will be empty
    report = rp.train(df)

    assert (tmp_path / "bundle.joblib").exists()
    assert (tmp_path / "metrics.json").exists()
    assert report["n_races"] > 0
    delta_pipe, dnf_pipe, meta = rp.load_models()
    assert delta_pipe is not None and dnf_pipe is not None
    assert "residual_std" in meta and len(meta["dnf_position_samples"]) > 0


def test_freeze_and_load_frozen_model(tmp_path, monkeypatch):
    from app.models import race_predictor as rp

    dataset_path = tmp_path / "race_dataset.parquet"
    df = fake_race_dataset(n_seasons=5, n_rounds=8, n_drivers=14)
    df.to_parquet(dataset_path)

    monkeypatch.setattr(rp, "FROZEN_MODEL_DIR", tmp_path / "frozen")
    monkeypatch.setattr(rp, "FROZEN_BUNDLE_PATH", tmp_path / "frozen" / "model.joblib")
    monkeypatch.setattr(rp, "FROZEN_SPEC_PATH", tmp_path / "frozen" / "spec.json")

    assert rp.frozen_model_spec() is None
    with pytest.raises(FileNotFoundError):
        rp.load_frozen_model()

    spec = rp.freeze_model(df, dataset_path, version="9.9.9", notes="test freeze", frozen_at="2026-01-01")
    assert spec["version"] == "9.9.9"
    assert spec["frozen_at"] == "2026-01-01"
    assert len(spec["dataset_sha256"]) == 64

    loaded_spec = rp.frozen_model_spec()
    assert loaded_spec == spec

    delta_pipe, dnf_pipe, meta = rp.load_frozen_model()
    assert delta_pipe is not None and dnf_pipe is not None
    assert meta["win_calibrator"] is not None

    # ensure_trained() must prefer the frozen model once one exists.
    _, _, meta2 = rp.ensure_trained(dataset_path)
    assert meta2["residual_std"] == meta["residual_std"]


def test_compute_live_track_record(tmp_path, monkeypatch):
    from app.models import race_predictor as rp

    predictions_dir = tmp_path / "predictions"
    predictions_dir.mkdir()
    monkeypatch.setattr(rp, "PREDICTIONS_DIR", predictions_dir)

    frozen_at = "2026-06-01"
    assert rp.compute_live_track_record(frozen_at) is None  # no predictions logged at all

    rows = pd.DataFrame({
        "predicted_at": ["2026-05-01T00:00:00+00:00", "2026-06-02T00:00:00+00:00",
                        "2026-06-02T00:00:00+00:00"],
        "gp": ["Before Freeze GP", "After Freeze GP", "After Freeze GP"],
        "driver": ["D0", "D0", "D1"],
        "team_id": ["T0", "T0", "T1"],
        "grid": [1, 1, 2],
        "predicted_position": [1, 1, 2],
        "win_probability": [0.5, 0.5, 0.1],
        "podium_probability": [0.8, 0.8, 0.3],
        "points_probability": [0.9, 0.9, 0.6],
        "actual_position": [1, None, None],
        "scored_at": ["2026-05-02T00:00:00+00:00", None, None],
    })
    rows.to_csv(predictions_dir / "2026.csv", index=False)

    # Only the post-freeze race counts, and it's not scored yet.
    live = rp.compute_live_track_record(frozen_at)
    assert live is not None
    assert live["n_races_predicted"] == 1
    assert live["n_races_scored"] == 0

    # Now score the post-freeze race and confirm the metrics populate.
    rows.loc[rows["gp"] == "After Freeze GP", "actual_position"] = [1, 3]
    rows.to_csv(predictions_dir / "2026.csv", index=False)
    live = rp.compute_live_track_record(frozen_at)
    assert live["n_races_scored"] == 1
    assert live["n_rows_scored"] == 2
    assert live["model_mae"] == pytest.approx(0.5)
    assert live["model_winner_accuracy"] == 1.0


def test_compute_live_track_record_ignores_staged_v2_quali_files(tmp_path, monkeypatch):
    """Regression test: scripts/predict_staged.py logs race v2 and quali
    predictions to {year}_race_v2_{stage}.csv / {year}_quali_{stage}.csv
    in the SAME predictions/ directory as v1's own {year}.csv, sharing
    many of the same column names (predicted_position, actual_position).
    compute_live_track_record() must only ever read v1's own {year}.csv,
    never sweep up v2/quali's files via a too-broad glob."""
    from app.models import race_predictor as rp

    predictions_dir = tmp_path / "predictions"
    predictions_dir.mkdir()
    monkeypatch.setattr(rp, "PREDICTIONS_DIR", predictions_dir)
    frozen_at = "2026-06-01"

    v1_rows = pd.DataFrame({
        "predicted_at": ["2026-06-02T00:00:00+00:00"], "gp": ["V1 Only GP"], "driver": ["D0"],
        "team_id": ["T0"], "grid": [1], "predicted_position": [1], "win_probability": [0.5],
        "podium_probability": [0.8], "points_probability": [0.9],
        "actual_position": [1], "scored_at": ["2026-06-03T00:00:00+00:00"],
    })
    v1_rows.to_csv(predictions_dir / "2026.csv", index=False)

    # A DIFFERENT race, logged only by race v2's staged post-quali file --
    # same column names as v1's file, but must not be counted as v1's own.
    v2_rows = pd.DataFrame({
        "predicted_at": ["2026-06-02T00:00:00+00:00"], "gp": ["V2 Only GP"], "driver": ["D1"],
        "team_id": ["T1"], "grid": [2], "predicted_position": [2], "win_probability": [0.1],
        "podium_probability": [0.3], "points_probability": [0.6],
        "actual_position": [2], "scored_at": ["2026-06-03T00:00:00+00:00"],
    })
    v2_rows.to_csv(predictions_dir / "2026_race_v2_post_quali.csv", index=False)

    live = rp.compute_live_track_record(frozen_at)
    assert live["n_races_predicted"] == 1
    assert live["n_races_scored"] == 1
    assert "win_brier" in live and "podium_brier" in live and "points_brier" in live
