"""Offline tests for the race predictor, using synthetic data. No network."""
import numpy as np
import pandas as pd

from app.models.race_predictor import (
    FEATURES, GROUP_COLS, TARGET_POS, TARGET_PTS,
    _prep, evaluate, make_points_pipeline, make_position_pipeline,
    predict_race, rank_within_race, race_sequence, time_based_splits,
)


def fake_race_dataset(n_seasons=3, n_rounds=8, n_drivers=14, seed=0) -> pd.DataFrame:
    """Synthetic one-row-per-driver-per-race dataset matching
    build_race_dataset.py's schema, with a real (noisy) relationship between
    grid and finish position so the model has something to learn."""
    rng = np.random.default_rng(seed)
    rows = []
    drivers = [f"D{i}" for i in range(n_drivers)]
    teams = {d: f"T{i // 2}" for i, d in enumerate(drivers)}
    for season in range(2022, 2022 + n_seasons):
        for rnd in range(1, n_rounds + 1):
            grid = rng.permutation(np.arange(1, n_drivers + 1))
            finish_raw = grid + rng.normal(0, 1.5, n_drivers)
            finish = pd.Series(finish_raw).rank(method="first").astype(int).to_numpy()
            for i, drv in enumerate(drivers):
                rows.append({
                    "season": season, "round": rnd, "event_name": f"Race {rnd}",
                    "circuit_id": "test", "driver": drv, "team_id": teams[drv],
                    "grid": int(grid[i]), "grid_pit_lane": False,
                    "quali_position": int(grid[i]), "quali_gap_to_pole_s": float(grid[i]) * 0.1,
                    "teammate_quali_gap_s": 0.0,
                    "driver_rolling_avg_finish_3": np.nan, "driver_rolling_avg_finish_5": np.nan,
                    "team_rolling_avg_finish_3": np.nan, "team_rolling_pace_gap_3": np.nan,
                    "driver_dnf_rate_10": 0.0, "circuit_type": "mixed", "reg_change_flag": False,
                    "dnf": 0.0, "source": "test",
                    "target_finish_pos": int(finish[i]), "target_points_top10": bool(finish[i] <= 10),
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


def test_evaluate_runs_and_produces_valid_metrics():
    """Not asserting the model beats the baselines -- see race_predictor.py's
    module docstring / the project's own honest write-up for that. Just that
    evaluate() runs end to end and every metric is a plausible number."""
    df = fake_race_dataset(n_seasons=2, n_rounds=7)
    preds, metrics = evaluate(df, min_train_races=6)
    assert metrics["n_races"] > 0
    for method in ("model", "baseline_grid", "baseline_quali"):
        m = metrics[method]
        assert m["position_mae"] > 0
        assert -1 <= m["spearman"] <= 1
        assert 0 <= m["top3_hit_rate"] <= 1
        assert 0 <= m["winner_accuracy"] <= 1
        assert 0 <= m["points_f1"] <= 1
    for _, race in preds.groupby(GROUP_COLS):
        assert sorted(race["model_pos"]) == list(range(1, len(race) + 1))


def test_predict_race_ranks_single_race():
    df = fake_race_dataset(n_seasons=4, n_rounds=10)
    pos_pipe = make_position_pipeline().fit(_prep(df), df[TARGET_POS])
    pts_pipe = make_points_pipeline().fit(_prep(df), df[TARGET_PTS])
    next_race = df[df["round"] == 1].drop_duplicates("driver").copy()

    out = predict_race(pos_pipe, pts_pipe, next_race)
    assert sorted(out["predicted_position"]) == list(range(1, len(out) + 1))
    assert out["points_probability"].between(0, 1).all()
    assert list(out["predicted_position"]) == list(out["predicted_position"].sort_values())


def test_prep_handles_all_nan_numeric_column():
    """Regression test: an entirely-NaN numeric feature (e.g.
    team_rolling_pace_gap_3 in an early walk-forward fold trained only on
    seasons that predate it) must not crash HistGradientBoosting's binning.
    Confirmed live on sklearn 1.9.1: fitting on a column with 0 distinct
    non-null values raises "window shape cannot be larger than input array
    shape" inside sklearn's own _find_binning_thresholds. _prep() fills a
    fully-missing numeric column with 0.0 (zero variance either way, so this
    changes nothing predictively) specifically to avoid that crash."""
    df = fake_race_dataset(n_seasons=1, n_rounds=6, n_drivers=14)
    df["team_rolling_pace_gap_3"] = np.nan  # simulate the all-missing case
    X = _prep(df)
    assert X["team_rolling_pace_gap_3"].notna().all()
    pipe = make_position_pipeline().fit(X, df[TARGET_POS])
    pipe.predict(X)  # must not raise


def test_train_writes_model_and_metrics(tmp_path, monkeypatch):
    from app.models import race_predictor as rp

    monkeypatch.setattr(rp, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(rp, "MODEL_POS_PATH", tmp_path / "pos.joblib")
    monkeypatch.setattr(rp, "MODEL_PTS_PATH", tmp_path / "pts.joblib")
    monkeypatch.setattr(rp, "METRICS_PATH", tmp_path / "metrics.json")

    df = fake_race_dataset(n_seasons=3, n_rounds=8)  # 24 races: a few folds past train()'s default min_train_races=20
    metrics = rp.train(df)

    assert (tmp_path / "pos.joblib").exists()
    assert (tmp_path / "pts.joblib").exists()
    assert (tmp_path / "metrics.json").exists()
    assert metrics["n_races"] > 0
    pos_pipe, pts_pipe = rp.load_models()
    assert pos_pipe is not None and pts_pipe is not None
