"""Race outcome predictor: positions-gained regressor + DNF classifier,
combined through a Monte Carlo simulation into P(win)/P(podium)/P(points)
per driver, from pre-race features only (grid, qualifying, rolling form,
circuit type).

Framing (rewritten from a direct finish-position regressor, which lost to
the finish=grid baseline on every metric -- see git history / the README's
"Race outcome predictor" section for the old numbers):
- The finisher model predicts target_delta = finish - grid (positions
  gained), trained only on classified finishers (dnf == 0). Grid position
  alone already captures most of the signal in F1; asking the model for a
  small correction on top of it, instead of the whole answer, is a much
  easier regression problem and one that can shrink to "no change" when
  there's genuinely nothing to add.
- A separate DNF classifier predicts P(DNF) from a small, DNF-specific
  feature set (driver/team DNF rates, grid, circuit type, reg-change flag).
- predict_race() Monte Carlo simulates each race (default 10k runs):
  sample a DNF per driver per run from P(DNF), sample finisher noise from
  the walk-forward CV residual std of the delta model, rank the resulting
  raw positions within each run, and average over runs to get
  win/podium/points probabilities plus an expected position for display.

Evaluation protocol (see dev_holdout_split, select_hyperparams,
run_full_evaluation): all hyperparameter/design choices are made with
walk-forward CV on 2022-2024 ("dev") only; 2025-2026 ("holdout") is
evaluated exactly once and reported as-is, win or lose -- no tuning after
looking at it. select_hyperparams() refuses to run at all if it's ever
handed a holdout-season row, so that boundary can't be crossed by accident.

Trained on data/model/race_dataset.parquet (see scripts/build_race_dataset.py).
Pure functions -- everything here takes/returns plain DataFrames, no FastF1.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import brier_score_loss, f1_score, log_loss, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from app.config import MODEL_DIR

# --------------------------------------------------------------------------
# Feature sets. Two separate models -- the finisher delta regressor gets the
# full feature set (it's predicting a fine-grained continuous correction);
# the DNF classifier gets only the features the user asked for, since DNF is
# a much coarser, rarer event that doesn't need (and would overfit on) the
# full set.
# --------------------------------------------------------------------------

NUM_DELTA = ["grid", "quali_position", "quali_gap_to_pole_s", "teammate_quali_gap_s",
            "driver_rolling_avg_finish_3", "driver_rolling_avg_finish_5",
            "team_rolling_avg_finish_3", "team_rolling_pace_gap_3"]
BOOL_DELTA = ["grid_pit_lane", "reg_change_flag"]
CAT_DELTA = ["circuit_type"]
FEATURES_DELTA = NUM_DELTA + BOOL_DELTA + CAT_DELTA
TARGET_DELTA = "target_delta"

NUM_DNF = ["grid", "driver_dnf_rate_10", "team_dnf_rate_10"]
BOOL_DNF = ["reg_change_flag"]
CAT_DNF = ["circuit_type"]
FEATURES_DNF = NUM_DNF + BOOL_DNF + CAT_DNF
TARGET_DNF = "dnf"

# Kept around for backward-compatible metrics/labels: the underlying
# ground-truth finish position and its derived points flag.
TARGET_POS = "target_finish_pos"
TARGET_PTS = "target_points_top10"
GROUP_COLS = ["season", "round"]

# All design decisions (features above, hyperparameters below, the model
# architecture itself) were chosen using only these seasons -- see
# select_hyperparams()'s hard guard and tests/test_race_predictor.py's
# leakage test. 2025+ is the untouched holdout.
DEV_SEASON_MAX = 2024

# Regularization chosen via select_hyperparams() walk-forward CV on 2022-2024
# only (small trees, high min_samples_leaf, early stopping -- shrinks toward
# "no change" i.e. delta=0 / DNF-rate-only when there's no real signal to
# add on top of grid). See the module docstring's evaluation-protocol note
# and README's "Race outcome predictor" section for the actual dev-CV
# comparison across the candidate grid.
DEFAULT_DELTA_PARAMS = {"max_leaf_nodes": 7, "min_samples_leaf": 30,
                        "learning_rate": 0.05, "max_iter": 500, "l2_regularization": 1.0}
DEFAULT_DNF_PARAMS = {"max_leaf_nodes": 7, "min_samples_leaf": 60,
                      "learning_rate": 0.05, "max_iter": 500, "l2_regularization": 1.0}

DELTA_HP_GRID = [
    {"max_leaf_nodes": 7, "min_samples_leaf": 60, "learning_rate": 0.05, "max_iter": 500, "l2_regularization": 1.0},
    {"max_leaf_nodes": 7, "min_samples_leaf": 30, "learning_rate": 0.05, "max_iter": 500, "l2_regularization": 1.0},
    {"max_leaf_nodes": 15, "min_samples_leaf": 30, "learning_rate": 0.05, "max_iter": 500, "l2_regularization": 1.0},
]
DNF_HP_GRID = [
    {"max_leaf_nodes": 7, "min_samples_leaf": 60, "learning_rate": 0.05, "max_iter": 500, "l2_regularization": 1.0},
    {"max_leaf_nodes": 7, "min_samples_leaf": 30, "learning_rate": 0.05, "max_iter": 500, "l2_regularization": 1.0},
    {"max_leaf_nodes": 15, "min_samples_leaf": 30, "learning_rate": 0.05, "max_iter": 500, "l2_regularization": 1.0},
]

MODEL_DELTA_PATH = MODEL_DIR / "race_predictor_delta.joblib"
MODEL_DNF_PATH = MODEL_DIR / "race_predictor_dnf.joblib"
META_PATH = MODEL_DIR / "race_predictor_meta.json"
METRICS_PATH = MODEL_DIR / "race_predictor_metrics.json"


def _prep(X: pd.DataFrame, num: list[str], bool_cols: list[str], cat: list[str]) -> pd.DataFrame:
    X = X.copy()
    for c in num:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype(float) if c in X else np.nan
        if X[c].notna().sum() == 0:
            # HistGradientBoosting's binning code (sklearn >=1.5-ish) raises
            # "window shape cannot be larger than input array shape" on a
            # column with zero non-missing values -- confirmed live: early
            # walk-forward folds train on 2022-only rows, where
            # team_rolling_pace_gap_3 is *always* NaN (prebuilt-only
            # feature). A fully-missing column carries no signal either way,
            # so filling with a constant costs nothing predictively.
            X[c] = 0.0
    for c in bool_cols:
        X[c] = X[c].fillna(False).astype(float) if c in X else 0.0
    for c in cat:
        X[c] = X[c].fillna("unknown").astype(str) if c in X else "unknown"
    return X[num + bool_cols + cat]


def _prep_delta(X: pd.DataFrame) -> pd.DataFrame:
    return _prep(X, NUM_DELTA, BOOL_DELTA, CAT_DELTA)


def _prep_dnf(X: pd.DataFrame) -> pd.DataFrame:
    return _prep(X, NUM_DNF, BOOL_DNF, CAT_DNF)


def _make_pipeline(estimator_cls, num, bool_cols, cat, params: dict) -> Pipeline:
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
        ("num", "passthrough", num + bool_cols),
    ])
    est = estimator_cls(random_state=42, early_stopping=True, validation_fraction=0.15,
                        n_iter_no_change=20, **params)
    step_name = "reg" if hasattr(est, "predict") and not hasattr(est, "predict_proba") else "clf"
    return Pipeline([("pre", pre), (step_name, est)])


def make_delta_pipeline(**params) -> Pipeline:
    return _make_pipeline(HistGradientBoostingRegressor, NUM_DELTA, BOOL_DELTA, CAT_DELTA,
                          {**DEFAULT_DELTA_PARAMS, **params})


def make_dnf_pipeline(**params) -> Pipeline:
    return _make_pipeline(HistGradientBoostingClassifier, NUM_DNF, BOOL_DNF, CAT_DNF,
                          {**DEFAULT_DNF_PARAMS, **params})


def rank_within_race(df: pd.DataFrame, value_col: str, out_col: str,
                     group_cols: list[str] = GROUP_COLS) -> pd.DataFrame:
    """Turn a raw (possibly continuous, tied, or NaN) per-driver score into a
    unique 1..N rank within each race -- two drivers can't both "finish P3".
    A NaN score (e.g. no valid qualifying time) is treated as worse than
    everyone else in that race, not dropped. Pass group_cols=None to rank a
    single race (no grouping needed)."""
    df = df.copy()
    if group_cols:
        race_max = df.groupby(group_cols)[value_col].transform("max")
        filled = df[value_col].fillna(race_max + 1)
        df[out_col] = filled.groupby([df[c] for c in group_cols]).rank(method="first").astype(int)
    else:
        filled = df[value_col].fillna(df[value_col].max() + 1)
        df[out_col] = filled.rank(method="first").astype(int)
    return df


def race_sequence(df: pd.DataFrame) -> pd.Series:
    """Chronological index of each row's race (0, 1, 2, ...), ordered by
    (season, round). season/round is a reliable proxy for calendar order:
    round is defined as the calendar sequence number within a season."""
    races = df[GROUP_COLS].drop_duplicates().sort_values(GROUP_COLS).reset_index(drop=True)
    races["_race_seq"] = np.arange(len(races))
    return df.merge(races, on=GROUP_COLS, how="left")["_race_seq"].to_numpy()


def time_based_splits(df: pd.DataFrame, min_train_races: int = 20):
    """Expanding-window walk-forward splits: train on every race strictly
    before race k, test on race k alone. Starts after `min_train_races`
    distinct races so early folds aren't trained on almost nothing.
    Yields (train_index, test_index) pairs of df's index values.
    """
    seq = pd.Series(race_sequence(df), index=df.index)
    max_seq = int(seq.max())
    for k in range(min_train_races, max_seq + 1):
        train_idx = seq.index[seq < k]
        test_idx = seq.index[seq == k]
        if len(test_idx) == 0 or len(train_idx) == 0:
            continue
        yield train_idx, test_idx


def dev_holdout_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Season <= DEV_SEASON_MAX (2022-2024) is "dev": the only data allowed
    to influence any design/hyperparameter decision. Everything after that
    (2025-2026) is "holdout": evaluated exactly once, never used to pick
    anything."""
    dev = df[df["season"] <= DEV_SEASON_MAX].reset_index(drop=True)
    holdout = df[df["season"] > DEV_SEASON_MAX].reset_index(drop=True)
    return dev, holdout


# --------------------------------------------------------------------------
# Monte Carlo race simulation
# --------------------------------------------------------------------------

def simulate_positions(grid: pd.Series, delta_pred: pd.Series, p_dnf: pd.Series,
                       dnf_position_samples: np.ndarray, residual_std: float,
                       n_sims: int = 10_000, seed: int | None = None) -> dict[str, np.ndarray]:
    """Monte Carlo simulate one race n_sims times: each run, every driver
    either DNFs (drawn from p_dnf, classified at a position resampled from
    dnf_position_samples -- the empirical distribution of where retirees
    actually got classified historically) or finishes at
    grid + delta_pred + Normal(0, residual_std) noise. Raw positions are
    ranked within each run (argsort trick) so every run is a valid unique
    1..N finishing order, then averaged into per-driver probabilities.
    Fully vectorized: one (n_sims, n_drivers) array of draws, not a python
    loop per simulation.
    """
    n = len(grid)
    rng = np.random.default_rng(seed)
    grid_a = grid.to_numpy(dtype=float)
    delta_a = delta_pred.to_numpy(dtype=float)
    p_dnf_a = np.clip(p_dnf.to_numpy(dtype=float), 0.0, 1.0)

    dnf_draw = rng.random((n_sims, n)) < p_dnf_a[None, :]
    noise = rng.normal(0.0, max(residual_std, 1e-6), size=(n_sims, n))
    finisher_raw = grid_a[None, :] + delta_a[None, :] + noise
    samples = dnf_position_samples if len(dnf_position_samples) else np.array([grid_a.max()])
    dnf_raw = rng.choice(samples, size=(n_sims, n))
    raw = np.where(dnf_draw, dnf_raw, finisher_raw)

    order = np.argsort(raw, axis=1, kind="stable")
    ranks = np.empty_like(order)
    rows = np.arange(n_sims)[:, None]
    ranks[rows, order] = np.arange(1, n + 1)[None, :]

    return {
        "p_win": (ranks == 1).mean(axis=0),
        "p_podium": (ranks <= 3).mean(axis=0),
        "p_points": (ranks <= 10).mean(axis=0),
        "expected_position": ranks.mean(axis=0),
    }


# --------------------------------------------------------------------------
# Empirical grid -> probability baseline ("turn grid position into a
# probability", per the evaluation spec), fit once from dev seasons only.
# --------------------------------------------------------------------------

def empirical_grid_probs(dev_df: pd.DataFrame, max_grid: int = 20) -> pd.DataFrame:
    """P(win|grid), P(podium|grid), P(points|grid) empirical frequencies
    from dev_df only. Grid values above max_grid (pit-lane starts, heavy
    penalties) are folded into the max_grid bucket -- sample sizes get thin
    at the very back, and "started somewhere near the back" is the
    meaningful signal there anyway."""
    d = dev_df.copy()
    d["grid_bucket"] = d["grid"].clip(upper=max_grid).round().astype(int)
    rows = []
    for bucket, g in d.groupby("grid_bucket"):
        rows.append({
            "grid_bucket": bucket, "n": len(g),
            "p_win": (g[TARGET_POS] == 1).mean(),
            "p_podium": (g[TARGET_POS] <= 3).mean(),
            "p_points": (g[TARGET_POS] <= 10).mean(),
        })
    return pd.DataFrame(rows).sort_values("grid_bucket").reset_index(drop=True)


def baseline_grid_probs(grid: pd.Series, table: pd.DataFrame, max_grid: int = 20) -> pd.DataFrame:
    bucket = grid.clip(upper=max_grid).round().astype(int)
    idx = table.set_index("grid_bucket")
    return pd.DataFrame({
        "p_win": bucket.map(idx["p_win"]).fillna(idx["p_win"].min()).to_numpy(),
        "p_podium": bucket.map(idx["p_podium"]).fillna(idx["p_podium"].min()).to_numpy(),
        "p_points": bucket.map(idx["p_points"]).fillna(idx["p_points"].min()).to_numpy(),
    }, index=grid.index)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def _race_metrics(preds: pd.DataFrame, pos_col: str) -> dict:
    maes, spearmans, top3_hits, winner_hits = [], [], [], []
    for _, race in preds.groupby(GROUP_COLS):
        maes.append(mean_absolute_error(race[TARGET_POS], race[pos_col]))
        if race[TARGET_POS].nunique() > 1 and race[pos_col].nunique() > 1:
            spearmans.append(spearmanr(race[TARGET_POS], race[pos_col]).correlation)
        pred_winner = race.loc[race[pos_col] == 1]
        if not pred_winner.empty:
            actual_pos = pred_winner[TARGET_POS].iloc[0]
            top3_hits.append(actual_pos <= 3)
            winner_hits.append(actual_pos == 1)
    return {
        "position_mae": round(float(np.mean(maes)), 3) if maes else None,
        "spearman": round(float(np.nanmean(spearmans)), 3) if spearmans else None,
        "top3_hit_rate": round(float(np.mean(top3_hits)), 3) if top3_hits else None,
        "winner_accuracy": round(float(np.mean(winner_hits)), 3) if winner_hits else None,
    }


def _points_f1(y_true: pd.Series, y_pred) -> float | None:
    if y_true.nunique() < 2:
        return None
    return round(float(f1_score(y_true, y_pred)), 3)


def _prob_metrics(actual: pd.Series, p: pd.Series) -> dict:
    if actual.nunique() < 2:
        return {"brier": None, "logloss": None}
    p = p.clip(1e-6, 1 - 1e-6)
    return {"brier": round(float(brier_score_loss(actual, p)), 4),
           "logloss": round(float(log_loss(actual, p, labels=[0, 1])), 4)}


def _calibration(actual: pd.Series, p: pd.Series, n_bins: int = 10) -> list[dict] | None:
    if actual.nunique() < 2 or len(actual) < n_bins:
        return None
    try:
        observed, predicted = calibration_curve(actual, p, n_bins=n_bins, strategy="quantile")
    except ValueError:
        return None
    return [{"predicted": round(float(pr), 4), "observed": round(float(ob), 4)}
           for pr, ob in zip(predicted, observed)]


def summarise(preds: pd.DataFrame, grid_prob_table: pd.DataFrame) -> dict:
    """preds must have: season, round, driver, grid, target_finish_pos,
    model_pos/baseline_grid_pos/baseline_quali_pos (+*_points), and
    p_win/p_podium/p_points (the model's Monte Carlo probabilities)."""
    if preds.empty:
        return {"n_races": 0, "n_rows": 0}
    out = {"n_races": int(preds[GROUP_COLS].drop_duplicates().shape[0]), "n_rows": int(len(preds))}

    point_metrics = {}
    for method in ("model", "baseline_grid", "baseline_quali"):
        m = _race_metrics(preds, f"{method}_pos")
        m["points_f1"] = _points_f1(preds[TARGET_PTS], preds[f"{method}_points"])
        point_metrics[method] = m
    out["point_metrics"] = point_metrics

    bp = baseline_grid_probs(preds["grid"], grid_prob_table)
    actual_win = (preds[TARGET_POS] == 1).astype(int)
    actual_podium = (preds[TARGET_POS] <= 3).astype(int)
    actual_points = (preds[TARGET_POS] <= 10).astype(int)
    out["prob_metrics"] = {
        "model": {
            "win": _prob_metrics(actual_win, preds["p_win"]),
            "podium": _prob_metrics(actual_podium, preds["p_podium"]),
            "points": _prob_metrics(actual_points, preds["p_points"]),
        },
        "baseline_grid_prob": {
            "win": _prob_metrics(actual_win, bp["p_win"]),
            "podium": _prob_metrics(actual_podium, bp["p_podium"]),
            "points": _prob_metrics(actual_points, bp["p_points"]),
        },
    }
    out["calibration"] = {
        "win": _calibration(actual_win, preds["p_win"]),
        "podium": _calibration(actual_podium, preds["p_podium"]),
        "points": _calibration(actual_points, preds["p_points"]),
    }
    return out


# --------------------------------------------------------------------------
# Walk-forward evaluation
# --------------------------------------------------------------------------

def evaluate(df: pd.DataFrame, min_train_races: int = 15, delta_params: dict | None = None,
            dnf_params: dict | None = None, n_sims: int = 10_000,
            seed: int = 42) -> tuple[pd.DataFrame, dict]:
    """Runs the expanding-window walk-forward loop: at each fold, fits the
    delta regressor (on that fold's classified finishers only) and the DNF
    classifier (on all of that fold's training rows), predicts the held-out
    race, then Monte Carlo simulates it for win/podium/points probabilities.
    Never trains on a race that hasn't happened yet relative to the race
    being predicted. Returns (per-row predictions, {"residual_std": ...,
    "dnf_position_prior": ...}) -- summarise() turns predictions into the
    actual metrics dict; this function only produces the raw predictions.
    """
    df = df.reset_index(drop=True)
    delta_params = delta_params or {}
    dnf_params = dnf_params or {}
    rows = []
    dnf_samples_by_race: dict[tuple, np.ndarray] = {}

    for train_idx, test_idx in time_based_splits(df, min_train_races):
        train, test = df.loc[train_idx], df.loc[test_idx]

        train_delta = train[train[TARGET_DNF] == 0].dropna(subset=[TARGET_DELTA])
        train_dnf = train.dropna(subset=[TARGET_DNF])
        if len(train_delta) < 10 or train_dnf[TARGET_DNF].nunique() < 2:
            continue

        delta_pipe = make_delta_pipeline(**delta_params).fit(_prep_delta(train_delta), train_delta[TARGET_DELTA])
        dnf_pipe = make_dnf_pipeline(**dnf_params).fit(_prep_dnf(train_dnf), train_dnf[TARGET_DNF])

        dnf_rows = train.loc[train[TARGET_DNF] == 1, TARGET_POS].dropna()
        dnf_prior = float(dnf_rows.mean()) if len(dnf_rows) else float(train[TARGET_POS].max())
        dnf_samples = dnf_rows.to_numpy() if len(dnf_rows) >= 5 else np.array([dnf_prior])

        out = test[GROUP_COLS + ["driver", "grid", "quali_position", TARGET_POS, TARGET_PTS, TARGET_DNF]].copy()
        out["delta_pred"] = delta_pipe.predict(_prep_delta(test))
        out["p_dnf"] = dnf_pipe.predict_proba(_prep_dnf(test))[:, 1]
        out["dnf_prior"] = dnf_prior
        out["actual_delta"] = out[TARGET_POS] - out["grid"]
        out["baseline_grid_pos_raw"] = test["grid"]
        out["baseline_quali_pos_raw"] = test["quali_position"]
        key = (test[GROUP_COLS[0]].iloc[0], test[GROUP_COLS[1]].iloc[0])
        dnf_samples_by_race[key] = dnf_samples
        rows.append(out)

    if not rows:
        raise ValueError(f"Not enough races for a single fold (need > {min_train_races})")
    preds = pd.concat(rows, ignore_index=True)

    residual_std = float((preds.loc[preds[TARGET_DNF] == 0, "actual_delta"]
                          - preds.loc[preds[TARGET_DNF] == 0, "delta_pred"]).std())
    if not np.isfinite(residual_std) or residual_std <= 0:
        residual_std = float(preds["actual_delta"].abs().mean()) or 1.0

    sim_cols = {"p_win": [], "p_podium": [], "p_points": [], "expected_position": []}
    for key, race in preds.groupby(GROUP_COLS, sort=False):
        samples = dnf_samples_by_race.get(key, np.array([race[TARGET_POS].max()]))
        sim = simulate_positions(race["grid"], race["delta_pred"], race["p_dnf"], samples,
                                 residual_std, n_sims=n_sims, seed=seed)
        for k in sim_cols:
            sim_cols[k].append(pd.Series(sim[k], index=race.index))
    for k, parts in sim_cols.items():
        preds[k] = pd.concat(parts).sort_index()

    preds = rank_within_race(preds, "expected_position", "model_pos")
    preds = rank_within_race(preds, "baseline_grid_pos_raw", "baseline_grid_pos")
    preds = rank_within_race(preds, "baseline_quali_pos_raw", "baseline_quali_pos")
    preds["model_points"] = preds["model_pos"] <= 10
    preds["baseline_grid_points"] = preds["baseline_grid_pos"] <= 10
    preds["baseline_quali_points"] = preds["baseline_quali_pos"] <= 10

    meta = {"residual_std": round(residual_std, 4),
           "dnf_position_prior": round(float(preds["dnf_prior"].mean()), 3)}
    return preds, meta


def run_full_evaluation(df: pd.DataFrame, delta_params: dict | None = None, dnf_params: dict | None = None,
                        min_train_races: int = 15, n_sims: int = 10_000, seed: int = 42) -> dict:
    """The full reported evaluation: one walk-forward pass across the whole
    timeline (causal throughout), then the resulting predictions are split
    by season into dev (2022-2024) and holdout (2025-2026) and summarised
    separately. The holdout numbers here are meant to be looked at ONCE --
    see the module docstring."""
    dev_df, _ = dev_holdout_split(df)
    grid_prob_table = empirical_grid_probs(dev_df)
    preds, meta = evaluate(df, min_train_races, delta_params, dnf_params, n_sims=n_sims, seed=seed)

    dev_preds = preds[preds["season"] <= DEV_SEASON_MAX]
    holdout_preds = preds[preds["season"] > DEV_SEASON_MAX]

    report = {
        "n_races": int(preds[GROUP_COLS].drop_duplicates().shape[0]),
        "n_rows": int(len(preds)),
        "residual_std": meta["residual_std"],
        "dnf_position_prior": meta["dnf_position_prior"],
        "dev": summarise(dev_preds, grid_prob_table),
        "holdout": summarise(holdout_preds, grid_prob_table),
        "by_season": {str(season): summarise(g, grid_prob_table) for season, g in preds.groupby("season")},
    }
    if not holdout_preds.empty:
        report["holdout"]["bootstrap_ci"] = bootstrap_diff_ci(holdout_preds, grid_prob_table)
    return report


# --------------------------------------------------------------------------
# Bootstrap CIs for model-minus-baseline differences (resample races, not
# rows -- a race is the unit of "how would a different sample of races have
# looked", so this is a cluster/block bootstrap).
# --------------------------------------------------------------------------

def _per_race_bootstrap_data(preds: pd.DataFrame, grid_prob_table: pd.DataFrame) -> list[dict]:
    races = []
    for _, race in preds.groupby(GROUP_COLS, sort=False):
        bp = baseline_grid_probs(race["grid"], grid_prob_table)
        races.append({
            "model_mae": mean_absolute_error(race[TARGET_POS], race["model_pos"]),
            "grid_mae": mean_absolute_error(race[TARGET_POS], race["baseline_grid_pos"]),
            "actual_points": (race[TARGET_POS] <= 10).to_numpy(dtype=float),
            "actual_win": (race[TARGET_POS] == 1).to_numpy(dtype=float),
            "model_p_points": race["p_points"].to_numpy(),
            "base_p_points": bp["p_points"].to_numpy(),
            "model_p_win": race["p_win"].to_numpy(),
            "base_p_win": bp["p_win"].to_numpy(),
        })
    return races


def bootstrap_diff_ci(preds: pd.DataFrame, grid_prob_table: pd.DataFrame, n_boot: int = 2000,
                      ci: float = 0.95, seed: int = 42) -> dict:
    """Percentile bootstrap CI, resampling whole races with replacement, for
    model-minus-baseline differences on position MAE (vs finish=grid),
    points Brier (vs the empirical grid-probability baseline), and win
    Brier (same baseline). Negative mean_diff = model better (lower
    error/Brier) than the baseline; the CI says how strongly the holdout
    data actually supports that, rather than just the point estimate."""
    race_data = _per_race_bootstrap_data(preds, grid_prob_table)
    n = len(race_data)
    rng = np.random.default_rng(seed)
    mae_diff = np.empty(n_boot)
    points_brier_diff = np.empty(n_boot)
    win_brier_diff = np.empty(n_boot)

    for b in range(n_boot):
        sample = [race_data[i] for i in rng.integers(0, n, size=n)]
        mae_diff[b] = (np.mean([r["model_mae"] for r in sample])
                      - np.mean([r["grid_mae"] for r in sample]))

        actual_points = np.concatenate([r["actual_points"] for r in sample])
        model_p_points = np.clip(np.concatenate([r["model_p_points"] for r in sample]), 1e-6, 1 - 1e-6)
        base_p_points = np.clip(np.concatenate([r["base_p_points"] for r in sample]), 1e-6, 1 - 1e-6)
        points_brier_diff[b] = (brier_score_loss(actual_points, model_p_points)
                                - brier_score_loss(actual_points, base_p_points))

        actual_win = np.concatenate([r["actual_win"] for r in sample])
        model_p_win = np.clip(np.concatenate([r["model_p_win"] for r in sample]), 1e-6, 1 - 1e-6)
        base_p_win = np.clip(np.concatenate([r["base_p_win"] for r in sample]), 1e-6, 1 - 1e-6)
        win_brier_diff[b] = (brier_score_loss(actual_win, model_p_win)
                             - brier_score_loss(actual_win, base_p_win))

    alpha = (1 - ci) / 2

    def _summarise(arr: np.ndarray) -> dict:
        return {"mean_diff": round(float(arr.mean()), 4),
               "ci_low": round(float(np.percentile(arr, 100 * alpha)), 4),
               "ci_high": round(float(np.percentile(arr, 100 * (1 - alpha))), 4)}

    return {"n_boot": n_boot, "ci": ci,
           "position_mae_diff": _summarise(mae_diff),
           "points_brier_diff": _summarise(points_brier_diff),
           "win_brier_diff": _summarise(win_brier_diff)}


# --------------------------------------------------------------------------
# Hyperparameter selection -- dev seasons (<=2024) only. This is the
# mechanism, not just a convention: select_hyperparams() refuses to run if
# handed any holdout-season row, so 2025+ structurally cannot influence the
# choice. See tests/test_race_predictor.py for the regression test.
# --------------------------------------------------------------------------

def select_hyperparams(dev_df: pd.DataFrame, grid: list[dict] | None = None,
                       min_train_races: int = 10, n_sims: int = 200) -> dict:
    """Pick delta-regressor hyperparams by walk-forward CV delta-MAE on
    dev_df alone. Raises if dev_df contains any row from a holdout season --
    the whole point of this function is that it structurally cannot see
    2025+ data, not just that callers are asked nicely not to pass it."""
    if (dev_df["season"] > DEV_SEASON_MAX).any():
        raise ValueError(f"select_hyperparams must only see dev seasons (<= {DEV_SEASON_MAX}) -- "
                         "pass dev_holdout_split(df)[0], not the full dataset")
    best_cfg, best_score = None, np.inf
    for cfg in (grid or DELTA_HP_GRID):
        preds, _ = evaluate(dev_df, min_train_races=min_train_races, delta_params=cfg, n_sims=n_sims)
        finishers = preds[preds[TARGET_DNF] == 0]
        score = mean_absolute_error(finishers["actual_delta"], finishers["delta_pred"])
        if score < best_score:
            best_score, best_cfg = score, cfg
    return best_cfg


def select_dnf_hyperparams(dev_df: pd.DataFrame, grid: list[dict] | None = None,
                           min_train_races: int = 10, n_sims: int = 200) -> dict:
    """Same guard and mechanism as select_hyperparams(), scored by dev
    walk-forward log loss of P(DNF) instead of delta MAE."""
    if (dev_df["season"] > DEV_SEASON_MAX).any():
        raise ValueError(f"select_dnf_hyperparams must only see dev seasons (<= {DEV_SEASON_MAX}) -- "
                         "pass dev_holdout_split(df)[0], not the full dataset")
    best_cfg, best_score = None, np.inf
    for cfg in (grid or DNF_HP_GRID):
        preds, _ = evaluate(dev_df, min_train_races=min_train_races, dnf_params=cfg, n_sims=n_sims)
        p = preds["p_dnf"].clip(1e-6, 1 - 1e-6)
        score = log_loss(preds[TARGET_DNF], p, labels=[0, 1])
        if score < best_score:
            best_score, best_cfg = score, cfg
    return best_cfg


# --------------------------------------------------------------------------
# Final fit (all data) + single-race prediction, for the API/dashboard.
# --------------------------------------------------------------------------

def _quick_residual_std(df: pd.DataFrame, delta_params: dict | None = None, holdout_frac: float = 0.2) -> float:
    """Cheap single-split out-of-sample residual estimate for the fast
    ensure_trained() path -- NOT the honest walk-forward figure (that's
    what train()/run_full_evaluation() compute and cache). Falls back to an
    in-sample estimate if there isn't enough data for a clean split."""
    seq = race_sequence(df)
    d = df.assign(_seq=seq)
    cutoff = d["_seq"].quantile(1 - holdout_frac)
    train, test = d[d["_seq"] <= cutoff], d[d["_seq"] > cutoff]
    train_delta = train[train[TARGET_DNF] == 0].dropna(subset=[TARGET_DELTA])
    test_delta = test[test[TARGET_DNF] == 0].dropna(subset=[TARGET_DELTA])
    if len(train_delta) < 20 or len(test_delta) < 5:
        all_delta = d[d[TARGET_DNF] == 0].dropna(subset=[TARGET_DELTA])
        pipe = make_delta_pipeline(**(delta_params or {})).fit(_prep_delta(all_delta), all_delta[TARGET_DELTA])
        resid = all_delta[TARGET_DELTA] - pipe.predict(_prep_delta(all_delta))
        return float(resid.std()) or 1.0
    pipe = make_delta_pipeline(**(delta_params or {})).fit(_prep_delta(train_delta), train_delta[TARGET_DELTA])
    resid = test_delta[TARGET_DELTA] - pipe.predict(_prep_delta(test_delta))
    return float(resid.std()) or 1.0


def fit_final(df: pd.DataFrame, delta_params: dict | None = None,
             dnf_params: dict | None = None) -> tuple[Pipeline, Pipeline, dict]:
    """Fit delta + DNF pipelines on ALL of df -- no CV/holdout. This is the
    model actually used for real predictions, as opposed to the
    walk-forward copies fit inside evaluate() purely for honest metrics."""
    train_delta = df[df[TARGET_DNF] == 0].dropna(subset=[TARGET_DELTA])
    train_dnf = df.dropna(subset=[TARGET_DNF])
    delta_pipe = make_delta_pipeline(**(delta_params or {})).fit(_prep_delta(train_delta), train_delta[TARGET_DELTA])
    dnf_pipe = make_dnf_pipeline(**(dnf_params or {})).fit(_prep_dnf(train_dnf), train_dnf[TARGET_DNF])

    dnf_rows = df.loc[df[TARGET_DNF] == 1, TARGET_POS].dropna()
    meta = {
        "residual_std": _quick_residual_std(df, delta_params),
        "dnf_position_prior": float(dnf_rows.mean()) if len(dnf_rows) else float(df[TARGET_POS].max()),
        "dnf_position_samples": (dnf_rows.tolist() if len(dnf_rows) >= 5
                                 else [float(dnf_rows.mean()) if len(dnf_rows) else float(df[TARGET_POS].max())]),
    }
    return delta_pipe, dnf_pipe, meta


def train(df: pd.DataFrame, delta_params: dict | None = None, dnf_params: dict | None = None) -> dict:
    """Fit the final model, run the full dev/holdout walk-forward evaluation
    for honest metrics, and cache everything to disk. Slow (many CV folds
    plus Monte Carlo per fold -- see evaluate()); use ensure_trained()
    instead when a caller just needs a usable model quickly (predictions,
    the API, the dashboard)."""
    delta_params = delta_params or DEFAULT_DELTA_PARAMS
    dnf_params = dnf_params or DEFAULT_DNF_PARAMS
    delta_pipe, dnf_pipe, meta = fit_final(df, delta_params, dnf_params)
    report = run_full_evaluation(df, delta_params, dnf_params)
    # Prefer the walk-forward (honest, out-of-sample-across-many-folds)
    # residual_std over the quick single-split estimate, now that we have it.
    meta["residual_std"] = report["residual_std"]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(delta_pipe, MODEL_DELTA_PATH)
    joblib.dump(dnf_pipe, MODEL_DNF_PATH)
    META_PATH.write_text(json.dumps(meta))
    METRICS_PATH.write_text(json.dumps(report, indent=2))
    return report


def load_models():
    if not (MODEL_DELTA_PATH.exists() and MODEL_DNF_PATH.exists() and META_PATH.exists()):
        return None, None, None
    meta = json.loads(META_PATH.read_text())
    meta["dnf_position_samples"] = np.array(meta["dnf_position_samples"])
    return joblib.load(MODEL_DELTA_PATH), joblib.load(MODEL_DNF_PATH), meta


def ensure_trained(dataset_path) -> tuple[Pipeline, Pipeline, dict]:
    """Load cached models if present, else fit fresh from the dataset at
    dataset_path (fast -- no walk-forward evaluation) and cache the result.
    Models are gitignored (regenerable, not source), so every consumer --
    the predict script, the API, the dashboard -- can be self-sufficient
    from just the small committed dataset parquet, with no pre-committed
    model artifact needed anywhere (local dev, CI, or Streamlit Cloud)."""
    delta_pipe, dnf_pipe, meta = load_models()
    if delta_pipe is not None:
        return delta_pipe, dnf_pipe, meta
    df = pd.read_parquet(dataset_path)
    delta_pipe, dnf_pipe, meta = fit_final(df)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(delta_pipe, MODEL_DELTA_PATH)
    joblib.dump(dnf_pipe, MODEL_DNF_PATH)
    META_PATH.write_text(json.dumps(meta))
    return delta_pipe, dnf_pipe, meta


def predict_race(delta_pipe, dnf_pipe, race_features: pd.DataFrame, meta: dict,
                 n_sims: int = 10_000, seed: int | None = None) -> pd.DataFrame:
    """race_features: one row per driver for a single upcoming race (same
    columns as the training features, plus "driver" and "grid" for display).
    Monte Carlo simulates the race and returns race_features with
    predicted_position (unique 1..N, ranked by expected position),
    win_probability, podium_probability and points_probability added,
    sorted by predicted_position.
    """
    out = race_features.copy()
    out["delta_pred"] = delta_pipe.predict(_prep_delta(out))
    out["p_dnf"] = dnf_pipe.predict_proba(_prep_dnf(out))[:, 1]
    sim = simulate_positions(out["grid"], out["delta_pred"], out["p_dnf"],
                             meta["dnf_position_samples"], meta["residual_std"],
                             n_sims=n_sims, seed=seed)
    out["win_probability"] = sim["p_win"]
    out["podium_probability"] = sim["p_podium"]
    out["points_probability"] = sim["p_points"]
    out["expected_position"] = sim["expected_position"]
    out = rank_within_race(out, "expected_position", "predicted_position", group_cols=None)
    return out.sort_values("predicted_position").reset_index(drop=True)
