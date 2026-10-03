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

import hashlib
import json
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, f1_score, log_loss, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from app.config import FROZEN_MODEL_DIR, MODEL_DIR, PREDICTIONS_DIR

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

# One joblib bundle {"delta_pipe":, "dnf_pipe":, "meta": {...}} rather than
# separate delta/dnf/meta files -- meta can hold live sklearn objects
# (IsotonicRegression calibrators), so it can't be plain JSON.
MODEL_BUNDLE_PATH = MODEL_DIR / "race_predictor_bundle.joblib"
METRICS_PATH = MODEL_DIR / "race_predictor_metrics.json"

# Committed frozen snapshot (see freeze_model()/load_frozen_model() below).
FROZEN_BUNDLE_PATH = FROZEN_MODEL_DIR / "model.joblib"
FROZEN_SPEC_PATH = FROZEN_MODEL_DIR / "spec.json"


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
# Grid-bucket-dependent Monte Carlo noise. Diagnosed on 2022-2024 dev-only
# walk-forward CV: front-row finishes are reliably noisier than mid-pack
# ones (residual std ~3.71 for grid 1-3 vs ~3.07 for grid 4-10, vs a global
# ~3.27), and a single global residual_std understates that -- which showed
# up as P(win) being under-confident for grid 1-3 (predicted ~0.225,
# actual ~0.283) and over-confident for grid 4-10 (predicted ~0.043, actual
# ~0.019). See README's "Race outcome predictor" section for the full
# before/after comparison.
# --------------------------------------------------------------------------

GRID_BUCKET_BINS = [0, 3, 10, np.inf]
GRID_BUCKET_LABELS = ["1-3", "4-10", "11+"]


def grid_bucket(grid: pd.Series) -> pd.Series:
    return pd.cut(grid, bins=GRID_BUCKET_BINS, labels=GRID_BUCKET_LABELS)


def residual_std_by_grid_bucket(preds: pd.DataFrame, min_bucket_n: int = 10) -> dict[str, float]:
    """Per-grid-bucket residual std (actual_delta - delta_pred) over
    classified finishers in a walk-forward preds DataFrame. A bucket with
    too few rows (< min_bucket_n) falls back to the overall std instead of
    an unstable small-sample estimate."""
    finishers = preds[preds[TARGET_DNF] == 0].copy()
    resid = finishers[TARGET_POS] - finishers["grid"] - finishers["delta_pred"]
    overall = float(resid.std()) if len(resid) > 1 else 1.0
    finishers = finishers.assign(_resid=resid, _bucket=grid_bucket(finishers["grid"]))
    out = {}
    for label in GRID_BUCKET_LABELS:
        g = finishers.loc[finishers["_bucket"] == label, "_resid"]
        out[label] = float(g.std()) if len(g) >= min_bucket_n else overall
    return out


def grid_bucket_residual_std_array(grid: pd.Series, bucket_stds: dict[str, float],
                                   fallback: float) -> np.ndarray:
    """Map each driver's grid to their bucket's residual std, for
    simulate_positions()'s per-driver residual_std array."""
    buckets = grid_bucket(grid).astype(str)
    return buckets.map(bucket_stds).fillna(fallback).to_numpy(dtype=float)


# --------------------------------------------------------------------------
# Monte Carlo race simulation
# --------------------------------------------------------------------------

def simulate_positions(grid: pd.Series, delta_pred: pd.Series, p_dnf: pd.Series,
                       dnf_position_samples: np.ndarray, residual_std: float | np.ndarray,
                       n_sims: int = 10_000, seed: int | None = None) -> dict[str, np.ndarray]:
    """Monte Carlo simulate one race n_sims times: each run, every driver
    either DNFs (drawn from p_dnf, classified at a position resampled from
    dnf_position_samples -- the empirical distribution of where retirees
    actually got classified historically) or finishes at
    grid + delta_pred + Normal(0, residual_std) noise. residual_std can be a
    single scalar (broadcast to every driver) or a per-driver array (e.g.
    grid-bucket-dependent -- see residual_std_by_grid_bucket()/
    grid_bucket_residual_std_array(), used because front-row finishes are
    reliably noisier than mid-pack ones: see the module docstring's
    calibration note). Raw positions are ranked within each run (argsort
    trick) so every run is a valid unique 1..N finishing order, then
    averaged into per-driver probabilities. Fully vectorized: one
    (n_sims, n_drivers) array of draws, not a python loop per simulation.
    """
    n = len(grid)
    rng = np.random.default_rng(seed)
    grid_a = grid.to_numpy(dtype=float)
    delta_a = delta_pred.to_numpy(dtype=float)
    p_dnf_a = np.clip(p_dnf.to_numpy(dtype=float), 0.0, 1.0)
    residual_std_a = np.maximum(np.broadcast_to(np.asarray(residual_std, dtype=float), (n,)), 1e-6)

    dnf_draw = rng.random((n_sims, n)) < p_dnf_a[None, :]
    noise = rng.normal(0.0, 1.0, size=(n_sims, n)) * residual_std_a[None, :]
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

def _walk_forward_raw(df: pd.DataFrame, min_train_races: int, delta_params: dict | None,
                      dnf_params: dict | None) -> tuple[pd.DataFrame, dict[tuple, np.ndarray]]:
    """The expanding-window walk-forward loop itself: at each fold, fits the
    delta regressor (on that fold's classified finishers only) and the DNF
    classifier (on all of that fold's training rows), and predicts the
    held-out race. Never trains on a race that hasn't happened yet relative
    to the race being predicted. Returns raw per-row predictions (no Monte
    Carlo/probabilities yet -- see _run_monte_carlo()) plus each race's
    empirical DNF-position sample pool."""
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
    return pd.concat(rows, ignore_index=True), dnf_samples_by_race


def _run_monte_carlo(preds: pd.DataFrame, dnf_samples_by_race: dict[tuple, np.ndarray],
                     residual_std: float, bucket_stds: dict[str, float] | None = None,
                     n_sims: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """Adds p_win/p_podium/p_points/expected_position columns by Monte Carlo
    simulating every race in preds. residual_std is the scalar fallback;
    pass bucket_stds (from residual_std_by_grid_bucket(), fit on dev only)
    to additionally scale noise per grid bucket."""
    preds = preds.copy()
    sim_cols = {"p_win": [], "p_podium": [], "p_points": [], "expected_position": []}
    for key, race in preds.groupby(GROUP_COLS, sort=False):
        samples = dnf_samples_by_race.get(key, np.array([race[TARGET_POS].max()]))
        std = (grid_bucket_residual_std_array(race["grid"], bucket_stds, residual_std)
              if bucket_stds is not None else residual_std)
        sim = simulate_positions(race["grid"], race["delta_pred"], race["p_dnf"], samples,
                                 std, n_sims=n_sims, seed=seed)
        for k in sim_cols:
            sim_cols[k].append(pd.Series(sim[k], index=race.index))
    for k, parts in sim_cols.items():
        preds[k] = pd.concat(parts).sort_index()
    return preds


def _rank_and_flag(preds: pd.DataFrame) -> pd.DataFrame:
    preds = rank_within_race(preds, "expected_position", "model_pos")
    preds = rank_within_race(preds, "baseline_grid_pos_raw", "baseline_grid_pos")
    preds = rank_within_race(preds, "baseline_quali_pos_raw", "baseline_quali_pos")
    preds["model_points"] = preds["model_pos"] <= 10
    preds["baseline_grid_points"] = preds["baseline_grid_pos"] <= 10
    preds["baseline_quali_points"] = preds["baseline_quali_pos"] <= 10
    return preds


def evaluate(df: pd.DataFrame, min_train_races: int = 15, delta_params: dict | None = None,
            dnf_params: dict | None = None, n_sims: int = 10_000,
            seed: int = 42) -> tuple[pd.DataFrame, dict]:
    """Walk-forward point predictions + Monte Carlo with a single global
    residual_std (no grid-bucket noise, no isotonic calibration) -- used by
    select_hyperparams()/select_dnf_hyperparams() and tests that only need a
    quick, self-contained evaluation. run_full_evaluation() is the fuller
    version used for the officially reported metrics (dev-only-fit bucket
    noise + calibration, applied consistently to dev and holdout)."""
    preds, dnf_samples_by_race = _walk_forward_raw(df, min_train_races, delta_params, dnf_params)

    residual_std = float((preds.loc[preds[TARGET_DNF] == 0, "actual_delta"]
                          - preds.loc[preds[TARGET_DNF] == 0, "delta_pred"]).std())
    if not np.isfinite(residual_std) or residual_std <= 0:
        residual_std = float(preds["actual_delta"].abs().mean()) or 1.0

    preds = _run_monte_carlo(preds, dnf_samples_by_race, residual_std, n_sims=n_sims, seed=seed)
    preds = _rank_and_flag(preds)

    meta = {"residual_std": round(residual_std, 4),
           "dnf_position_prior": round(float(preds["dnf_prior"].mean()), 3)}
    return preds, meta


def _fit_dev_calibration(dev_df: pd.DataFrame, delta_params: dict | None = None,
                         dnf_params: dict | None = None, min_train_races: int = 15,
                         n_sims: int = 10_000, seed: int = 42) -> dict:
    """Fits everything calibration-related on dev (2022-2024) walk-forward
    CV ONLY: the global + grid-bucket residual std (see the module's
    grid-bucket-noise note) and isotonic P(win)/P(podium) calibrators
    (diagnosed necessary on top of bucket noise -- see README's "Race
    outcome predictor" section for the before/after numbers). The returned
    calibrators must then be applied unchanged to any other data (holdout,
    a live prediction) -- never refit on it."""
    dev_raw, dnf_samples_by_race = _walk_forward_raw(dev_df, min_train_races, delta_params, dnf_params)
    finishers = dev_raw[dev_raw[TARGET_DNF] == 0]
    residual_std = float((finishers["actual_delta"] - finishers["delta_pred"]).std())
    if not np.isfinite(residual_std) or residual_std <= 0:
        residual_std = float(dev_raw["actual_delta"].abs().mean()) or 1.0
    bucket_stds = residual_std_by_grid_bucket(dev_raw)

    dev_sim = _run_monte_carlo(dev_raw, dnf_samples_by_race, residual_std, bucket_stds=bucket_stds,
                              n_sims=n_sims, seed=seed)
    actual_win = (dev_sim[TARGET_POS] == 1).astype(int)
    actual_podium = (dev_sim[TARGET_POS] <= 3).astype(int)
    win_calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(
        dev_sim["p_win"], actual_win)
    podium_calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(
        dev_sim["p_podium"], actual_podium)

    return {
        "residual_std": round(residual_std, 4),
        "residual_std_by_bucket": {k: round(v, 4) for k, v in bucket_stds.items()},
        "win_calibrator": win_calibrator,
        "podium_calibrator": podium_calibrator,
    }


def run_full_evaluation(df: pd.DataFrame, delta_params: dict | None = None, dnf_params: dict | None = None,
                        min_train_races: int = 15, n_sims: int = 10_000, seed: int = 42) -> dict:
    """The full reported evaluation: one walk-forward pass across the whole
    timeline (causal throughout) for delta/DNF point predictions, but the
    Monte Carlo noise scheme and P(win)/P(podium) isotonic calibrators are
    fit on dev (2022-2024) ONLY (_fit_dev_calibration) and then applied
    unchanged to every race, dev and holdout alike -- holdout never
    contributes to its own calibration. Predictions are then split by
    season into dev and holdout and summarised separately. The holdout
    numbers here are meant to be looked at ONCE -- see the module
    docstring."""
    dev_df, _ = dev_holdout_split(df)
    grid_prob_table = empirical_grid_probs(dev_df)

    preds_raw, dnf_samples_by_race = _walk_forward_raw(df, min_train_races, delta_params, dnf_params)
    calib = _fit_dev_calibration(dev_df, delta_params, dnf_params, min_train_races, n_sims, seed)

    preds = _run_monte_carlo(preds_raw, dnf_samples_by_race, calib["residual_std"],
                             bucket_stds=calib["residual_std_by_bucket"], n_sims=n_sims, seed=seed)
    preds["p_win"] = calib["win_calibrator"].predict(preds["p_win"])
    preds["p_podium"] = calib["podium_calibrator"].predict(preds["p_podium"])
    preds = _rank_and_flag(preds)

    dev_preds = preds[preds["season"] <= DEV_SEASON_MAX]
    holdout_preds = preds[preds["season"] > DEV_SEASON_MAX]

    report = {
        "n_races": int(preds[GROUP_COLS].drop_duplicates().shape[0]),
        "n_rows": int(len(preds)),
        "residual_std": calib["residual_std"],
        "residual_std_by_bucket": calib["residual_std_by_bucket"],
        "dnf_position_prior": round(float(preds["dnf_prior"].mean()), 3),
        "dev": summarise(dev_preds, grid_prob_table),
        "holdout": summarise(holdout_preds, grid_prob_table),
        "by_season": {str(season): summarise(g, grid_prob_table) for season, g in preds.groupby("season")},
    }
    if not holdout_preds.empty:
        report["holdout"]["bootstrap_ci"] = bootstrap_diff_ci(holdout_preds, grid_prob_table)
    return report


def holdout_predictions_with_circuit(df: pd.DataFrame, delta_params: dict | None = None,
                                     dnf_params: dict | None = None, min_train_races: int = 15,
                                     n_sims: int = 10_000, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Same walk-forward + dev-fit-calibration pipeline as
    run_full_evaluation() (byte-for-byte: same dev_df, same calibration,
    same Monte Carlo), but returns the raw per-row HOLDOUT predictions
    (merged with circuit_id from `df`) and the grid_prob_table, instead of
    a summary dict -- so a caller can slice accuracy by circuit class
    (dashboard's Street circuits view) using the exact same summarise()/
    bootstrap_diff_ci() this module already reports with. Read-only reuse
    of this module's own private pipeline pieces; doesn't change anything
    run_full_evaluation() itself reports."""
    dev_df, _ = dev_holdout_split(df)
    grid_prob_table = empirical_grid_probs(dev_df)

    preds_raw, dnf_samples_by_race = _walk_forward_raw(df, min_train_races, delta_params, dnf_params)
    calib = _fit_dev_calibration(dev_df, delta_params, dnf_params, min_train_races, n_sims, seed)

    preds = _run_monte_carlo(preds_raw, dnf_samples_by_race, calib["residual_std"],
                             bucket_stds=calib["residual_std_by_bucket"], n_sims=n_sims, seed=seed)
    preds["p_win"] = calib["win_calibrator"].predict(preds["p_win"])
    preds["p_podium"] = calib["podium_calibrator"].predict(preds["p_podium"])
    preds = _rank_and_flag(preds)

    holdout_preds = preds[preds["season"] > DEV_SEASON_MAX]
    circuit_map = df[["season", "round", "circuit_id"]].drop_duplicates()
    holdout_preds = holdout_preds.merge(circuit_map, on=["season", "round"], how="left")
    return holdout_preds, grid_prob_table


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
             dnf_params: dict | None = None, fit_calibration: bool = True) -> tuple[Pipeline, Pipeline, dict]:
    """Fit delta + DNF pipelines on ALL of df -- no CV/holdout. This is the
    model actually used for real predictions, as opposed to the
    walk-forward copies fit inside evaluate() purely for honest metrics.
    meta's residual_std_by_bucket/win_calibrator/podium_calibrator are fit
    on dev (2022-2024) walk-forward CV ONLY, per _fit_dev_calibration()'s
    leakage rule -- set fit_calibration=False to skip that (much slower)
    step when only the point-prediction pipelines are needed (e.g. some
    tests)."""
    train_delta = df[df[TARGET_DNF] == 0].dropna(subset=[TARGET_DELTA])
    train_dnf = df.dropna(subset=[TARGET_DNF])
    delta_pipe = make_delta_pipeline(**(delta_params or {})).fit(_prep_delta(train_delta), train_delta[TARGET_DELTA])
    dnf_pipe = make_dnf_pipeline(**(dnf_params or {})).fit(_prep_dnf(train_dnf), train_dnf[TARGET_DNF])

    dnf_rows = df.loc[df[TARGET_DNF] == 1, TARGET_POS].dropna()
    meta = {
        "residual_std": _quick_residual_std(df, delta_params),
        "residual_std_by_bucket": None,
        "dnf_position_prior": float(dnf_rows.mean()) if len(dnf_rows) else float(df[TARGET_POS].max()),
        "dnf_position_samples": (dnf_rows.tolist() if len(dnf_rows) >= 5
                                 else [float(dnf_rows.mean()) if len(dnf_rows) else float(df[TARGET_POS].max())]),
        "win_calibrator": None,
        "podium_calibrator": None,
    }
    if fit_calibration:
        dev_df, _ = dev_holdout_split(df)
        try:
            calib = _fit_dev_calibration(dev_df, delta_params, dnf_params)
            meta["residual_std"] = calib["residual_std"]
            meta["residual_std_by_bucket"] = calib["residual_std_by_bucket"]
            meta["win_calibrator"] = calib["win_calibrator"]
            meta["podium_calibrator"] = calib["podium_calibrator"]
        except ValueError:
            pass  # not enough dev races for a single fold (e.g. a tiny synthetic test dataset)
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

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"delta_pipe": delta_pipe, "dnf_pipe": dnf_pipe, "meta": meta}, MODEL_BUNDLE_PATH)
    METRICS_PATH.write_text(json.dumps(report, indent=2))
    return report


def load_models():
    if not MODEL_BUNDLE_PATH.exists():
        return None, None, None
    bundle = joblib.load(MODEL_BUNDLE_PATH)
    return bundle["delta_pipe"], bundle["dnf_pipe"], bundle["meta"]


def ensure_trained(dataset_path) -> tuple[Pipeline, Pipeline, dict]:
    """Prefers the committed frozen model (see freeze_model()/
    load_frozen_model()) if one exists -- that's the one genuinely "live"
    model now, per the freeze. Falls back to a cached fast self-trained
    model, else fits fresh from the dataset at dataset_path (fast -- no
    walk-forward evaluation) and caches the result. The fast-train path
    exists so a fresh checkout/deploy before any freeze, or a test, is
    never stuck without a usable model."""
    if FROZEN_BUNDLE_PATH.exists():
        return load_frozen_model()
    delta_pipe, dnf_pipe, meta = load_models()
    if delta_pipe is not None:
        return delta_pipe, dnf_pipe, meta
    df = pd.read_parquet(dataset_path)
    delta_pipe, dnf_pipe, meta = fit_final(df)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"delta_pipe": delta_pipe, "dnf_pipe": dnf_pipe, "meta": meta}, MODEL_BUNDLE_PATH)
    return delta_pipe, dnf_pipe, meta


# --------------------------------------------------------------------------
# Freezing: a deliberate, rare, manual snapshot -- never done automatically
# by a test, a script's default path, or a CI job. Once frozen,
# ensure_trained() (and therefore the API route, the dashboard, and
# scripts/predict_next_race.py) always uses this exact committed model,
# not whatever a fresh self-train from the current dataset would produce.
# That's what makes the "live track record since {frozen_at}" (see
# compute_live_track_record()) a genuinely clean, un-touched-by-further-
# tuning test going forward.
# --------------------------------------------------------------------------

def _dataset_sha256(dataset_path) -> str:
    return hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest()


def freeze_model(df: pd.DataFrame, dataset_path, version: str, notes: str = "",
                 delta_params: dict | None = None, dnf_params: dict | None = None,
                 frozen_at: str | None = None) -> dict:
    """Trains the final delta/DNF pipelines (weights on ALL of df; Monte
    Carlo noise + P(win)/P(podium) calibration on dev/2022-2024 CV only,
    via fit_final()) and commits them as the frozen snapshot at
    FROZEN_BUNDLE_PATH/FROZEN_SPEC_PATH. Run via
    `python -m scripts.freeze_race_predictor`, not automatically."""
    delta_params = delta_params or DEFAULT_DELTA_PARAMS
    dnf_params = dnf_params or DEFAULT_DNF_PARAMS
    delta_pipe, dnf_pipe, meta = fit_final(df, delta_params, dnf_params)
    frozen_at = frozen_at or date.today().isoformat()

    FROZEN_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"delta_pipe": delta_pipe, "dnf_pipe": dnf_pipe, "meta": meta}, FROZEN_BUNDLE_PATH)

    spec = {
        "version": version,
        "frozen_at": frozen_at,
        "dataset_sha256": _dataset_sha256(dataset_path),
        "n_rows": int(len(df)),
        "delta_params": delta_params,
        "dnf_params": dnf_params,
        "residual_std": meta["residual_std"],
        "residual_std_by_bucket": meta["residual_std_by_bucket"],
        "dnf_position_prior": meta["dnf_position_prior"],
        "notes": notes,
    }
    FROZEN_SPEC_PATH.write_text(json.dumps(spec, indent=2))
    return spec


def load_frozen_model() -> tuple[Pipeline, Pipeline, dict]:
    if not FROZEN_BUNDLE_PATH.exists():
        raise FileNotFoundError(
            f"No frozen model at {FROZEN_BUNDLE_PATH} -- run "
            "`python -m scripts.freeze_race_predictor` first.")
    bundle = joblib.load(FROZEN_BUNDLE_PATH)
    return bundle["delta_pipe"], bundle["dnf_pipe"], bundle["meta"]


def frozen_model_spec() -> dict | None:
    """The committed spec.json (version, frozen_at, hyperparams, dataset
    hash, ...) if a model has been frozen, else None."""
    if not FROZEN_SPEC_PATH.exists():
        return None
    return json.loads(FROZEN_SPEC_PATH.read_text())


def predict_race(delta_pipe, dnf_pipe, race_features: pd.DataFrame, meta: dict,
                 n_sims: int = 10_000, seed: int | None = None) -> pd.DataFrame:
    """race_features: one row per driver for a single upcoming race (same
    columns as the training features, plus "driver" and "grid" for display).
    Monte Carlo simulates the race (grid-bucket-dependent noise if
    meta["residual_std_by_bucket"] is set) and returns race_features with
    predicted_position (unique 1..N, ranked by expected position),
    win_probability, podium_probability and points_probability added
    (win/podium isotonic-calibrated if meta has those calibrators),
    sorted by predicted_position.
    """
    out = race_features.copy()
    out["delta_pred"] = delta_pipe.predict(_prep_delta(out))
    out["p_dnf"] = dnf_pipe.predict_proba(_prep_dnf(out))[:, 1]
    bucket_stds = meta.get("residual_std_by_bucket")
    std = (grid_bucket_residual_std_array(out["grid"], bucket_stds, meta["residual_std"])
          if bucket_stds else meta["residual_std"])
    sim = simulate_positions(out["grid"], out["delta_pred"], out["p_dnf"],
                             meta["dnf_position_samples"], std, n_sims=n_sims, seed=seed)
    win_cal, podium_cal = meta.get("win_calibrator"), meta.get("podium_calibrator")
    out["win_probability"] = win_cal.predict(sim["p_win"]) if win_cal is not None else sim["p_win"]
    out["podium_probability"] = podium_cal.predict(sim["p_podium"]) if podium_cal is not None else sim["p_podium"]
    out["points_probability"] = sim["p_points"]
    out["expected_position"] = sim["expected_position"]
    out = rank_within_race(out, "expected_position", "predicted_position", group_cols=None)
    return out.sort_values("predicted_position").reset_index(drop=True)


# --------------------------------------------------------------------------
# Live track record since the freeze -- the clean, genuinely prospective
# test: only predictions/{year}.csv rows logged at/after frozen_at, which by
# construction were made by the frozen model (predict_next_race.py loads
# only that model -- see load_frozen_model()), scored against real results.
# --------------------------------------------------------------------------

def compute_live_track_record(frozen_at: str) -> dict | None:
    """Reads every predictions/{year}.csv, keeps rows predicted at/after
    frozen_at, and scores the ones with a real result in yet. Returns None
    if nothing has been predicted since the freeze yet (a fresh freeze, or
    before the next scheduled prediction run) -- callers should render that
    as "no races yet", not fabricate a metric from zero rows."""
    if not PREDICTIONS_DIR.exists():
        return None
    # Exactly "{year}.csv" (v1's own log) -- NOT "*.csv", which would also
    # sweep up predict_staged.py's "{year}_quali_{stage}.csv" /
    # "{year}_race_v2_{stage}.csv" files living in the same directory.
    # Those share this exact column schema (predicted_position,
    # actual_position, ...), so a bare "*.csv" wouldn't even error -- it
    # would silently double-count races between v1 and v2's logs.
    frames = [pd.read_csv(p) for p in sorted(PREDICTIONS_DIR.glob("[12][0-9][0-9][0-9].csv"))]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return None

    all_preds = pd.concat(frames, ignore_index=True)
    all_preds["predicted_at"] = pd.to_datetime(all_preds["predicted_at"], utc=True)
    cutoff = pd.Timestamp(frozen_at)
    cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    live = all_preds[all_preds["predicted_at"] >= cutoff]
    if live.empty:
        return None

    out = {"frozen_at": frozen_at, "n_races_predicted": int(live["gp"].nunique())}
    scored = live.dropna(subset=["actual_position"])
    out["n_races_scored"] = int(scored["gp"].nunique())
    out["n_rows_scored"] = int(len(scored))
    if scored.empty:
        return out

    out["model_mae"] = round(float((scored["predicted_position"] - scored["actual_position"]).abs().mean()), 3)
    out["grid_mae"] = round(float((scored["grid"] - scored["actual_position"]).abs().mean()), 3)
    pred_winners = scored[scored["predicted_position"] == 1]
    out["model_winner_accuracy"] = (round(float((pred_winners["actual_position"] == 1).mean()), 3)
                                    if not pred_winners.empty else None)
    if "win_probability" in scored:
        actual_win = (scored["actual_position"] == 1).astype(int)
        out["win_brier"] = round(float(brier_score_loss(actual_win, scored["win_probability"].clip(1e-6, 1 - 1e-6))), 4)
    if "podium_probability" in scored:
        actual_podium = (scored["actual_position"] <= 3).astype(int)
        out["podium_brier"] = round(float(brier_score_loss(
            actual_podium, scored["podium_probability"].clip(1e-6, 1 - 1e-6))), 4)
    if "points_probability" in scored:
        actual_points = (scored["actual_position"] <= 10).astype(int)
        out["points_brier"] = round(float(brier_score_loss(
            actual_points, scored["points_probability"].clip(1e-6, 1 - 1e-6))), 4)
    return out
