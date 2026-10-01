"""Race outcome predictor v2: v1's positions-gained regressor + DNF
classifier (app.models.race_predictor), extended with circuit-history
features on the position model, plus the ability to sample the grid from
the qualifying predictor's own Monte Carlo distribution instead of a fixed
grid -- the pre-qualifying prediction case.

Reuses v1's DNF model and several stateless utilities by import (not
duplication): v1 is frozen and this module never touches its source or its
frozen artifact, it just calls the same pure functions. Only the position
("delta") model gets new features here; see FEATURES_DELTA for the
v1-plus-circuit-history feature list.

Grid handling (predict_race_v2()):
- Post-qualifying (grid known): works exactly like v1 -- grid is a fixed
  per-driver value, broadcast across every Monte Carlo run.
- Pre-qualifying (grid not known yet): each Monte Carlo run instead draws
  its own grid from the qualifying predictor's own simulated distribution
  (one full valid 1..N qualifying order per run, from
  quali_predictor.simulate_quali_positions(..., return_ranks=True)) --
  propagating qualifying uncertainty into the race prediction run by run,
  instead of collapsing it to a single point estimate first.

Evaluation protocol: identical discipline to v1 (dev_holdout_split,
select_hyperparams-style dev-only guard, one holdout pass, bootstrap CIs)
using the REAL grid throughout -- the same conditions v1 was evaluated
under, so the v2-vs-v1 comparison in the report is apples to apples. The
quali-grid-sampling capability is a prediction-time feature for live use
(see tests for its own direct validation), not something this walk-forward
evaluation exercises -- backtesting two uncertainty-propagating models in
sync is a separate, much larger undertaking than this round's scope.

Trained on data/model/race_dataset.parquet. Pure functions -- no FastF1.
"""
from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from app.config import FROZEN_MODEL_DIR, MODEL_DIR, PREDICTIONS_DIR
from app.models import race_predictor as v1

# --------------------------------------------------------------------------
# Position ("delta") model features: v1's feature set plus circuit history.
# The DNF model is reused unchanged from v1 (v1.NUM_DNF/make_dnf_pipeline/
# _prep_dnf) -- the user's spec for v2 is "v1 features + circuit history
# features" on the position model specifically.
# --------------------------------------------------------------------------

NUM_DELTA = v1.NUM_DELTA + [
    "driver_circuit_avg_quali_3", "driver_circuit_avg_finish_3", "driver_circuit_last_quali",
    "driver_circuit_races_here",
    "team_circuit_avg_quali_3", "team_circuit_avg_finish_3", "team_circuit_last_quali",
    "team_circuit_races_here",
]
BOOL_DELTA = v1.BOOL_DELTA + [
    "circuit_new_or_changed", "driver_circuit_hist_pre_reg_change", "team_circuit_hist_pre_reg_change",
]
CAT_DELTA = v1.CAT_DELTA
FEATURES_DELTA = NUM_DELTA + BOOL_DELTA + CAT_DELTA
TARGET_DELTA = v1.TARGET_DELTA
TARGET_DNF = v1.TARGET_DNF
TARGET_POS = v1.TARGET_POS
TARGET_PTS = v1.TARGET_PTS
GROUP_COLS = v1.GROUP_COLS
DEV_SEASON_MAX = v1.DEV_SEASON_MAX

# Chosen via select_hyperparams() walk-forward CV on 2022-2024 only.
DEFAULT_DELTA_PARAMS = {"max_leaf_nodes": 7, "min_samples_leaf": 30,
                        "learning_rate": 0.05, "max_iter": 500, "l2_regularization": 1.0}
DEFAULT_DNF_PARAMS = v1.DEFAULT_DNF_PARAMS  # DNF model/features unchanged from v1
DELTA_HP_GRID = [
    {"max_leaf_nodes": 7, "min_samples_leaf": 60, "learning_rate": 0.05, "max_iter": 500, "l2_regularization": 1.0},
    {"max_leaf_nodes": 7, "min_samples_leaf": 30, "learning_rate": 0.05, "max_iter": 500, "l2_regularization": 1.0},
    {"max_leaf_nodes": 15, "min_samples_leaf": 30, "learning_rate": 0.05, "max_iter": 500, "l2_regularization": 1.0},
]

MODEL_BUNDLE_PATH = MODEL_DIR / "race_predictor_v2_bundle.joblib"
METRICS_PATH = MODEL_DIR / "race_predictor_v2_metrics.json"
FROZEN_BUNDLE_PATH = FROZEN_MODEL_DIR / "race_v2_model.joblib"
FROZEN_SPEC_PATH = FROZEN_MODEL_DIR / "race_v2_spec.json"


def _prep_delta(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for c in NUM_DELTA:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype(float) if c in X else np.nan
        if X[c].notna().sum() == 0:
            X[c] = 0.0  # see v1._prep's note: avoids a HistGradientBoosting binning crash
    for c in BOOL_DELTA:
        X[c] = X[c].fillna(False).astype(float) if c in X else 0.0
    for c in CAT_DELTA:
        X[c] = X[c].fillna("unknown").astype(str) if c in X else "unknown"
    return X[FEATURES_DELTA]


_prep_dnf = v1._prep_dnf  # unchanged feature set -- reuse v1's prep directly


def make_delta_pipeline(**params) -> Pipeline:
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_DELTA),
        ("num", "passthrough", NUM_DELTA + BOOL_DELTA),
    ])
    reg = HistGradientBoostingRegressor(random_state=42, early_stopping=True, validation_fraction=0.15,
                                        n_iter_no_change=20, **{**DEFAULT_DELTA_PARAMS, **params})
    return Pipeline([("pre", pre), ("reg", reg)])


make_dnf_pipeline = v1.make_dnf_pipeline  # unchanged -- reuse v1's DNF model directly

# Stateless utilities reused directly from v1 (pure functions, no state tied
# to v1's specific frozen weights -- safe to share, doesn't touch v1).
rank_within_race = v1.rank_within_race
race_sequence = v1.race_sequence
time_based_splits = v1.time_based_splits
dev_holdout_split = v1.dev_holdout_split
grid_bucket = v1.grid_bucket
residual_std_by_grid_bucket = v1.residual_std_by_grid_bucket
grid_bucket_residual_std_array = v1.grid_bucket_residual_std_array
empirical_grid_probs = v1.empirical_grid_probs
baseline_grid_probs = v1.baseline_grid_probs


# --------------------------------------------------------------------------
# Monte Carlo race simulation -- generalizes v1's simulate_positions() to
# accept grid either as a fixed per-driver Series (broadcast across every
# run, post-qualifying) or as a full (n_sims, n_drivers) array (one
# different valid grid realization per run, pre-qualifying -- see
# predict_race_v2()).
# --------------------------------------------------------------------------

def simulate_positions(grid, delta_pred: pd.Series, p_dnf: pd.Series, dnf_position_samples: np.ndarray,
                       residual_std, n_sims: int = 10_000, seed: int | None = None) -> dict[str, np.ndarray]:
    n = len(delta_pred)
    rng = np.random.default_rng(seed)
    if isinstance(grid, np.ndarray) and grid.ndim == 2:
        grid_a = grid.astype(float)  # (n_sims, n) -- one grid realization per run
    else:
        grid_series = grid if isinstance(grid, pd.Series) else pd.Series(grid)
        grid_a = grid_series.to_numpy(dtype=float)[None, :]  # (1, n) -- broadcasts across runs
    delta_a = delta_pred.to_numpy(dtype=float)
    p_dnf_a = np.clip(p_dnf.to_numpy(dtype=float), 0.0, 1.0)
    residual_std_a = np.maximum(np.broadcast_to(np.asarray(residual_std, dtype=float), (n,)), 1e-6)

    dnf_draw = rng.random((n_sims, n)) < p_dnf_a[None, :]
    noise = rng.normal(0.0, 1.0, size=(n_sims, n)) * residual_std_a[None, :]
    finisher_raw = grid_a + delta_a[None, :] + noise
    samples = dnf_position_samples if len(dnf_position_samples) else np.array([np.nanmax(grid_a)])
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
# Metrics (identical shape to v1's, duplicated since they key off this
# module's own preds columns -- see v1 for the canonical documented version)
# --------------------------------------------------------------------------

def _race_metrics(preds: pd.DataFrame, pos_col: str) -> dict:
    from scipy.stats import spearmanr
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


def _points_f1(y_true, y_pred):
    from sklearn.metrics import f1_score
    if y_true.nunique() < 2:
        return None
    return round(float(f1_score(y_true, y_pred)), 3)


def _prob_metrics(actual: pd.Series, p: pd.Series) -> dict:
    from sklearn.metrics import log_loss
    if actual.nunique() < 2:
        return {"brier": None, "logloss": None}
    p = p.clip(1e-6, 1 - 1e-6)
    return {"brier": round(float(brier_score_loss(actual, p)), 4),
           "logloss": round(float(log_loss(actual, p, labels=[0, 1])), 4)}


def summarise(preds: pd.DataFrame, grid_prob_table: pd.DataFrame) -> dict:
    if preds.empty:
        return {"n_races": 0, "n_rows": 0}
    out = {"n_races": int(preds[GROUP_COLS].drop_duplicates().shape[0]), "n_rows": int(len(preds))}

    point_metrics = {}
    for method in ("model", "baseline_grid"):
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
    return out


# --------------------------------------------------------------------------
# Walk-forward evaluation (same structure as v1: _walk_forward_raw ->
# _fit_dev_calibration (bucket noise + isotonic, dev-only) -> apply
# everywhere). Grid is always the REAL grid here -- see the module
# docstring on why the quali-sampling mode isn't backtested this way.
# --------------------------------------------------------------------------

def _walk_forward_raw(df: pd.DataFrame, min_train_races: int, delta_params: dict | None,
                      dnf_params: dict | None):
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
        key = (test[GROUP_COLS[0]].iloc[0], test[GROUP_COLS[1]].iloc[0])
        dnf_samples_by_race[key] = dnf_samples
        rows.append(out)

    if not rows:
        raise ValueError(f"Not enough races for a single fold (need > {min_train_races})")
    return pd.concat(rows, ignore_index=True), dnf_samples_by_race


def _run_monte_carlo(preds: pd.DataFrame, dnf_samples_by_race, residual_std: float,
                     bucket_stds: dict | None = None, n_sims: int = 10_000, seed: int = 42) -> pd.DataFrame:
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
    preds["model_points"] = preds["model_pos"] <= 10
    preds["baseline_grid_points"] = preds["baseline_grid_pos"] <= 10
    return preds


def evaluate(df: pd.DataFrame, min_train_races: int = 15, delta_params: dict | None = None,
            dnf_params: dict | None = None, n_sims: int = 10_000, seed: int = 42):
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


def _fit_dev_calibration(dev_df: pd.DataFrame, delta_params: dict | None = None, dnf_params: dict | None = None,
                         min_train_races: int = 15, n_sims: int = 10_000, seed: int = 42) -> dict:
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
    win_calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(dev_sim["p_win"], actual_win)
    podium_calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(
        dev_sim["p_podium"], actual_podium)
    return {"residual_std": round(residual_std, 4),
           "residual_std_by_bucket": {k: round(v, 4) for k, v in bucket_stds.items()},
           "win_calibrator": win_calibrator, "podium_calibrator": podium_calibrator}


def run_full_evaluation(df: pd.DataFrame, delta_params: dict | None = None, dnf_params: dict | None = None,
                        min_train_races: int = 15, n_sims: int = 10_000, seed: int = 42) -> dict:
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
        report["holdout"]["bootstrap_ci"] = bootstrap_diff_ci_vs_grid(holdout_preds, grid_prob_table)
        report["holdout"]["bootstrap_ci_vs_v1"] = bootstrap_diff_ci_vs_v1(df, holdout_preds, min_train_races, n_sims, seed)
    return report


# --------------------------------------------------------------------------
# Bootstrap CIs: v2 vs grid baseline (same mechanism as v1's
# bootstrap_diff_ci), and v2 vs v1 (v1's own walk-forward out-of-sample
# predictions on the identical holdout races, via v1.run_full_evaluation --
# read-only reuse, v1's source/frozen artifact are untouched).
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


def bootstrap_diff_ci_vs_grid(preds: pd.DataFrame, grid_prob_table: pd.DataFrame, n_boot: int = 2000,
                              ci: float = 0.95, seed: int = 42) -> dict:
    race_data = _per_race_bootstrap_data(preds, grid_prob_table)
    n = len(race_data)
    rng = np.random.default_rng(seed)
    mae_diff = np.empty(n_boot)
    points_brier_diff = np.empty(n_boot)
    win_brier_diff = np.empty(n_boot)
    for b in range(n_boot):
        sample = [race_data[i] for i in rng.integers(0, n, size=n)]
        mae_diff[b] = np.mean([r["model_mae"] for r in sample]) - np.mean([r["grid_mae"] for r in sample])
        actual_points = np.concatenate([r["actual_points"] for r in sample])
        model_p_points = np.clip(np.concatenate([r["model_p_points"] for r in sample]), 1e-6, 1 - 1e-6)
        base_p_points = np.clip(np.concatenate([r["base_p_points"] for r in sample]), 1e-6, 1 - 1e-6)
        points_brier_diff[b] = (brier_score_loss(actual_points, model_p_points)
                                - brier_score_loss(actual_points, base_p_points))
        actual_win = np.concatenate([r["actual_win"] for r in sample])
        model_p_win = np.clip(np.concatenate([r["model_p_win"] for r in sample]), 1e-6, 1 - 1e-6)
        base_p_win = np.clip(np.concatenate([r["base_p_win"] for r in sample]), 1e-6, 1 - 1e-6)
        win_brier_diff[b] = brier_score_loss(actual_win, model_p_win) - brier_score_loss(actual_win, base_p_win)
    alpha = (1 - ci) / 2

    def _s(arr):
        return {"mean_diff": round(float(arr.mean()), 4), "ci_low": round(float(np.percentile(arr, 100 * alpha)), 4),
               "ci_high": round(float(np.percentile(arr, 100 * (1 - alpha))), 4)}
    return {"n_boot": n_boot, "ci": ci, "position_mae_diff": _s(mae_diff),
           "points_brier_diff": _s(points_brier_diff), "win_brier_diff": _s(win_brier_diff)}


def bootstrap_diff_ci_vs_v1(df: pd.DataFrame, v2_holdout_preds: pd.DataFrame, min_train_races: int,
                            n_sims: int, seed: int, n_boot: int = 2000, ci: float = 0.95) -> dict:
    """v2 minus v1's OWN walk-forward out-of-sample predictions (v1's
    private pieces, read-only) on the identical holdout races -- the
    apples-to-apples comparison the "vs v1" part of the report needs."""
    v1_preds_raw, v1_dnf_samples = v1._walk_forward_raw(df, min_train_races, v1.DEFAULT_DELTA_PARAMS,
                                                        v1.DEFAULT_DNF_PARAMS)
    dev_df, _ = v1.dev_holdout_split(df)
    v1_calib = v1._fit_dev_calibration(dev_df, v1.DEFAULT_DELTA_PARAMS, v1.DEFAULT_DNF_PARAMS,
                                       min_train_races, n_sims, seed)
    v1_preds = v1._run_monte_carlo(v1_preds_raw, v1_dnf_samples, v1_calib["residual_std"],
                                   bucket_stds=v1_calib["residual_std_by_bucket"], n_sims=n_sims, seed=seed)
    v1_preds["p_win"] = v1_calib["win_calibrator"].predict(v1_preds["p_win"])
    v1_preds = v1._rank_and_flag(v1_preds)
    v1_holdout = v1_preds[v1_preds["season"] > DEV_SEASON_MAX]

    v1_by_race = {key: race for key, race in v1_holdout.groupby(GROUP_COLS, sort=False)}
    races = []
    for key, race in v2_holdout_preds.groupby(GROUP_COLS, sort=False):
        v1_race = v1_by_race.get(key)
        if v1_race is None:
            continue
        v1_merged = race[["driver", TARGET_POS]].merge(
            v1_race[["driver", "model_pos", "p_win"]], on="driver", how="inner")
        if v1_merged.empty:
            continue
        races.append({
            "v2_mae": mean_absolute_error(race[TARGET_POS], race["model_pos"]),
            "v1_mae": mean_absolute_error(v1_merged[TARGET_POS], v1_merged["model_pos"]),
            "actual_win": (v1_merged[TARGET_POS] == 1).to_numpy(dtype=float),
            "v2_p_win": race.set_index("driver").loc[v1_merged["driver"], "p_win"].to_numpy(),
            "v1_p_win": v1_merged["p_win"].to_numpy(),
        })
    n = len(races)
    if n == 0:
        return {"n_boot": 0, "ci": ci, "note": "no overlapping holdout races with v1"}
    rng = np.random.default_rng(seed + 1)
    mae_diff = np.empty(n_boot)
    win_brier_diff = np.empty(n_boot)
    for b in range(n_boot):
        sample = [races[i] for i in rng.integers(0, n, size=n)]
        mae_diff[b] = np.mean([r["v2_mae"] for r in sample]) - np.mean([r["v1_mae"] for r in sample])
        actual = np.concatenate([r["actual_win"] for r in sample])
        v2_p = np.clip(np.concatenate([r["v2_p_win"] for r in sample]), 1e-6, 1 - 1e-6)
        v1_p = np.clip(np.concatenate([r["v1_p_win"] for r in sample]), 1e-6, 1 - 1e-6)
        win_brier_diff[b] = brier_score_loss(actual, v2_p) - brier_score_loss(actual, v1_p)
    alpha = (1 - ci) / 2

    def _s(arr):
        return {"mean_diff": round(float(arr.mean()), 4), "ci_low": round(float(np.percentile(arr, 100 * alpha)), 4),
               "ci_high": round(float(np.percentile(arr, 100 * (1 - alpha))), 4)}
    return {"n_boot": n_boot, "ci": ci, "n_races": n,
           "position_mae_diff_v2_minus_v1": _s(mae_diff), "win_brier_diff_v2_minus_v1": _s(win_brier_diff)}


# --------------------------------------------------------------------------
# Hyperparameter selection -- dev seasons (<=2024) only, same structural
# guard as v1's select_hyperparams().
# --------------------------------------------------------------------------

def select_hyperparams(dev_df: pd.DataFrame, grid: list[dict] | None = None,
                       min_train_races: int = 10, n_sims: int = 200) -> dict:
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


# --------------------------------------------------------------------------
# Final fit + single-race prediction + freezing.
# --------------------------------------------------------------------------

def fit_final(df: pd.DataFrame, delta_params: dict | None = None, dnf_params: dict | None = None,
             fit_calibration: bool = True) -> tuple[Pipeline, Pipeline, dict]:
    train_delta = df[df[TARGET_DNF] == 0].dropna(subset=[TARGET_DELTA])
    train_dnf = df.dropna(subset=[TARGET_DNF])
    delta_pipe = make_delta_pipeline(**(delta_params or {})).fit(_prep_delta(train_delta), train_delta[TARGET_DELTA])
    dnf_pipe = make_dnf_pipeline(**(dnf_params or {})).fit(_prep_dnf(train_dnf), train_dnf[TARGET_DNF])

    dnf_rows = df.loc[df[TARGET_DNF] == 1, TARGET_POS].dropna()
    meta = {
        "residual_std": float((train_delta[TARGET_DELTA] - delta_pipe.predict(_prep_delta(train_delta))).std()) or 1.0,
        "residual_std_by_bucket": None,
        "dnf_position_prior": float(dnf_rows.mean()) if len(dnf_rows) else float(df[TARGET_POS].max()),
        "dnf_position_samples": (dnf_rows.tolist() if len(dnf_rows) >= 5
                                 else [float(dnf_rows.mean()) if len(dnf_rows) else float(df[TARGET_POS].max())]),
        "win_calibrator": None, "podium_calibrator": None,
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
            pass
    return delta_pipe, dnf_pipe, meta


def train(df: pd.DataFrame, delta_params: dict | None = None, dnf_params: dict | None = None) -> dict:
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


def _dataset_sha256(dataset_path) -> str:
    return hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest()


def freeze_model(df: pd.DataFrame, dataset_path, version: str, notes: str = "",
                 delta_params: dict | None = None, dnf_params: dict | None = None,
                 frozen_at: str | None = None) -> dict:
    delta_params = delta_params or DEFAULT_DELTA_PARAMS
    dnf_params = dnf_params or DEFAULT_DNF_PARAMS
    delta_pipe, dnf_pipe, meta = fit_final(df, delta_params, dnf_params)
    frozen_at = frozen_at or date.today().isoformat()

    FROZEN_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"delta_pipe": delta_pipe, "dnf_pipe": dnf_pipe, "meta": meta}, FROZEN_BUNDLE_PATH)

    spec = {"version": version, "frozen_at": frozen_at, "dataset_sha256": _dataset_sha256(dataset_path),
           "n_rows": int(len(df)), "delta_params": delta_params, "dnf_params": dnf_params,
           "residual_std": meta["residual_std"], "residual_std_by_bucket": meta["residual_std_by_bucket"],
           "dnf_position_prior": meta["dnf_position_prior"], "notes": notes}
    FROZEN_SPEC_PATH.write_text(json.dumps(spec, indent=2))
    return spec


def load_frozen_model() -> tuple[Pipeline, Pipeline, dict]:
    if not FROZEN_BUNDLE_PATH.exists():
        raise FileNotFoundError(f"No frozen race v2 model at {FROZEN_BUNDLE_PATH} -- run "
                               "`python -m scripts.freeze_race_predictor_v2` first.")
    bundle = joblib.load(FROZEN_BUNDLE_PATH)
    return bundle["delta_pipe"], bundle["dnf_pipe"], bundle["meta"]


def frozen_model_spec() -> dict | None:
    if not FROZEN_SPEC_PATH.exists():
        return None
    return json.loads(FROZEN_SPEC_PATH.read_text())


def predict_race_v2(delta_pipe, dnf_pipe, race_features: pd.DataFrame, meta: dict,
                    quali_pipe=None, quali_meta: dict | None = None, quali_features: pd.DataFrame | None = None,
                    n_sims: int = 10_000, seed: int | None = None) -> pd.DataFrame:
    """race_features: one row per driver, v2's feature columns (v1's +
    circuit history) plus "driver" and "grid" for display. If grid is fully
    known (post-qualifying), pass quali_pipe=None and this behaves exactly
    like v1's predict_race(). Pre-qualifying, pass quali_pipe/quali_meta/
    quali_features (same driver order as race_features) -- each of the
    n_sims runs then uses its OWN simulated qualifying order as that run's
    grid, instead of a single fixed estimate.
    """
    from app.models.quali_predictor import simulate_quali_positions, _prep as _prep_quali

    out = race_features.copy()
    out["delta_pred"] = delta_pipe.predict(_prep_delta(out))
    out["p_dnf"] = dnf_pipe.predict_proba(_prep_dnf(out))[:, 1]
    bucket_stds = meta.get("residual_std_by_bucket")

    if quali_pipe is not None:
        q = quali_features.copy()
        q["delta_pred"] = quali_pipe.predict(_prep_quali(q))
        baseline = q["driver_rolling_quali_position_3"]
        if baseline.isna().any():
            baseline = baseline.fillna(baseline.max() if baseline.notna().any() else float(len(q)))
        quali_sim = simulate_quali_positions(baseline, q["delta_pred"], quali_meta["residual_std"],
                                             n_sims=n_sims, seed=seed, return_ranks=True)
        grid_samples = quali_sim["ranks"].astype(float)  # (n_sims, n_drivers), aligned to race_features' row order
        std = (grid_bucket_residual_std_array(out["grid"], bucket_stds, meta["residual_std"])
              if bucket_stds else meta["residual_std"])
        sim = simulate_positions(grid_samples, out["delta_pred"], out["p_dnf"], meta["dnf_position_samples"],
                                 std, n_sims=n_sims, seed=seed)
    else:
        std = (grid_bucket_residual_std_array(out["grid"], bucket_stds, meta["residual_std"])
              if bucket_stds else meta["residual_std"])
        sim = simulate_positions(out["grid"], out["delta_pred"], out["p_dnf"], meta["dnf_position_samples"],
                                 std, n_sims=n_sims, seed=seed)

    win_cal, podium_cal = meta.get("win_calibrator"), meta.get("podium_calibrator")
    out["win_probability"] = win_cal.predict(sim["p_win"]) if win_cal is not None else sim["p_win"]
    out["podium_probability"] = podium_cal.predict(sim["p_podium"]) if podium_cal is not None else sim["p_podium"]
    out["points_probability"] = sim["p_points"]
    out["expected_position"] = sim["expected_position"]
    out = rank_within_race(out, "expected_position", "predicted_position", group_cols=None)
    return out.sort_values("predicted_position").reset_index(drop=True)


def compute_live_track_record(frozen_at: str, stage: str) -> dict | None:
    """Reads every predictions/{year}_race_v2_{stage}.csv (scripts/
    predict_staged.py's output), keeps rows predicted at/after frozen_at,
    and scores the ones with a real result in yet against the grid
    baseline (only meaningful at stage="post_quali", where grid is a real
    pre-race value rather than NaN) and, paired by (gp, driver), against
    v1's own prediction for the same race from predictions/{year}.csv --
    v1 only ever predicts post-quali, so this comparison is also only
    populated at that stage. None if nothing's been predicted at this
    stage since the freeze yet."""
    files = sorted(PREDICTIONS_DIR.glob(f"*_race_v2_{stage}.csv")) if PREDICTIONS_DIR.exists() else []
    frames = [pd.read_csv(p) for p in files]
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

    out = {"frozen_at": frozen_at, "stage": stage, "n_races_predicted": int(live["gp"].nunique())}
    scored = live.dropna(subset=["actual_position"])
    out["n_races_scored"] = int(scored["gp"].nunique())
    out["n_rows_scored"] = int(len(scored))
    if scored.empty:
        return out

    out["model_mae"] = round(float((scored["predicted_position"] - scored["actual_position"]).abs().mean()), 3)
    grid_known = scored.dropna(subset=["grid"])
    if not grid_known.empty:
        out["grid_mae"] = round(float((grid_known["grid"] - grid_known["actual_position"]).abs().mean()), 3)
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

    v1_frames = []
    for year in {int(pd.to_datetime(d).year) for d in live["predicted_at"]}:
        p = PREDICTIONS_DIR / f"{year}.csv"
        if p.exists():
            v1_frames.append(pd.read_csv(p))
    if v1_frames:
        v1_all = pd.concat(v1_frames, ignore_index=True).dropna(subset=["actual_position"])
        paired = scored.merge(v1_all[["gp", "driver", "predicted_position", "actual_position"]],
                              on=["gp", "driver"], suffixes=("_v2", "_v1"))
        if not paired.empty:
            out["n_rows_vs_v1"] = int(len(paired))
            out["v2_mae_vs_v1_pairs"] = round(float((paired["predicted_position_v2"]
                                                      - paired["actual_position_v2"]).abs().mean()), 3)
            out["v1_mae_vs_v1_pairs"] = round(float((paired["predicted_position_v1"]
                                                      - paired["actual_position_v1"]).abs().mean()), 3)
    return out
