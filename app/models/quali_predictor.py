"""Qualifying predictor: positions-gained-vs-rolling-form regressor, Monte
Carlo simulated into P(pole)/P(top 3)/P(Q3) per driver, from pre-session
features only (rolling qualifying form, circuit quali history, practice
pace when available).

Same framing as the race predictor (app/models/race_predictor.py): predict
a small delta on top of an already-strong baseline (the driver's own recent
qualifying form) rather than the raw position outright, with heavy
regularization so the model shrinks to "no change" when there's nothing to
add. target_quali_delta = quali_position - driver_rolling_quali_position_3
(the most current rolling baseline), trained only on rows where that
baseline exists (a driver's very first race in the data has no prior form
to take a delta from).

Two "modes" aren't two separate models: fp_best_gap_s/fp_long_run_gap_s
(practice pace) are NaN before FP2/FP3 happens and populated after.
HistGradientBoosting handles missing values natively (a NaN feature routes
to whichever split side the training data says is better), so the exact
same trained pipeline produces a pre-weekend-quality prediction when those
columns are NaN and a post-practice-quality one once they're populated --
no mode-switching logic needed anywhere in this module or its callers.

Evaluation protocol (same as the race predictor): dev_holdout_split,
select_hyperparams, run_full_evaluation. All design decisions are made
with walk-forward CV on 2022-2024 ("dev") only; 2025-2026 ("holdout") is
evaluated exactly once. Baselines: rolling quali position (last 5), and
the single most recent prior edition's quali position at this circuit
("last year").

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
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from app.config import FROZEN_MODEL_DIR, MODEL_DIR

# --------------------------------------------------------------------------
# Features. All strictly pre-session: rolling/circuit-history columns are
# shift(1)-before-rolling in the dataset builder (never include this race),
# and fp_best_gap_s/fp_long_run_gap_s are this weekend's own practice but
# from BEFORE qualifying, not qualifying itself.
# --------------------------------------------------------------------------

NUM = ["driver_rolling_quali_position_3", "driver_rolling_quali_position_5",
      "team_rolling_quali_position_3", "teammate_quali_gap_trend_3", "team_rolling_pace_gap_3",
      "driver_circuit_avg_quali_3", "driver_circuit_last_quali", "driver_circuit_races_here",
      "team_circuit_avg_quali_3", "team_circuit_last_quali", "team_circuit_races_here",
      "fp_best_gap_s", "fp_long_run_gap_s"]
BOOL = ["circuit_new_or_changed", "reg_change_flag",
       "driver_circuit_hist_pre_reg_change", "team_circuit_hist_pre_reg_change"]
CAT = ["circuit_type"]
FEATURES = NUM + BOOL + CAT

TARGET_DELTA = "target_quali_delta"
TARGET_POS = "quali_position"  # ground truth -- this IS the thing being predicted
GROUP_COLS = ["season", "round"]
BASELINE_DELTA_COL = "driver_rolling_quali_position_3"  # what target_quali_delta is relative to

DEV_SEASON_MAX = 2024

# Chosen via select_hyperparams() walk-forward CV on 2022-2024 only -- see
# the module docstring's evaluation-protocol note and README.
DEFAULT_PARAMS = {"max_leaf_nodes": 7, "min_samples_leaf": 60,
                 "learning_rate": 0.05, "max_iter": 500, "l2_regularization": 1.0}
HP_GRID = [
    {"max_leaf_nodes": 7, "min_samples_leaf": 60, "learning_rate": 0.05, "max_iter": 500, "l2_regularization": 1.0},
    {"max_leaf_nodes": 7, "min_samples_leaf": 30, "learning_rate": 0.05, "max_iter": 500, "l2_regularization": 1.0},
    {"max_leaf_nodes": 15, "min_samples_leaf": 30, "learning_rate": 0.05, "max_iter": 500, "l2_regularization": 1.0},
]

MODEL_BUNDLE_PATH = MODEL_DIR / "quali_predictor_bundle.joblib"
METRICS_PATH = MODEL_DIR / "quali_predictor_metrics.json"
FROZEN_BUNDLE_PATH = FROZEN_MODEL_DIR / "quali_model.joblib"
FROZEN_SPEC_PATH = FROZEN_MODEL_DIR / "quali_spec.json"


def _prep(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for c in NUM:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype(float) if c in X else np.nan
        if X[c].notna().sum() == 0:
            # Same HistGradientBoosting binning crash as race_predictor.py's
            # _prep() guards against: a fully-missing column (e.g. fp_* in
            # an early walk-forward fold, before FP backfill existed for
            # those races) can't be binned. Safe fill: zero variance either
            # way.
            X[c] = 0.0
    for c in BOOL:
        X[c] = X[c].fillna(False).astype(float) if c in X else 0.0
    for c in CAT:
        X[c] = X[c].fillna("unknown").astype(str) if c in X else "unknown"
    return X[FEATURES]


def make_pipeline(**params) -> Pipeline:
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
        ("num", "passthrough", NUM + BOOL),
    ])
    reg = HistGradientBoostingRegressor(random_state=42, early_stopping=True, validation_fraction=0.15,
                                        n_iter_no_change=20, **{**DEFAULT_PARAMS, **params})
    return Pipeline([("pre", pre), ("reg", reg)])


def rank_within_race(df: pd.DataFrame, value_col: str, out_col: str,
                     group_cols: list[str] | None = GROUP_COLS) -> pd.DataFrame:
    """Same NaN-safe unique-rank logic as race_predictor.py's
    rank_within_race, PLUS a whole-group-NaN fallback it doesn't need: the
    "last year's quali position" baseline is 100% NaN for every 2022 race
    (no prior season exists yet), which would otherwise leave race_max
    itself NaN and crash the final astype(int). Duplicated rather than
    imported so this module doesn't depend on the frozen race predictor's
    internals changing."""
    df = df.copy()
    if group_cols:
        grp = df.groupby(group_cols)[value_col]
        race_max = grp.transform("max")
        group_size = grp.transform("size")
        safe_max = race_max.fillna(group_size)  # whole-group-NaN fallback: field size
        filled = df[value_col].fillna(safe_max + 1)
        df[out_col] = filled.groupby([df[c] for c in group_cols]).rank(method="first").astype(int)
    else:
        col_max = df[value_col].max()
        safe_max = col_max if pd.notna(col_max) else float(len(df))
        filled = df[value_col].fillna(safe_max + 1)
        df[out_col] = filled.rank(method="first").astype(int)
    return df


def race_sequence(df: pd.DataFrame) -> np.ndarray:
    races = df[GROUP_COLS].drop_duplicates().sort_values(GROUP_COLS).reset_index(drop=True)
    races["_race_seq"] = np.arange(len(races))
    return df.merge(races, on=GROUP_COLS, how="left")["_race_seq"].to_numpy()


def time_based_splits(df: pd.DataFrame, min_train_races: int = 15):
    seq = pd.Series(race_sequence(df), index=df.index)
    max_seq = int(seq.max())
    for k in range(min_train_races, max_seq + 1):
        train_idx = seq.index[seq < k]
        test_idx = seq.index[seq == k]
        if len(test_idx) == 0 or len(train_idx) == 0:
            continue
        yield train_idx, test_idx


def dev_holdout_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dev = df[df["season"] <= DEV_SEASON_MAX].reset_index(drop=True)
    holdout = df[df["season"] > DEV_SEASON_MAX].reset_index(drop=True)
    return dev, holdout


def add_quali_target(df: pd.DataFrame) -> pd.DataFrame:
    """target_quali_delta = quali_position - driver_rolling_quali_position_3
    (NaN, and so excluded from training, when a driver has no prior
    rolling-quali baseline yet -- their first race in the dataset)."""
    df = df.copy()
    df[TARGET_DELTA] = df[TARGET_POS] - df[BASELINE_DELTA_COL]
    return df


# --------------------------------------------------------------------------
# Monte Carlo simulation
# --------------------------------------------------------------------------

def simulate_quali_positions(baseline: pd.Series, delta_pred: pd.Series, residual_std: float,
                             n_sims: int = 10_000, seed: int | None = None) -> dict[str, np.ndarray]:
    """Each run: baseline + delta_pred + Normal(0, residual_std) noise,
    ranked within the run (argsort trick, same as race_predictor.py's
    simulate_positions) so every run is a valid unique 1..N qualifying
    order. P(Q3) = P(final position <= 10) -- F1's Q1/Q2/Q3 knockout
    format cuts to the top 10 after Q2."""
    n = len(baseline)
    rng = np.random.default_rng(seed)
    base_a = baseline.to_numpy(dtype=float)
    delta_a = delta_pred.to_numpy(dtype=float)
    noise = rng.normal(0.0, max(residual_std, 1e-6), size=(n_sims, n))
    raw = base_a[None, :] + delta_a[None, :] + noise

    order = np.argsort(raw, axis=1, kind="stable")
    ranks = np.empty_like(order)
    rows = np.arange(n_sims)[:, None]
    ranks[rows, order] = np.arange(1, n + 1)[None, :]

    return {
        "p_pole": (ranks == 1).mean(axis=0),
        "p_top3": (ranks <= 3).mean(axis=0),
        "p_q3": (ranks <= 10).mean(axis=0),
        "expected_position": ranks.mean(axis=0),
    }


# --------------------------------------------------------------------------
# Empirical rolling-position -> probability baseline (mirrors
# race_predictor.py's empirical_grid_probs, keyed on the rolling-5 baseline
# instead of grid since quali position isn't known pre-session). Fit once
# from dev seasons only.
# --------------------------------------------------------------------------

def empirical_rolling_probs(dev_df: pd.DataFrame, max_bucket: int = 20) -> pd.DataFrame:
    d = dev_df.dropna(subset=["driver_rolling_quali_position_5"]).copy()
    d["bucket"] = d["driver_rolling_quali_position_5"].round().clip(upper=max_bucket).astype(int)
    rows = []
    for bucket, g in d.groupby("bucket"):
        rows.append({
            "bucket": bucket, "n": len(g),
            "p_pole": (g[TARGET_POS] == 1).mean(),
            "p_top3": (g[TARGET_POS] <= 3).mean(),
            "p_q3": (g[TARGET_POS] <= 10).mean(),
        })
    return pd.DataFrame(rows).sort_values("bucket").reset_index(drop=True)


def baseline_rolling_probs(rolling5: pd.Series, table: pd.DataFrame, max_bucket: int = 20) -> pd.DataFrame:
    if table.empty:
        return pd.DataFrame({"p_pole": np.full(len(rolling5), np.nan),
                            "p_top3": np.full(len(rolling5), np.nan),
                            "p_q3": np.full(len(rolling5), np.nan)}, index=rolling5.index)
    bucket = rolling5.round().clip(upper=max_bucket)
    idx = table.set_index("bucket")
    fallback = {"p_pole": idx["p_pole"].min(), "p_top3": idx["p_top3"].min(), "p_q3": idx["p_q3"].min()}
    return pd.DataFrame({
        "p_pole": bucket.map(idx["p_pole"]).fillna(fallback["p_pole"]).to_numpy(),
        "p_top3": bucket.map(idx["p_top3"]).fillna(fallback["p_top3"]).to_numpy(),
        "p_q3": bucket.map(idx["p_q3"]).fillna(fallback["p_q3"]).to_numpy(),
    }, index=rolling5.index)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def _race_metrics(preds: pd.DataFrame, pos_col: str) -> dict:
    maes, spearmans, top3_hits, pole_hits = [], [], [], []
    for _, race in preds.groupby(GROUP_COLS):
        maes.append(mean_absolute_error(race[TARGET_POS], race[pos_col]))
        if race[TARGET_POS].nunique() > 1 and race[pos_col].nunique() > 1:
            spearmans.append(spearmanr(race[TARGET_POS], race[pos_col]).correlation)
        pred_pole = race.loc[race[pos_col] == 1]
        if not pred_pole.empty:
            actual_pos = pred_pole[TARGET_POS].iloc[0]
            top3_hits.append(actual_pos <= 3)
            pole_hits.append(actual_pos == 1)
    return {
        "position_mae": round(float(np.mean(maes)), 3) if maes else None,
        "spearman": round(float(np.nanmean(spearmans)), 3) if spearmans else None,
        "top3_hit_rate": round(float(np.mean(top3_hits)), 3) if top3_hits else None,
        "pole_accuracy": round(float(np.mean(pole_hits)), 3) if pole_hits else None,
    }


def _prob_metrics(actual: pd.Series, p: pd.Series) -> dict:
    if actual.nunique() < 2:
        return {"brier": None, "logloss": None}
    p = p.clip(1e-6, 1 - 1e-6)
    return {"brier": round(float(brier_score_loss(actual, p)), 4),
           "logloss": round(float(log_loss(actual, p, labels=[0, 1])), 4)}


def summarise(preds: pd.DataFrame, rolling_prob_table: pd.DataFrame) -> dict:
    # A handful of real rows have no valid qualifying time at all (no lap
    # set in Q1) -- quali_position is genuinely NaN, not a data gap. They
    # still took part in ranking (via _run_monte_carlo/_rank_and_flag), but
    # can't be scored against an unknown truth.
    preds = preds.dropna(subset=[TARGET_POS])
    if preds.empty:
        return {"n_races": 0, "n_rows": 0}
    out = {"n_races": int(preds[GROUP_COLS].drop_duplicates().shape[0]), "n_rows": int(len(preds))}

    point_metrics = {}
    for method in ("model", "baseline_rolling5", "baseline_last_year"):
        point_metrics[method] = _race_metrics(preds, f"{method}_pos")
    out["point_metrics"] = point_metrics

    bp = baseline_rolling_probs(preds["driver_rolling_quali_position_5"], rolling_prob_table)
    actual_pole = (preds[TARGET_POS] == 1).astype(int)
    actual_top3 = (preds[TARGET_POS] <= 3).astype(int)
    actual_q3 = (preds[TARGET_POS] <= 10).astype(int)
    out["prob_metrics"] = {
        "model": {
            "pole": _prob_metrics(actual_pole, preds["p_pole"]),
            "top3": _prob_metrics(actual_top3, preds["p_top3"]),
            "q3": _prob_metrics(actual_q3, preds["p_q3"]),
        },
        "baseline_rolling_prob": {
            "pole": _prob_metrics(actual_pole, bp["p_pole"]),
            "top3": _prob_metrics(actual_top3, bp["p_top3"]),
            "q3": _prob_metrics(actual_q3, bp["p_q3"]),
        },
    }
    return out


# --------------------------------------------------------------------------
# Walk-forward evaluation
# --------------------------------------------------------------------------

def _walk_forward_raw(df: pd.DataFrame, min_train_races: int, params: dict | None):
    df = add_quali_target(df).reset_index(drop=True)
    params = params or {}
    rows = []
    for train_idx, test_idx in time_based_splits(df, min_train_races):
        train, test = df.loc[train_idx], df.loc[test_idx]
        train_rows = train.dropna(subset=[TARGET_DELTA])
        if len(train_rows) < 10:
            continue
        pipe = make_pipeline(**params).fit(_prep(train_rows), train_rows[TARGET_DELTA])

        out = test[GROUP_COLS + ["driver", TARGET_POS, "driver_rolling_quali_position_5",
                                 "driver_circuit_last_quali", BASELINE_DELTA_COL]].copy()
        out["delta_pred"] = pipe.predict(_prep(test))
        out["baseline"] = test[BASELINE_DELTA_COL]
        out["actual_delta"] = test[TARGET_POS] - test[BASELINE_DELTA_COL]
        out["baseline_rolling5_pos_raw"] = test["driver_rolling_quali_position_5"]
        out["baseline_last_year_pos_raw"] = test["driver_circuit_last_quali"]
        rows.append(out)
    if not rows:
        raise ValueError(f"Not enough races for a single fold (need > {min_train_races})")
    return pd.concat(rows, ignore_index=True)


def _run_monte_carlo(preds: pd.DataFrame, residual_std: float, n_sims: int, seed: int) -> pd.DataFrame:
    preds = preds.copy()
    sim_cols = {"p_pole": [], "p_top3": [], "p_q3": [], "expected_position": []}
    for _, race in preds.groupby(GROUP_COLS, sort=False):
        # A driver with no rolling baseline yet (NaN) can't be meaningfully
        # simulated -- fall back to the field's own worst known baseline so
        # they still get ranked (toward the back), rather than dropping them.
        baseline = race["baseline"].fillna(race["baseline"].max() if race["baseline"].notna().any()
                                          else len(race))
        sim = simulate_quali_positions(baseline, race["delta_pred"], residual_std, n_sims=n_sims, seed=seed)
        for k in sim_cols:
            sim_cols[k].append(pd.Series(sim[k], index=race.index))
    for k, parts in sim_cols.items():
        preds[k] = pd.concat(parts).sort_index()
    return preds


def _rank_and_flag(preds: pd.DataFrame) -> pd.DataFrame:
    preds = rank_within_race(preds, "expected_position", "model_pos")
    preds = rank_within_race(preds, "baseline_rolling5_pos_raw", "baseline_rolling5_pos")
    preds = rank_within_race(preds, "baseline_last_year_pos_raw", "baseline_last_year_pos")
    return preds


def evaluate(df: pd.DataFrame, min_train_races: int = 15, params: dict | None = None,
            n_sims: int = 10_000, seed: int = 42) -> tuple[pd.DataFrame, dict]:
    """Quick, self-contained walk-forward evaluation with a single global
    residual_std (no dev-only-fit calibration layering -- see
    race_predictor.py for that pattern if this model needs it later). Used
    by select_hyperparams() and tests."""
    preds = _walk_forward_raw(df, min_train_races, params)
    residual_std = float(preds["actual_delta"].std())
    if not np.isfinite(residual_std) or residual_std <= 0:
        residual_std = float(preds["actual_delta"].abs().mean()) or 1.0
    preds = _run_monte_carlo(preds, residual_std, n_sims, seed)
    preds = _rank_and_flag(preds)
    return preds, {"residual_std": round(residual_std, 4)}


def run_full_evaluation(df: pd.DataFrame, params: dict | None = None, min_train_races: int = 15,
                        n_sims: int = 10_000, seed: int = 42) -> dict:
    dev_df, _ = dev_holdout_split(df)
    rolling_prob_table = empirical_rolling_probs(dev_df)
    preds, meta = evaluate(df, min_train_races, params, n_sims=n_sims, seed=seed)

    dev_preds = preds[preds["season"] <= DEV_SEASON_MAX]
    holdout_preds = preds[preds["season"] > DEV_SEASON_MAX]

    report = {
        "n_races": int(preds[GROUP_COLS].drop_duplicates().shape[0]),
        "n_rows": int(len(preds)),
        "residual_std": meta["residual_std"],
        "dev": summarise(dev_preds, rolling_prob_table),
        "holdout": summarise(holdout_preds, rolling_prob_table),
        "by_season": {str(season): summarise(g, rolling_prob_table) for season, g in preds.groupby("season")},
    }
    if not holdout_preds.empty:
        report["holdout"]["bootstrap_ci"] = bootstrap_diff_ci(holdout_preds, rolling_prob_table)
    return report


# --------------------------------------------------------------------------
# Bootstrap CIs for model-minus-baseline differences (resample whole races).
# --------------------------------------------------------------------------

def bootstrap_diff_ci(preds: pd.DataFrame, rolling_prob_table: pd.DataFrame, n_boot: int = 2000,
                      ci: float = 0.95, seed: int = 42) -> dict:
    preds = preds.dropna(subset=[TARGET_POS])  # see summarise()'s note on the handful of no-time rows
    bp = baseline_rolling_probs(preds["driver_rolling_quali_position_5"], rolling_prob_table)
    races = []
    for _, race in preds.groupby(GROUP_COLS, sort=False):
        base_p = bp.loc[race.index, "p_pole"]
        races.append({
            "model_mae": mean_absolute_error(race[TARGET_POS], race["model_pos"]),
            "rolling5_mae": mean_absolute_error(race[TARGET_POS], race["baseline_rolling5_pos"]),
            "actual_pole": (race[TARGET_POS] == 1).to_numpy(dtype=float),
            "model_p_pole": race["p_pole"].to_numpy(),
            "base_p_pole": base_p.to_numpy(),
        })
    n = len(races)
    rng = np.random.default_rng(seed)
    mae_diff = np.empty(n_boot)
    pole_brier_diff = np.empty(n_boot)
    for b in range(n_boot):
        sample = [races[i] for i in rng.integers(0, n, size=n)]
        mae_diff[b] = (np.mean([r["model_mae"] for r in sample])
                      - np.mean([r["rolling5_mae"] for r in sample]))
        actual = np.concatenate([r["actual_pole"] for r in sample])
        p = np.clip(np.concatenate([r["model_p_pole"] for r in sample]), 1e-6, 1 - 1e-6)
        base_p = np.clip(np.concatenate([r["base_p_pole"] for r in sample]), 1e-6, 1 - 1e-6)
        pole_brier_diff[b] = brier_score_loss(actual, p) - brier_score_loss(actual, base_p)
    alpha = (1 - ci) / 2

    def _summarise(arr):
        return {"mean_diff": round(float(arr.mean()), 4),
               "ci_low": round(float(np.percentile(arr, 100 * alpha)), 4),
               "ci_high": round(float(np.percentile(arr, 100 * (1 - alpha))), 4)}
    return {"n_boot": n_boot, "ci": ci,
           "position_mae_diff_vs_rolling5": _summarise(mae_diff),
           "pole_brier_diff_vs_rolling_prob": _summarise(pole_brier_diff)}


# --------------------------------------------------------------------------
# Hyperparameter selection -- dev seasons (<=2024) only, same structural
# guard as race_predictor.py's select_hyperparams().
# --------------------------------------------------------------------------

def select_hyperparams(dev_df: pd.DataFrame, grid: list[dict] | None = None,
                       min_train_races: int = 10, n_sims: int = 200) -> dict:
    if (dev_df["season"] > DEV_SEASON_MAX).any():
        raise ValueError(f"select_hyperparams must only see dev seasons (<= {DEV_SEASON_MAX}) -- "
                         "pass dev_holdout_split(df)[0], not the full dataset")
    best_cfg, best_score = None, np.inf
    for cfg in (grid or HP_GRID):
        preds, _ = evaluate(dev_df, min_train_races=min_train_races, params=cfg, n_sims=n_sims)
        score = mean_absolute_error(preds["actual_delta"].dropna(),
                                    preds.loc[preds["actual_delta"].notna(), "delta_pred"])
        if score < best_score:
            best_score, best_cfg = score, cfg
    return best_cfg


# --------------------------------------------------------------------------
# Final fit + single-session prediction + freezing.
# --------------------------------------------------------------------------

def fit_final(df: pd.DataFrame, params: dict | None = None) -> tuple[Pipeline, dict]:
    df = add_quali_target(df)
    train_rows = df.dropna(subset=[TARGET_DELTA])
    pipe = make_pipeline(**(params or {})).fit(_prep(train_rows), train_rows[TARGET_DELTA])
    resid = train_rows[TARGET_DELTA] - pipe.predict(_prep(train_rows))
    meta = {"residual_std": float(resid.std()) or 1.0}
    return pipe, meta


def train(df: pd.DataFrame, params: dict | None = None) -> dict:
    params = params or DEFAULT_PARAMS
    pipe, meta = fit_final(df, params)
    report = run_full_evaluation(df, params)
    meta["residual_std"] = report["residual_std"]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipe": pipe, "meta": meta}, MODEL_BUNDLE_PATH)
    METRICS_PATH.write_text(json.dumps(report, indent=2))
    return report


def load_models():
    if not MODEL_BUNDLE_PATH.exists():
        return None, None
    bundle = joblib.load(MODEL_BUNDLE_PATH)
    return bundle["pipe"], bundle["meta"]


def ensure_trained(dataset_path) -> tuple[Pipeline, dict]:
    if FROZEN_BUNDLE_PATH.exists():
        return load_frozen_model()
    pipe, meta = load_models()
    if pipe is not None:
        return pipe, meta
    df = pd.read_parquet(dataset_path)
    pipe, meta = fit_final(df)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipe": pipe, "meta": meta}, MODEL_BUNDLE_PATH)
    return pipe, meta


def _dataset_sha256(dataset_path) -> str:
    return hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest()


def freeze_model(df: pd.DataFrame, dataset_path, version: str, notes: str = "",
                 params: dict | None = None, frozen_at: str | None = None) -> dict:
    params = params or DEFAULT_PARAMS
    pipe, meta = fit_final(df, params)
    frozen_at = frozen_at or date.today().isoformat()

    FROZEN_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipe": pipe, "meta": meta}, FROZEN_BUNDLE_PATH)

    spec = {"version": version, "frozen_at": frozen_at, "dataset_sha256": _dataset_sha256(dataset_path),
           "n_rows": int(len(df)), "params": params, "residual_std": meta["residual_std"], "notes": notes}
    FROZEN_SPEC_PATH.write_text(json.dumps(spec, indent=2))
    return spec


def load_frozen_model() -> tuple[Pipeline, dict]:
    if not FROZEN_BUNDLE_PATH.exists():
        raise FileNotFoundError(f"No frozen quali model at {FROZEN_BUNDLE_PATH} -- run "
                               "`python -m scripts.freeze_quali_predictor` first.")
    bundle = joblib.load(FROZEN_BUNDLE_PATH)
    return bundle["pipe"], bundle["meta"]


def frozen_model_spec() -> dict | None:
    if not FROZEN_SPEC_PATH.exists():
        return None
    return json.loads(FROZEN_SPEC_PATH.read_text())


def predict_quali(pipe: Pipeline, race_features: pd.DataFrame, meta: dict,
                  n_sims: int = 10_000, seed: int | None = None) -> pd.DataFrame:
    """race_features: one row per driver for an upcoming qualifying session
    (same columns as the training features, plus "driver" and
    driver_rolling_quali_position_3 for the delta baseline). Monte Carlo
    simulates the session and returns predicted_quali_position (unique
    1..N), pole_probability, top3_probability and q3_probability, sorted by
    predicted_quali_position."""
    out = race_features.copy()
    out["delta_pred"] = pipe.predict(_prep(out))
    baseline = out[BASELINE_DELTA_COL]
    if baseline.isna().any():
        fallback = baseline.max() if baseline.notna().any() else float(len(out))
        baseline = baseline.fillna(fallback)
    sim = simulate_quali_positions(baseline, out["delta_pred"], meta["residual_std"], n_sims=n_sims, seed=seed)
    out["pole_probability"] = sim["p_pole"]
    out["top3_probability"] = sim["p_top3"]
    out["q3_probability"] = sim["p_q3"]
    out["expected_quali_position"] = sim["expected_position"]
    out = rank_within_race(out, "expected_quali_position", "predicted_quali_position", group_cols=None)
    return out.sort_values("predicted_quali_position").reset_index(drop=True)
