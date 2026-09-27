"""Race outcome predictor: finish position + points probability from
pre-race features only (grid, qualifying, rolling form, circuit type).

Trained on data/model/race_dataset.parquet (see scripts/build_race_dataset.py).
Pure functions -- everything here takes/returns plain DataFrames, no FastF1.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from app.config import MODEL_DIR

NUM = ["grid", "quali_position", "quali_gap_to_pole_s", "teammate_quali_gap_s",
      "driver_rolling_avg_finish_3", "driver_rolling_avg_finish_5",
      "team_rolling_avg_finish_3", "team_rolling_pace_gap_3", "driver_dnf_rate_10"]
BOOL = ["grid_pit_lane", "reg_change_flag"]
CAT = ["circuit_type"]
FEATURES = NUM + BOOL + CAT
TARGET_POS = "target_finish_pos"
TARGET_PTS = "target_points_top10"
GROUP_COLS = ["season", "round"]

MODEL_POS_PATH = MODEL_DIR / "race_predictor_position.joblib"
MODEL_PTS_PATH = MODEL_DIR / "race_predictor_points.joblib"
METRICS_PATH = MODEL_DIR / "race_predictor_metrics.json"


def _prep(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for c in NUM:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype(float) if c in X else np.nan
    for c in BOOL:
        X[c] = X[c].fillna(False).astype(float) if c in X else 0.0
    for c in CAT:
        X[c] = X[c].fillna("unknown").astype(str) if c in X else "unknown"
    return X[FEATURES]


def make_position_pipeline() -> Pipeline:
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
        ("num", "passthrough", NUM + BOOL),
    ])
    reg = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                        l2_regularization=1.0, random_state=42)
    return Pipeline([("pre", pre), ("reg", reg)])


def make_points_pipeline() -> Pipeline:
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
        ("num", "passthrough", NUM + BOOL),
    ])
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                         l2_regularization=1.0, random_state=42)
    return Pipeline([("pre", pre), ("clf", clf)])


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


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def _race_metrics(preds: pd.DataFrame, pos_col: str) -> dict:
    """Per-race-aware metrics for one prediction column (model or baseline).
    top3_hit_rate: fraction of races where the predicted winner (rank 1)
    actually finished in the real top 3. winner_accuracy: fraction of races
    where the predicted winner was the actual winner.
    """
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


def summarise(preds: pd.DataFrame) -> dict:
    """preds must have: season, round, driver, target_finish_pos,
    target_points_top10, and one *_pos column + one *_points column per
    method in {"model", "baseline_grid", "baseline_quali"}."""
    out = {"n_races": int(preds[GROUP_COLS].drop_duplicates().shape[0]), "n_rows": int(len(preds))}
    for method in ("model", "baseline_grid", "baseline_quali"):
        m = _race_metrics(preds, f"{method}_pos")
        m["points_f1"] = _points_f1(preds[TARGET_PTS], preds[f"{method}_points"])
        out[method] = m
        for season, g in preds.groupby("season"):
            sm = _race_metrics(g, f"{method}_pos")
            sm["points_f1"] = _points_f1(g[TARGET_PTS], g[f"{method}_points"])
            out.setdefault("by_season", {}).setdefault(str(season), {})[method] = sm
    return out


# --------------------------------------------------------------------------
# Walk-forward evaluation
# --------------------------------------------------------------------------

def evaluate(df: pd.DataFrame, min_train_races: int = 20) -> tuple[pd.DataFrame, dict]:
    """Runs the full expanding-window walk-forward evaluation and returns
    (per-row predictions, summary metrics dict) -- never trains on a race
    that hasn't happened yet relative to the race being predicted.
    """
    df = df.reset_index(drop=True)
    rows = []
    for train_idx, test_idx in time_based_splits(df, min_train_races):
        train, test = df.loc[train_idx], df.loc[test_idx]

        train_pos = train.dropna(subset=[TARGET_POS])
        train_pts = train.dropna(subset=[TARGET_PTS])
        if len(train_pos) < 10 or train_pts[TARGET_PTS].nunique() < 2:
            continue

        pos_pipe = make_position_pipeline().fit(_prep(train_pos), train_pos[TARGET_POS])
        pts_pipe = make_points_pipeline().fit(_prep(train_pts), train_pts[TARGET_PTS])

        out = test[GROUP_COLS + ["driver", "grid", "quali_position", TARGET_POS, TARGET_PTS]].copy()
        out["model_pos_raw"] = pos_pipe.predict(_prep(test))
        out["model_points"] = pts_pipe.predict(_prep(test))
        out["baseline_grid_pos_raw"] = test["grid"]
        out["baseline_quali_pos_raw"] = test["quali_position"]
        rows.append(out)

    if not rows:
        raise ValueError(f"Not enough races for a single fold (need > {min_train_races})")
    preds = pd.concat(rows, ignore_index=True)

    preds = rank_within_race(preds, "model_pos_raw", "model_pos")
    preds = rank_within_race(preds, "baseline_grid_pos_raw", "baseline_grid_pos")
    preds = rank_within_race(preds, "baseline_quali_pos_raw", "baseline_quali_pos")
    preds["baseline_grid_points"] = preds["baseline_grid_pos"] <= 10
    preds["baseline_quali_points"] = preds["baseline_quali_pos"] <= 10

    return preds, summarise(preds)


# --------------------------------------------------------------------------
# Final fit (all data) + single-race prediction, for the API/dashboard.
# --------------------------------------------------------------------------

def train(df: pd.DataFrame) -> dict:
    train_pos = df.dropna(subset=[TARGET_POS])
    train_pts = df.dropna(subset=[TARGET_PTS])
    pos_pipe = make_position_pipeline().fit(_prep(train_pos), train_pos[TARGET_POS])
    pts_pipe = make_points_pipeline().fit(_prep(train_pts), train_pts[TARGET_PTS])

    _, metrics = evaluate(df)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pos_pipe, MODEL_POS_PATH)
    joblib.dump(pts_pipe, MODEL_PTS_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    return metrics


def load_models():
    if not (MODEL_POS_PATH.exists() and MODEL_PTS_PATH.exists()):
        return None, None
    return joblib.load(MODEL_POS_PATH), joblib.load(MODEL_PTS_PATH)


def predict_race(pos_pipe, pts_pipe, race_features: pd.DataFrame) -> pd.DataFrame:
    """race_features: one row per driver for a single upcoming race (same
    columns as the training features, plus "driver" and "grid" for display).
    Returns race_features with predicted_position (unique 1..N) and
    points_probability added, sorted by predicted_position.
    """
    out = race_features.copy()
    out["predicted_finish_raw"] = pos_pipe.predict(_prep(out))
    out["points_probability"] = pts_pipe.predict_proba(_prep(out))[:, 1]
    out = rank_within_race(out, "predicted_finish_raw", "predicted_position", group_cols=None)
    return out.sort_values("predicted_position").reset_index(drop=True)
