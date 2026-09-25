"""ML model: predict fuel-corrected lap-time loss vs driver median from tyre/track state."""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from app.config import MODEL_DIR
from app.models.degradation import fuel_correct

NUM = ["TyreLife", "LapNumber", "Stint", "TrackTemp", "AirTemp"]
CAT = ["Compound", "Circuit"]
TARGET = "Rel"
MODEL_PATH = MODEL_DIR / "tyre_model.joblib"
METRICS_PATH = MODEL_DIR / "tyre_metrics.json"


def build_features(clean: pd.DataFrame, circuit: str) -> pd.DataFrame:
    """clean = data.clean_laps(session, with_weather=True)."""
    df = fuel_correct(clean)
    df = df[df["Compound"].isin(["SOFT", "MEDIUM", "HARD"])].copy()
    df["Rel"] = df["CorrS"] - df.groupby("Driver")["CorrS"].transform("median")
    df["Circuit"] = circuit
    for c in NUM:
        if c not in df:
            df[c] = np.nan
    return df[NUM + CAT + [TARGET, "Driver", "Team"]].reset_index(drop=True)


def _prep(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for c in NUM:
        if c not in X:
            X[c] = np.nan
        X[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
    for c in CAT:
        if c not in X:
            X[c] = "unknown"
        X[c] = X[c].fillna("unknown").astype(str)
    return X[NUM + CAT]


def make_pipeline() -> Pipeline:
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
        ("num", "passthrough", NUM),
    ])
    reg = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                        l2_regularization=1.0, random_state=42)
    return Pipeline([("pre", pre), ("reg", reg)])


def train(df: pd.DataFrame) -> dict:
    df = df.dropna(subset=[TARGET, "TyreLife", "Compound"])
    X, y, groups = _prep(df), df[TARGET], df["Circuit"]
    metrics = {"rows": int(len(df)), "circuits": int(groups.nunique())}

    if groups.nunique() >= 2:  # leave-circuits-out CV: does it generalise to unseen tracks?
        maes, base = [], []
        for tr, te in GroupKFold(n_splits=min(5, groups.nunique())).split(X, y, groups):
            p = make_pipeline().fit(X.iloc[tr], y.iloc[tr])
            maes.append(mean_absolute_error(y.iloc[te], p.predict(X.iloc[te])))
            base.append(mean_absolute_error(y.iloc[te], np.zeros(len(te))))
        metrics.update(cv_mae_s=round(float(np.mean(maes)), 4),
                       baseline_mae_s=round(float(np.mean(base)), 4))

    pipe = make_pipeline().fit(X, y)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    return metrics


def load_model():
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


def predict(pipe, rows: list[dict]) -> list[float]:
    return [round(float(v), 4) for v in pipe.predict(_prep(pd.DataFrame(rows)))]
