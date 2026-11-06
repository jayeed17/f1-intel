"""Shared, FastF1-free helpers for predictions/*.csv.

Used by both the FastF1-touching layer (app/data.py, scripts/predict_*.py)
and the pure app/models/*_predictor.py modules, which must stay FastF1-free
(see CLAUDE.md) and therefore can't import app.data directly.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.config import PREDICTIONS_DIR

MISSED_PREDICTIONS_PATH = PREDICTIONS_DIR / "missed.csv"
_MISSED_COLUMNS = ["year", "gp", "model", "stage", "reason", "noted_at"]


def prediction_already_logged(path: Path, gp: str) -> bool:
    """True if `path` (a predictions/*.csv log) already has a row for this
    gp -- callers must never log the same race's prediction twice."""
    if not path.exists():
        return False
    try:
        df = pd.read_csv(path, usecols=["gp"])
    except (ValueError, pd.errors.EmptyDataError):
        return False
    return bool((df["gp"] == gp).any())


def _missed_path(predictions_dir: Path | None) -> Path:
    """missed.csv under `predictions_dir` if given, else the real
    predictions/ dir. The optional override exists so callers that already
    hold their own (possibly monkeypatched-in-tests) PREDICTIONS_DIR
    reference -- e.g. each app/models/*_predictor.py's compute_live_track_
    record() -- can point this module at the same directory a test
    redirected them to, instead of always hitting the real repo path."""
    return (Path(predictions_dir) / "missed.csv") if predictions_dir is not None else MISSED_PREDICTIONS_PATH


def log_missed_prediction(year: int, gp: str, model: str, stage: str, reason: str,
                          noted_at: str | None = None, predictions_dir: Path | None = None) -> None:
    """Record that `model`'s `stage` prediction for (year, gp) was never
    logged -- a pipeline failure, not a bye week. Read by each model's
    compute_live_track_record() so the live track record is honest about
    the gap instead of silently omitting the race or backfilling a
    prediction after the fact. Deduped by (year, gp, model, stage)."""
    path = _missed_path(predictions_dir)
    noted_at = noted_at or pd.Timestamp.now(tz="UTC").isoformat()
    row = pd.DataFrame([{"year": year, "gp": gp, "model": model, "stage": stage,
                         "reason": reason, "noted_at": noted_at}])
    if path.exists():
        existing = pd.read_csv(path)
        dup = ((existing["year"] == year) & (existing["gp"] == gp)
              & (existing["model"] == model) & (existing["stage"] == stage))
        if dup.any():
            return
        row = pd.concat([existing, row], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    row.to_csv(path, index=False)


def missed_predictions(model: str | None = None, stage: str | None = None,
                       predictions_dir: Path | None = None) -> pd.DataFrame:
    """Every logged pipeline-failure gap, optionally filtered by model/stage."""
    path = _missed_path(predictions_dir)
    if not path.exists():
        return pd.DataFrame(columns=_MISSED_COLUMNS)
    df = pd.read_csv(path)
    if model is not None:
        df = df[df["model"] == model]
    if stage is not None:
        df = df[df["stage"] == stage]
    return df


def missed_records_since(model: str, stage: str, cutoff, predictions_dir: Path | None = None) -> list[dict]:
    """missed_predictions() filtered to `model`/`stage` and to noted_at >=
    cutoff, as a list of {gp, reason, noted_at} dicts -- what each model's
    compute_live_track_record() mixes into its output."""
    df = missed_predictions(model=model, stage=stage, predictions_dir=predictions_dir)
    if df.empty:
        return []
    noted = pd.to_datetime(df["noted_at"], utc=True)
    cutoff_ts = pd.Timestamp(cutoff)
    cutoff_ts = cutoff_ts.tz_localize("UTC") if cutoff_ts.tzinfo is None else cutoff_ts.tz_convert("UTC")
    df = df[noted >= cutoff_ts]
    return df[["gp", "reason", "noted_at"]].to_dict("records")
