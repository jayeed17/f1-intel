"""Offline tests for app.predictions_log -- shared, FastF1-free helpers for
predictions/*.csv, used by both the FastF1-touching scripts/predict_*.py
and the pure app/models/*_predictor.py modules. No network, no FastF1."""
import pandas as pd

import app.predictions_log as log_mod
from app.predictions_log import (log_missed_prediction, missed_predictions,
                                 missed_records_since, prediction_already_logged)


def test_prediction_already_logged_false_when_file_missing(tmp_path):
    assert prediction_already_logged(tmp_path / "nope.csv", "Some Grand Prix") is False


def test_prediction_already_logged_true_once_a_row_exists(tmp_path):
    path = tmp_path / "2026.csv"
    pd.DataFrame({"gp": ["Azerbaijan Grand Prix"], "driver": ["VER"]}).to_csv(path, index=False)
    assert prediction_already_logged(path, "Azerbaijan Grand Prix") is True
    assert prediction_already_logged(path, "Monaco Grand Prix") is False


def test_prediction_already_logged_false_on_empty_file(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("")
    assert prediction_already_logged(path, "Monaco Grand Prix") is False


def test_log_missed_prediction_writes_and_dedupes(monkeypatch, tmp_path):
    missed_path = tmp_path / "missed.csv"
    monkeypatch.setattr(log_mod, "MISSED_PREDICTIONS_PATH", missed_path)

    log_missed_prediction(2026, "United States Grand Prix", "race_v1", "post_quali",
                          "pipeline failure", noted_at="2026-10-26T00:00:00+00:00")
    df = pd.read_csv(missed_path)
    assert len(df) == 1
    assert df.iloc[0]["gp"] == "United States Grand Prix"

    # Same (year, gp, model, stage) again -- must not duplicate.
    log_missed_prediction(2026, "United States Grand Prix", "race_v1", "post_quali",
                          "pipeline failure (again)", noted_at="2026-10-27T00:00:00+00:00")
    df = pd.read_csv(missed_path)
    assert len(df) == 1

    # A different stage for the same race IS a distinct entry.
    log_missed_prediction(2026, "United States Grand Prix", "quali", "post_practice",
                          "pipeline failure", noted_at="2026-10-26T00:00:00+00:00")
    df = pd.read_csv(missed_path)
    assert len(df) == 2


def test_missed_predictions_filters_by_model_and_stage(monkeypatch, tmp_path):
    missed_path = tmp_path / "missed.csv"
    monkeypatch.setattr(log_mod, "MISSED_PREDICTIONS_PATH", missed_path)

    log_missed_prediction(2026, "Bahrain Grand Prix", "race_v1", "post_quali", "x",
                          noted_at="2026-10-03T00:00:00+00:00")
    log_missed_prediction(2026, "Bahrain Grand Prix", "race_v2", "post_quali", "x",
                          noted_at="2026-10-03T00:00:00+00:00")
    log_missed_prediction(2026, "Bahrain Grand Prix", "quali", "post_practice", "x",
                          noted_at="2026-10-03T00:00:00+00:00")

    assert len(missed_predictions(model="race_v1")) == 1
    assert len(missed_predictions(stage="post_quali")) == 2
    assert len(missed_predictions(model="quali", stage="post_practice")) == 1
    assert len(missed_predictions()) == 3


def test_missed_predictions_empty_when_no_file(monkeypatch, tmp_path):
    monkeypatch.setattr(log_mod, "MISSED_PREDICTIONS_PATH", tmp_path / "nope.csv")
    df = missed_predictions()
    assert df.empty
    assert list(df.columns) == ["year", "gp", "model", "stage", "reason", "noted_at"]


def test_missed_records_since_respects_cutoff(monkeypatch, tmp_path):
    missed_path = tmp_path / "missed.csv"
    monkeypatch.setattr(log_mod, "MISSED_PREDICTIONS_PATH", missed_path)

    # Before the model existed -- must never surface in a live track record.
    log_missed_prediction(2022, "Pre-freeze Grand Prix", "race_v1", "post_quali", "x",
                          noted_at="2022-01-01T00:00:00+00:00")
    # After the freeze -- the real case (2026 US GP).
    log_missed_prediction(2026, "United States Grand Prix", "race_v1", "post_quali",
                          "quali ended and the race started before a prediction was ever "
                          "logged (pipeline failure) -- not backfilled",
                          noted_at="2026-10-26T00:00:00+00:00")

    records = missed_records_since("race_v1", "post_quali", "2026-09-28")
    assert len(records) == 1
    assert records[0]["gp"] == "United States Grand Prix"
    assert "pipeline failure" in records[0]["reason"]


def test_missed_records_since_empty_when_nothing_matches(monkeypatch, tmp_path):
    monkeypatch.setattr(log_mod, "MISSED_PREDICTIONS_PATH", tmp_path / "nope.csv")
    assert missed_records_since("race_v1", "post_quali", "2026-09-28") == []
