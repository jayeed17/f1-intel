"""Offline tests for app.data.predictions_race_options() -- the dashboard's
2026 Predictions dropdown -- and is_event_cancelled(), its OpenF1
cross-check. No network: fastf1.get_event_schedule and openf1_meetings
are monkeypatched with synthetic data.

Regression context: the dropdown used to add only the single
chronologically-next unbuilt round, so every round past that one (e.g.
most of the back half of a season) was silently missing. predictions_
race_options() must return every remaining round on the calendar."""
import pandas as pd

import app.data as data_mod
from app.data import is_event_cancelled, predictions_race_options


def _fake_schedule(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_predictions_race_options_includes_every_non_cancelled_round(monkeypatch):
    monkeypatch.setattr(data_mod, "prebuilt_races", lambda: [
        {"year": 2026, "round": 1, "name": "Australian Grand Prix"},
        {"year": 2026, "round": 2, "name": "Chinese Grand Prix"},
    ])
    monkeypatch.setattr(data_mod.fastf1, "get_event_schedule", lambda year, include_testing: _fake_schedule([
        {"RoundNumber": 1, "EventName": "Australian Grand Prix", "EventDate": pd.Timestamp("2026-03-08")},
        {"RoundNumber": 2, "EventName": "Chinese Grand Prix", "EventDate": pd.Timestamp("2026-03-15")},
        {"RoundNumber": 3, "EventName": "Bahrain Grand Prix", "EventDate": pd.Timestamp("2026-10-04")},
        {"RoundNumber": 4, "EventName": "Saudi Arabian Grand Prix", "EventDate": pd.Timestamp("2026-04-19")},
        {"RoundNumber": 5, "EventName": "Miami Grand Prix", "EventDate": pd.Timestamp("2026-05-03")},
    ]))
    monkeypatch.setattr(data_mod, "openf1_meetings", lambda year: [
        {"meeting_name": "Bahrain Grand Prix", "date_start": "2026-04-10T00:00:00+00:00", "is_cancelled": True},
        {"meeting_name": "Bahrain Grand Prix", "date_start": "2026-10-02T00:00:00+00:00", "is_cancelled": False},
        {"meeting_name": "Saudi Arabian Grand Prix", "date_start": "2026-04-17T00:00:00+00:00", "is_cancelled": True},
    ])

    races, excluded = predictions_race_options(2026)
    names = [r["name"] for r in races]
    # Every non-cancelled round on the schedule shows up -- built rounds
    # AND every remaining upcoming one, not just the next one.
    assert names == ["Australian Grand Prix", "Chinese Grand Prix", "Bahrain Grand Prix", "Miami Grand Prix"]
    assert excluded == ["Saudi Arabian Grand Prix"]


def test_rescheduled_event_is_not_excluded_as_cancelled(monkeypatch):
    """The cancelled-then-rescheduled case (2026 Bahrain GP, April ->
    October) must be matched by nearest date, not flagged cancelled
    forever just because an earlier same-named meeting was cancelled."""
    meetings = [
        {"meeting_name": "Bahrain Grand Prix", "date_start": "2026-04-10T00:00:00+00:00", "is_cancelled": True},
        {"meeting_name": "Bahrain Grand Prix", "date_start": "2026-10-02T00:00:00+00:00", "is_cancelled": False},
    ]
    assert is_event_cancelled(meetings, "Bahrain Grand Prix", pd.Timestamp("2026-10-04")) is False
    assert is_event_cancelled(meetings, "Bahrain Grand Prix", pd.Timestamp("2026-04-11")) is True


def test_cancelled_with_no_reschedule_is_excluded():
    meetings = [
        {"meeting_name": "Saudi Arabian Grand Prix", "date_start": "2026-04-17T00:00:00+00:00", "is_cancelled": True},
    ]
    assert is_event_cancelled(meetings, "Saudi Arabian Grand Prix", pd.Timestamp("2026-04-19")) is True


def test_unknown_event_or_failed_lookup_defaults_to_not_cancelled():
    # No OpenF1 data for this name at all -- can't tell, don't block it.
    assert is_event_cancelled([{"meeting_name": "Other GP", "date_start": "2026-01-01T00:00:00+00:00",
                               "is_cancelled": True}], "Fictional Grand Prix", pd.Timestamp("2026-05-01")) is False
    # OpenF1 unreachable (empty list) -- can't check, don't block anything.
    assert is_event_cancelled([], "Bahrain Grand Prix", pd.Timestamp("2026-10-04")) is False


def test_schedule_fetch_failure_degrades_to_built_rounds_only(monkeypatch):
    monkeypatch.setattr(data_mod, "prebuilt_races", lambda: [
        {"year": 2026, "round": 1, "name": "Australian Grand Prix"},
    ])

    def _boom(*a, **k):
        raise RuntimeError("no network")
    monkeypatch.setattr(data_mod.fastf1, "get_event_schedule", _boom)

    races, excluded = predictions_race_options(2026)
    assert [r["name"] for r in races] == ["Australian Grand Prix"]
    assert excluded == []
