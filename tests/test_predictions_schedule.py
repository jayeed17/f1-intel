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
from app.data import is_event_cancelled, predictions_race_options, race_has_started, session_readiness


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


def _cota_like_schedule_row() -> pd.Series:
    """The 2026 US GP's real session timing (Qualifying 21:00-22:00 UTC,
    Race 20:00-22:00 UTC the next day) -- two hours later than most
    European-weekend races, which is why the old fixed "0 20 * * 6" cron
    missed it."""
    return pd.Series({
        "EventName": "United States Grand Prix",
        "Session1": "Practice 1", "Session1DateUtc": pd.Timestamp("2026-10-23T17:30:00Z"),
        "Session2": "Practice 2", "Session2DateUtc": pd.Timestamp("2026-10-23T21:00:00Z"),
        "Session3": "Practice 3", "Session3DateUtc": pd.Timestamp("2026-10-24T17:30:00Z"),
        "Session4": "Qualifying", "Session4DateUtc": pd.Timestamp("2026-10-24T21:00:00Z"),
        "Session5": "Race", "Session5DateUtc": pd.Timestamp("2026-10-25T20:00:00Z"),
    })


def test_session_readiness_false_before_the_buffer_elapses(monkeypatch):
    """This is exactly the 2026 US GP scenario: Qualifying started at
    21:00 UTC, so a cron firing at 22:13 UTC (13 minutes after the
    schedule's listed end, and well under the 90-minute buffer) must not
    be considered ready yet."""
    monkeypatch.setattr(data_mod.fastf1, "get_event_schedule",
                        lambda year, include_testing: pd.DataFrame([_cota_like_schedule_row()]))
    now = pd.Timestamp("2026-10-24T22:13:00Z")  # 13 min after quali's listed end
    ready, reason = session_readiness(now, 2026, "United States Grand Prix", ("Qualifying",),
                                       pd.Timedelta(minutes=90))
    assert ready is False
    assert "hasn't ended" in reason or "started yet" in reason


def test_session_readiness_true_once_the_buffer_elapses(monkeypatch):
    monkeypatch.setattr(data_mod.fastf1, "get_event_schedule",
                        lambda year, include_testing: pd.DataFrame([_cota_like_schedule_row()]))
    # Qualifying: 21:00 start + 60min default duration + 90min buffer = ready at 23:30 UTC.
    not_yet = pd.Timestamp("2026-10-24T23:00:00Z")
    ready, _ = session_readiness(not_yet, 2026, "United States Grand Prix", ("Qualifying",),
                                 pd.Timedelta(minutes=90))
    assert ready is False

    ready_now = pd.Timestamp("2026-10-24T23:31:00Z")
    ready, _ = session_readiness(ready_now, 2026, "United States Grand Prix", ("Qualifying",),
                                 pd.Timedelta(minutes=90))
    assert ready is True


def test_race_has_started_true_only_once_race_session_begins(monkeypatch):
    monkeypatch.setattr(data_mod.fastf1, "get_event_schedule",
                        lambda year, include_testing: pd.DataFrame([_cota_like_schedule_row()]))
    before_race = pd.Timestamp("2026-10-25T19:00:00Z")
    after_race_start = pd.Timestamp("2026-10-25T20:30:00Z")
    assert race_has_started(before_race, 2026, "United States Grand Prix") is False
    assert race_has_started(after_race_start, 2026, "United States Grand Prix") is True


def test_race_has_started_fails_safe_when_schedule_unavailable(monkeypatch):
    """A transient schedule-fetch failure must never let a late prediction
    be mistaken for "the race hasn't started" -- fail toward True (started)
    so nothing ever gets backfilled after the fact."""
    def _boom(*a, **k):
        raise RuntimeError("no network")
    monkeypatch.setattr(data_mod.fastf1, "get_event_schedule", _boom)
    assert race_has_started(pd.Timestamp("2026-10-25T20:30:00Z"), 2026, "United States Grand Prix") is True
