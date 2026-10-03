"""Street vs permanent circuit comparison for the dashboard's "Street
circuits" view. Pure DataFrame-in/DataFrame-out -- no FastF1, testable
with synthetic frames. Circuit classification (STREET_CIRCUIT_CLASS) is
display-only, deliberately separate from CIRCUIT_TYPE (a trained model
feature) -- see app/config.py.

SC/VSC frequency and braking-zone aggregation take already-loaded,
per-race/per-lap summaries (built by the FastF1-touching app/data.py
loaders) rather than raw telemetry, keeping this module itself FastF1-free.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from app.config import STREET_CIRCUIT_CLASS

CLASSES = ("street", "hybrid_street", "permanent")


def classify_circuit(circuit_id: str | None) -> str:
    """"street" / "hybrid_street" / "permanent" -- anything not in
    STREET_CIRCUIT_CLASS (including None/unresolved) defaults to
    "permanent", the same fallback convention CIRCUIT_TYPE uses ("mixed")."""
    if circuit_id is None:
        return "permanent"
    return STREET_CIRCUIT_CLASS.get(circuit_id, "permanent")


def add_circuit_class(df: pd.DataFrame, circuit_col: str = "circuit_id") -> pd.DataFrame:
    df = df.copy()
    df["circuit_class"] = df[circuit_col].map(classify_circuit)
    return df


def street_vs_permanent_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """One row per circuit_class: n_races, n_rows, grid->finish Spearman
    (how predictable the finish is from the grid), avg positions gained
    (grid - finish; positive = gained), DNF rate, pole conversion rate
    (P(win | grid==1)), and podium_from_top3_grid_rate (share of podium
    finishers who started inside the top 3) as a second "how much does
    quali position matter" signal alongside the Spearman figure. A metric
    is None where there isn't enough data in that class to compute it
    (e.g. no pole-sitters recorded)."""
    df = df if "circuit_class" in df else add_circuit_class(df)
    rows = []
    for cls in CLASSES:
        g = df[df["circuit_class"] == cls]
        if g.empty:
            continue
        finishers = g.dropna(subset=["grid", "target_finish_pos"])
        rho = (spearmanr(finishers["grid"], finishers["target_finish_pos"]).correlation
              if len(finishers) >= 3 else np.nan)
        pole_rows = g[g["grid"] == 1]
        pole_conv = float((pole_rows["target_finish_pos"] == 1).mean()) if not pole_rows.empty else np.nan
        podium = finishers[finishers["target_finish_pos"] <= 3]
        podium_from_top3 = float((podium["grid"] <= 3).mean()) if not podium.empty else np.nan
        rows.append({
            "circuit_class": cls,
            "n_races": int(g[["season", "round"]].drop_duplicates().shape[0]),
            "n_rows": int(len(g)),
            "grid_finish_spearman": round(float(rho), 3) if pd.notna(rho) else None,
            "avg_positions_gained": (round(float((finishers["grid"] - finishers["target_finish_pos"]).mean()), 3)
                                     if not finishers.empty else None),
            "dnf_rate": round(float(g["dnf"].mean()), 3) if "dnf" in g and g["dnf"].notna().any() else None,
            "pole_conversion_rate": round(pole_conv, 3) if pd.notna(pole_conv) else None,
            "podium_from_top3_grid_rate": round(podium_from_top3, 3) if pd.notna(podium_from_top3) else None,
        })
    return pd.DataFrame(rows)


def circuit_history_table(df: pd.DataFrame, n_editions: int = 3,
                          classes: tuple[str, ...] = ("street", "hybrid_street")) -> pd.DataFrame:
    """Per circuit (restricted to `classes` -- street-like by default):
    over the last `n_editions` editions actually present in df, each
    one's winner, the pole-to-win rate, avg DNFs per race, and the
    driver/team with the most podiums there in that window."""
    df = df if "circuit_class" in df else add_circuit_class(df)
    df = df[df["circuit_class"].isin(classes)]
    rows = []
    for cid, g in df.groupby("circuit_id"):
        races = g[["season", "round"]].drop_duplicates().sort_values(["season", "round"])
        recent = races.tail(n_editions)
        recent_g = g.merge(recent, on=["season", "round"])

        winners, dnf_counts, pole_to_win, n = [], [], 0, 0
        podium_driver: dict[str, int] = {}
        podium_team: dict[str, int] = {}
        for (season, _round), race_g in recent_g.groupby(["season", "round"]):
            n += 1
            winner_row = race_g[race_g["target_finish_pos"] == 1]
            if not winner_row.empty:
                w = winner_row.iloc[0]
                winners.append(f"{w['driver']} ({int(season)})")
                if w["grid"] == 1:
                    pole_to_win += 1
            if "dnf" in race_g:
                dnf_counts.append(int(race_g["dnf"].sum()))
            for _, r in race_g[race_g["target_finish_pos"] <= 3].iterrows():
                podium_driver[r["driver"]] = podium_driver.get(r["driver"], 0) + 1
                podium_team[r["team_id"]] = podium_team.get(r["team_id"], 0) + 1

        rows.append({
            "circuit_id": cid,
            "circuit_class": classify_circuit(cid),
            "editions": n,
            "winners": ", ".join(winners),
            "pole_to_win_rate": round(pole_to_win / n, 3) if n else None,
            "avg_dnfs": round(float(np.mean(dnf_counts)), 2) if dnf_counts else None,
            "top_driver": max(podium_driver, key=podium_driver.get) if podium_driver else None,
            "top_team": max(podium_team, key=podium_team.get) if podium_team else None,
        })
    return pd.DataFrame(rows).sort_values("circuit_id").reset_index(drop=True)


def sc_vsc_frequency_by_class(race_flags: pd.DataFrame) -> pd.DataFrame:
    """race_flags: one row per race, columns circuit_id (or
    circuit_class) and had_sc_or_vsc (bool). One row per circuit_class:
    n_races and the share of those races with at least one SC/VSC
    period."""
    df = race_flags if "circuit_class" in race_flags else add_circuit_class(race_flags)
    rows = []
    for cls in CLASSES:
        g = df[df["circuit_class"] == cls]
        if g.empty:
            continue
        rows.append({"circuit_class": cls, "n_races": int(len(g)),
                     "sc_vsc_rate": round(float(g["had_sc_or_vsc"].mean()), 3)})
    return pd.DataFrame(rows)


def braking_summary_by_class(race_driver_zones: pd.DataFrame) -> pd.DataFrame:
    """race_driver_zones: one row per driver's fastest lap in a race,
    columns circuit_id (or circuit_class), n_zones, avg_decel_g. One row
    per circuit_class: avg braking zones per lap and avg deceleration
    across every sampled lap."""
    df = race_driver_zones if "circuit_class" in race_driver_zones else add_circuit_class(race_driver_zones)
    rows = []
    for cls in CLASSES:
        g = df[df["circuit_class"] == cls]
        if g.empty:
            continue
        rows.append({"circuit_class": cls, "n_laps": int(len(g)),
                     "avg_zones_per_lap": round(float(g["n_zones"].mean()), 2),
                     "avg_decel_g": round(float(g["avg_decel_g"].dropna().mean()), 3)
                     if g["avg_decel_g"].notna().any() else None})
    return pd.DataFrame(rows)


def model_accuracy_by_class(df: pd.DataFrame, min_races_for_ci: int = 8) -> dict:
    """Holdout (2025-2026) accuracy of the frozen v1 race predictor, split
    by circuit class, each with its own bootstrap CI vs the grid baseline
    -- reuses race_predictor.py's own holdout_predictions_with_circuit()/
    summarise()/bootstrap_diff_ci() on a per-class slice of its holdout
    predictions, so these numbers are directly comparable to the
    project-wide holdout report in README. A class with fewer than
    `min_races_for_ci` holdout races gets a `note` instead of a CI --
    said honestly rather than reported with false precision from a
    handful of races."""
    from app.models.race_predictor import bootstrap_diff_ci, holdout_predictions_with_circuit, summarise

    holdout_preds, grid_prob_table = holdout_predictions_with_circuit(df)
    holdout_preds = add_circuit_class(holdout_preds)

    out = {}
    for cls in CLASSES:
        g = holdout_preds[holdout_preds["circuit_class"] == cls]
        n_races = int(g[["season", "round"]].drop_duplicates().shape[0]) if not g.empty else 0
        if g.empty:
            out[cls] = {"n_races": 0, "note": "No holdout races in this class."}
            continue
        entry = {"n_races": n_races, **summarise(g, grid_prob_table)}
        if n_races >= min_races_for_ci:
            entry["bootstrap_ci"] = bootstrap_diff_ci(g, grid_prob_table)
        else:
            entry["note"] = f"Only {n_races} holdout races in this class -- too few for a reliable bootstrap CI."
        out[cls] = entry
    return out


def live_track_record_by_class(df: pd.DataFrame, frozen_at: str, predictions_dir,
                               min_races_for_ci: int = 5) -> dict:
    """Post-freeze live track record of the frozen v1 race predictor
    (predictions/{year}.csv rows logged at/after `frozen_at`), split by
    circuit class -- resolves each logged race's circuit via a best-
    effort event_name -> circuit_id lookup built from `df` (the live log
    only stores the event name string, not circuit_id, so a race whose
    name doesn't match anything in `df` is silently dropped from this
    breakdown, not counted against any class). A class with fewer than
    `min_races_for_ci` SCORED races gets a `note` instead of point
    metrics -- too few to say anything reliable yet, which in practice
    is the honest answer for every class right after a freeze."""
    from pathlib import Path
    predictions_dir = Path(predictions_dir)
    empty = {cls: {"n_races_predicted": 0, "n_races_scored": 0,
                   "note": "No live predictions logged yet."} for cls in CLASSES}
    if not predictions_dir.exists():
        return empty

    frames = [pd.read_csv(p) for p in sorted(predictions_dir.glob("[12][0-9][0-9][0-9].csv"))]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return empty
    all_preds = pd.concat(frames, ignore_index=True)
    all_preds["predicted_at"] = pd.to_datetime(all_preds["predicted_at"], utc=True)
    cutoff = pd.Timestamp(frozen_at)
    cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    live = all_preds[all_preds["predicted_at"] >= cutoff]
    if live.empty:
        return empty

    name_to_circuit = dict(zip(df["event_name"], df["circuit_id"])) if "event_name" in df else {}
    live = live.copy()
    live["circuit_id"] = live["gp"].map(name_to_circuit)
    live = add_circuit_class(live.dropna(subset=["circuit_id"]))

    out = {}
    for cls in CLASSES:
        g = live[live["circuit_class"] == cls]
        if g.empty:
            out[cls] = {"n_races_predicted": 0, "n_races_scored": 0,
                       "note": "No live predictions logged yet for this class."}
            continue
        scored = g.dropna(subset=["actual_position"])
        n_races, n_scored = int(g["gp"].nunique()), int(scored["gp"].nunique())
        entry = {"n_races_predicted": n_races, "n_races_scored": n_scored}
        if n_scored < min_races_for_ci:
            entry["note"] = f"Only {n_scored} scored race(s) in this class since the freeze -- too few to say anything yet."
        else:
            entry["model_mae"] = round(float((scored["predicted_position"] - scored["actual_position"]).abs().mean()), 3)
            entry["grid_mae"] = round(float((scored["grid"] - scored["actual_position"]).abs().mean()), 3)
        out[cls] = entry
    return out
