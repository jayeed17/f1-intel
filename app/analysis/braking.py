"""Braking zones and corner-by-corner comparison.

Public F1 telemetry exposes Brake as on/off (no pressure) at ~3.7 Hz car-data rate,
so brake points are accurate to roughly one sample (~20 m at high speed).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.config import CORNER_WINDOW, MIN_SPEED_DROP_KPH


def braking_zones(tel: pd.DataFrame, min_drop: float = MIN_SPEED_DROP_KPH) -> pd.DataFrame:
    """One row per braking zone.

    tel needs: Distance (m), Speed (kph), Brake (bool), TimeS (s).
    """
    t = tel.sort_values("Distance").reset_index(drop=True)
    b = t["Brake"].astype(bool)
    t["zone"] = (b & ~b.shift(fill_value=False)).cumsum()

    rows = []
    for _, g in t[b].groupby("zone"):
        i0 = g.index[0]
        start, end = g["Distance"].iloc[0], g["Distance"].iloc[-1]
        entry = t["Speed"].iloc[i0 - 1] if i0 > 0 else g["Speed"].iloc[0]
        win = t[(t["Distance"] >= start) & (t["Distance"] <= end + 100)]
        apex_i = win["Speed"].idxmin()
        vmin, apex_d = t.at[apex_i, "Speed"], t.at[apex_i, "Distance"]
        if entry - vmin < min_drop:
            continue
        dt = t.at[apex_i, "TimeS"] - t.at[i0, "TimeS"]
        decel_g = ((entry - vmin) / 3.6 / dt / 9.81) if dt > 0 else np.nan
        rows.append({
            "brake_start_m": round(float(start), 1),
            "brake_end_m": round(float(end), 1),
            "brake_len_m": round(float(end - start), 1),
            "entry_kph": float(entry),
            "min_kph": float(vmin),
            "apex_m": round(float(apex_d), 1),
            "exit_kph": round(float(np.interp(apex_d + 150, t["Distance"], t["Speed"])), 1),
            "speed_drop_kph": float(entry - vmin),
            "avg_decel_g": round(float(decel_g), 2) if pd.notna(decel_g) else None,
        })
    return pd.DataFrame(rows)


def assign_corners(zones: pd.DataFrame, corners: pd.DataFrame, max_ahead: float = 400) -> pd.DataFrame:
    """Label each zone with the corner it's braking for (first corner ahead of brake start).

    corners may be empty if no circuit map is published for this session yet
    (see app.data.corners) — zones are still returned, just unlabelled.
    """
    if zones.empty or corners.empty:
        return zones.assign(corner=pd.Series(dtype=str))
    labels = []
    for start in zones["brake_start_m"]:
        ahead = corners[(corners["Distance"] >= start) & (corners["Distance"] - start <= max_ahead)]
        if not ahead.empty:
            labels.append(ahead.iloc[0]["Label"])
        else:
            labels.append(corners.iloc[(corners["Distance"] - start).abs().argmin()]["Label"])
    return zones.assign(corner=labels)


def _corner_stats(tel: pd.DataFrame, zones: pd.DataFrame, cd: float, label: str) -> dict:
    before, after = CORNER_WINDOW
    w = tel[(tel["Distance"] >= cd - before) & (tel["Distance"] <= cd + after)]
    z = zones[zones["corner"] == label] if "corner" in zones else zones.iloc[0:0]
    return {
        "min_kph": float(w["Speed"].min()) if not w.empty else np.nan,
        "exit_kph": float(np.interp(cd + 150, tel["Distance"], tel["Speed"])),
        "brake_start_m": float(z["brake_start_m"].iloc[0]) if not z.empty else np.nan,
    }


def compare_corners(tel_a: pd.DataFrame, tel_b: pd.DataFrame, corners: pd.DataFrame) -> pd.DataFrame:
    """Per corner: A vs B min speed, exit speed, brake point. Positive *_diff = B higher/later."""
    za = assign_corners(braking_zones(tel_a), corners)
    zb = assign_corners(braking_zones(tel_b), corners)
    rows = []
    for _, c in corners.iterrows():
        a = _corner_stats(tel_a, za, c["Distance"], c["Label"])
        b = _corner_stats(tel_b, zb, c["Distance"], c["Label"])
        rows.append({
            "corner": c["Label"], "distance_m": round(float(c["Distance"]), 1),
            "a_min_kph": a["min_kph"], "b_min_kph": b["min_kph"],
            "min_kph_diff": b["min_kph"] - a["min_kph"],
            "a_exit_kph": a["exit_kph"], "b_exit_kph": b["exit_kph"],
            "exit_kph_diff": b["exit_kph"] - a["exit_kph"],
            "a_brake_m": a["brake_start_m"], "b_brake_m": b["brake_start_m"],
            "brake_point_diff_m": b["brake_start_m"] - a["brake_start_m"],
        })
    return pd.DataFrame(rows).round(1)
