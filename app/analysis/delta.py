"""Lap delta and minisector track dominance."""
from __future__ import annotations

import numpy as np
import pandas as pd


def lap_delta(tel_ref: pd.DataFrame, tel_cmp: pd.DataFrame, step: float = 5.0) -> pd.DataFrame:
    """Cumulative time gap of cmp vs ref along distance. Positive = cmp is behind."""
    d_max = min(tel_ref["Distance"].max(), tel_cmp["Distance"].max())
    d = np.arange(0, d_max, step)
    t_ref = np.interp(d, tel_ref["Distance"], tel_ref["TimeS"])
    t_cmp = np.interp(d, tel_cmp["Distance"], tel_cmp["TimeS"])
    return pd.DataFrame({
        "distance_m": d,
        "delta_s": np.round(t_cmp - t_ref, 3),
        "ref_kph": np.interp(d, tel_ref["Distance"], tel_ref["Speed"]),
        "cmp_kph": np.interp(d, tel_cmp["Distance"], tel_cmp["Speed"]),
    })


def minisector_dominance(tels: dict[str, pd.DataFrame], n: int = 25) -> pd.DataFrame:
    """Fastest driver per minisector (by mean speed). Returns track points with Winner.

    Uses the first driver's telemetry for X/Y coordinates.
    """
    drivers = list(tels)
    d_max = min(t["Distance"].max() for t in tels.values())
    edges = np.linspace(0, d_max, n + 1)
    winners = []
    for i in range(n):
        speeds = {}
        for drv, t in tels.items():
            seg = t[(t["Distance"] >= edges[i]) & (t["Distance"] < edges[i + 1])]
            speeds[drv] = seg["Speed"].mean() if not seg.empty else -np.inf
        winners.append(max(speeds, key=speeds.get))
    ref = tels[drivers[0]]
    pts = ref[ref["Distance"] <= d_max][["Distance", "X", "Y"]].copy()
    pts["Minisector"] = np.clip(np.digitize(pts["Distance"], edges) - 1, 0, n - 1)
    pts["Winner"] = pts["Minisector"].map(dict(enumerate(winners)))
    return pts.reset_index(drop=True)
