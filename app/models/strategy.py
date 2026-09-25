"""Brute-force pit strategy simulator on top of a compound pace model."""
from __future__ import annotations

import itertools

import pandas as pd

DRY = ("SOFT", "MEDIUM", "HARD")


def stint_time(offset: float, deg: float, n: int, start_age: int = 1) -> float:
    """Sum of (offset + deg*age) for ages start_age..start_age+n-1."""
    ages = n * start_age + n * (n - 1) / 2
    return n * offset + deg * ages


def plan_time(plan: list[tuple[str, int]], model: dict, pit_loss: float) -> float:
    total = sum(stint_time(model[c]["offset"], model[c]["deg"], n) for c, n in plan)
    return total + pit_loss * (len(plan) - 1)


def _splits(total: int, stops: int, min_stint: int):
    for pits in itertools.combinations(range(min_stint, total - min_stint + 1), stops):
        lens = [b - a for a, b in zip((0, *pits), (*pits, total))]
        if min(lens) >= min_stint:
            yield pits, lens


def simulate(total_laps: int, model: dict, pit_loss: float, max_stops: int = 2,
             min_stint: int = 5, top: int = 10) -> pd.DataFrame:
    """Best plan per compound sequence, ranked. Total is relative race time (s), not absolute."""
    comps = [c for c in DRY if c in model]
    best: dict[tuple, dict] = {}
    for stops in range(1, max_stops + 1):
        for seq in itertools.product(comps, repeat=stops + 1):
            if len(set(seq)) < 2:  # must use two dry compounds
                continue
            for pits, lens in _splits(total_laps, stops, min_stint):
                t = plan_time(list(zip(seq, lens)), model, pit_loss)
                if seq not in best or t < best[seq]["total_s"]:
                    best[seq] = {"stops": stops, "compounds": "-".join(s[0] for s in seq),
                                 "pit_laps": list(pits), "stint_lengths": lens, "total_s": t}
    if not best:
        return pd.DataFrame()
    df = pd.DataFrame(best.values()).sort_values("total_s").head(top).reset_index(drop=True)
    df["gap_s"] = (df["total_s"] - df["total_s"].iloc[0]).round(2)
    df["total_s"] = df["total_s"].round(2)
    return df


def actual_plans(laps: pd.DataFrame, total_laps: int) -> dict[str, dict]:
    """Stint plan each finisher actually ran: {driver: {team, plan[(compound, laps)]}}."""
    out = {}
    df = pd.DataFrame(laps)
    for drv, g in df.groupby("Driver"):
        if g["LapNumber"].max() < total_laps:
            continue
        plan = [(s["Compound"].iloc[0], len(s)) for _, s in g.sort_values("LapNumber").groupby("Stint")]
        out[drv] = {"team": g["Team"].iloc[0], "plan": plan}
    return out


def compare_actual(laps: pd.DataFrame, total_laps: int, model: dict, pit_loss: float,
                   optimal_total: float) -> pd.DataFrame:
    rows = []
    for drv, info in actual_plans(laps, total_laps).items():
        if any(c not in model for c, _ in info["plan"]):
            continue  # wet tyres or compound missing from model
        t = plan_time(info["plan"], model, pit_loss)
        rows.append({"Driver": drv, "Team": info["team"],
                     "plan": " / ".join(f"{c[0]}{n}" for c, n in info["plan"]),
                     "stops": len(info["plan"]) - 1, "model_total_s": round(t, 2),
                     "lost_vs_optimal_s": round(t - optimal_total, 2)})
    return pd.DataFrame(rows).sort_values("lost_vs_optimal_s").reset_index(drop=True) if rows else pd.DataFrame()
