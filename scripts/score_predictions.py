"""Fill in actual race results for previously-logged predictions and report
running accuracy vs the grid baseline. Run after a race weekend:

    python -m scripts.score_predictions --year 2026

For every gp in predictions/{year}.csv with any unscored row, tries to load
its Race session (same prebuilt -> OpenF1 -> FastF1 chain as everywhere
else) and fills in actual_position + scored_at for matching drivers. Exits
non-zero if there were unscored predictions going in and none of them could
be scored -- a real failure, not "nothing to do" (an already-fully-scored
file, or one with no rows at all pending, is not an error).
"""
from __future__ import annotations

import argparse

import pandas as pd

from app.config import PREDICTIONS_DIR
from app.data import get_session


def score_year(year: int) -> pd.DataFrame:
    path = PREDICTIONS_DIR / f"{year}.csv"
    if not path.exists():
        raise SystemExit(f"No predictions logged for {year} yet ({path})")
    df = pd.read_csv(path)
    df["actual_position"] = df["actual_position"].astype("Float64")
    df["scored_at"] = df["scored_at"].astype("object")

    pending = df["actual_position"].isna()
    n_pending_before = int(pending.sum())
    now = pd.Timestamp.now(tz="UTC").isoformat()
    n_scored_now = 0

    for gp in df.loc[pending, "gp"].unique():
        try:
            s = get_session(year, gp, "R", telemetry=False)
        except Exception as e:  # noqa: BLE001
            print(f"  ! {gp}: race result not available yet ({e})")
            continue
        actual = {r.get("Abbreviation"): r.get("Position") for _, r in s.results.iterrows()}
        mask = (df["gp"] == gp) & df["actual_position"].isna()
        df.loc[mask, "actual_position"] = df.loc[mask, "driver"].map(actual)
        newly = mask & df["actual_position"].notna()
        df.loc[newly, "scored_at"] = now
        n_scored_now += int(newly.sum())
        print(f"  scored {gp}: {int(newly.sum())} predictions")

    df.to_csv(path, index=False)

    if n_pending_before > 0 and n_scored_now == 0:
        raise SystemExit(f"{n_pending_before} pending predictions but none could be scored -- "
                        "no race results available from any source for any of them")

    scored = df.dropna(subset=["actual_position"])
    if scored.empty:
        print(f"No predictions scored yet for {year}.")
        return scored

    model_mae = (scored["predicted_position"] - scored["actual_position"]).abs().mean()
    grid_mae = (scored["grid"] - scored["actual_position"]).abs().mean()
    pred_winners = scored[scored["predicted_position"] == 1]
    winner_acc = (pred_winners["actual_position"] == 1).mean() if not pred_winners.empty else float("nan")
    print(f"\n{year} running record ({len(scored)} scored predictions, {scored['gp'].nunique()} races):")
    print(f"  model MAE: {model_mae:.2f}  |  grid-baseline MAE: {grid_mae:.2f}")
    print(f"  model winner accuracy: {winner_acc:.2f}")
    return scored


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, required=True)
    args = p.parse_args()
    score_year(args.year)


if __name__ == "__main__":
    main()
