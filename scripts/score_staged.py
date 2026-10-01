"""Fill in actual results for scripts/predict_staged.py's staged
quali/race-v2 prediction files and report each stage's running accuracy.
Run on the same crons as scripts/score_predictions.py (Monday scores race
results; a quali file can also score as soon as its qualifying session is
in, even mid-week):

    python -m scripts.score_staged --year 2026

Each predictions/{year}_quali_{stage}.csv is scored from the Qualifying
session (quali_position, via the same ranking app.data uses when building
training features); each predictions/{year}_race_v2_{stage}.csv from the
Race session (actual_position) -- same prebuilt -> OpenF1 -> FastF1 chain
as everywhere else. Files are scored independently; a missing/empty file
is skipped, not an error. Like score_predictions.py, this is only a
failure if a file had pending rows and genuinely none of them could be
scored.
"""
from __future__ import annotations

import argparse

import pandas as pd

from app.config import PREDICTIONS_DIR
from app.data import _quali_features_for_prediction, get_session

_STAGES = ("forecast", "post_practice", "post_quali")


def _score_file(path, session_code: str, actual_col: str) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return df
    df[actual_col] = df[actual_col].astype("Float64")
    df["scored_at"] = df["scored_at"].astype("object")

    pending = df[actual_col].isna()
    n_pending_before = int(pending.sum())
    if n_pending_before == 0:
        print(f"  {path.name}: already fully scored ({len(df)} rows)")
        return df

    now = pd.Timestamp.now(tz="UTC").isoformat()
    n_scored_now = 0
    for gp in df.loc[pending, "gp"].unique():
        year = int(str(df.loc[df["gp"] == gp, "predicted_at"].iloc[0])[:4])
        try:
            s = get_session(year, gp, session_code, telemetry=False)
            if session_code == "Q":
                actual = _quali_features_for_prediction(s).set_index("driver")["quali_position"].to_dict()
            else:
                actual = {r.get("Abbreviation"): r.get("Position") for _, r in s.results.iterrows()}
        except Exception as e:  # noqa: BLE001
            print(f"  ! {gp} ({session_code}): result not available yet ({e})")
            continue
        mask = (df["gp"] == gp) & df[actual_col].isna()
        df.loc[mask, actual_col] = df.loc[mask, "driver"].map(actual)
        newly = mask & df[actual_col].notna()
        df.loc[newly, "scored_at"] = now
        n_scored_now += int(newly.sum())
        if int(newly.sum()):
            print(f"  scored {gp} ({path.name}): {int(newly.sum())} rows")

    df.to_csv(path, index=False)
    if n_pending_before > 0 and n_scored_now == 0:
        print(f"  ! {path.name}: {n_pending_before} pending, none scored yet")
    return df


def score_year(year: int) -> None:
    for stage in ("forecast", "post_practice"):
        _score_file(PREDICTIONS_DIR / f"{year}_quali_{stage}.csv", "Q", "actual_quali_position")
    for stage in _STAGES:
        _score_file(PREDICTIONS_DIR / f"{year}_race_v2_{stage}.csv", "R", "actual_position")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, required=True)
    args = p.parse_args()
    score_year(args.year)


if __name__ == "__main__":
    main()
