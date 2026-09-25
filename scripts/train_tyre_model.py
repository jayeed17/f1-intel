"""Build tyre dataset from past races and train the ML model.

python -m scripts.train_tyre_model --year 2025 --max-races 10
python -m scripts.train_tyre_model --year 2024 --year 2025
"""
from __future__ import annotations

import argparse

import fastf1
import pandas as pd

from app.config import MODEL_DIR
from app.data import clean_laps, load_session
from app.models.tyre_ml import build_features, train


def collect(years: list[int], max_races: int | None) -> pd.DataFrame:
    frames = []
    for year in years:
        sch = fastf1.get_event_schedule(year, include_testing=False)
        sch = sch[sch["EventDate"] < pd.Timestamp.now()]
        if max_races:
            sch = sch.head(max_races)
        for _, ev in sch.iterrows():
            name = ev["EventName"]
            try:
                s = load_session(year, int(ev["RoundNumber"]), "R", False)
                f = build_features(clean_laps(s, with_weather=True), name)
                f["Year"] = year
                frames.append(f)
                print(f"  {year} {name}: {len(f)} laps")
            except Exception as e:  # missing data, cancelled race, etc.
                print(f"  skip {year} {name}: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, action="append", required=True)
    ap.add_argument("--max-races", type=int, default=None)
    args = ap.parse_args()

    df = collect(args.year, args.max_races)
    if df.empty:
        raise SystemExit("No data collected")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(MODEL_DIR / "tyre_dataset.csv", index=False)
    print(train(df))


if __name__ == "__main__":
    main()
