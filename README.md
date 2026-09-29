# F1 Intel

Race-by-race Formula 1 telemetry analytics and ML. Where drivers brake, where they lose time, how fast tyres fall off, what the optimal strategy was, and where each team can improve.

Built on [FastF1](https://docs.fastf1.dev) (official F1 timing feed, 2018+), FastAPI, scikit-learn, Plotly and Streamlit.

## Live demo

**[f1-intel.streamlit.app](https://f1-intel.streamlit.app/)**

## Screenshots

| Braking zones | Head to head |
|---|---|
| ![Braking zones](docs/img/braking.png) | ![Head to head](docs/img/head_to_head.png) |

| Track dominance | Strategy simulator |
|---|---|
| ![Track dominance](docs/img/dominance.png) | ![Strategy simulator](docs/img/strategy.png) |

## Features

| Module | What it answers |
|---|---|
| Braking zones | Where does a driver brake, entry/min/exit speed, avg decel (g), per corner |
| Head to head | Speed/throttle traces, cumulative time delta, corner-by-corner brake point and min-speed diffs |
| Track dominance | Who is fastest in each minisector, drawn on the track map |
| Tyre degradation | Fuel-corrected deg rate (s/lap) per stint and per compound |
| Strategy simulator | Brute-force 1–3 stop plans vs what each driver actually ran, seconds lost vs optimal |
| Team report | Sector gaps, speed-trap deficit, race pace, pit lane time, auto "where to improve" notes |
| Tyre ML model | Gradient-boosted model predicting lap-time loss from tyre age, compound, fuel, temps, circuit, validated leave-circuit-out |
| Race predictor | Positions-gained + DNF models combined via Monte Carlo simulation into P(win)/P(podium)/P(points) per driver; dev/holdout CV'd against grid/quali baselines — [track record and honest results below](#race-outcome-predictor) |

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt   # dashboard + API + ML training + tests

uvicorn app.main:app --reload                 # API → http://127.0.0.1:8000/docs
streamlit run dashboard/streamlit_app.py      # dashboard
pytest -q                                     # offline tests
python -m scripts.train_tyre_model --year 2025 --max-races 10   # train ML model
```

Only running the dashboard (e.g. for a Streamlit Cloud–style deploy)? `pip install -r requirements.txt` alone is enough — it skips the FastAPI/scikit-learn/pytest deps the dashboard doesn't use at runtime.

First load of any session downloads from F1's feed (30–90 s). After that it's served from `cache/`.

## Deploy your own

1. Fork/push this repo to your own GitHub account.
2. On [share.streamlit.io](https://share.streamlit.io), click "New app" and point it at your repo.
3. Set **Main file path** to `dashboard/streamlit_app.py`, **Branch** to `main`, **Python version** to `3.11`, then deploy. `requirements.txt` (dashboard-only deps) is picked up automatically.

Streamlit Cloud can't reach F1's live timing feed at all, so the dashboard is backed by `data/prebuilt/` — every 2025 race and every completed 2026 race, rebuilt automatically by `.github/workflows/update-data.yml` every Monday and Sunday night (or on demand via workflow_dispatch), which commits and pushes any new data straight to this repo. A race that isn't prebuilt yet falls back to the [OpenF1](https://openf1.org) API, then to a live FastF1 load if you're running locally.

## API

| Method | Path | Example |
|---|---|---|
| GET | `/health` | |
| GET | `/schedule/{year}` | `/schedule/2025` |
| GET | `/{year}/{gp}/drivers?session=R` | `/2025/Monza/drivers` |
| GET | `/{year}/{gp}/braking/{driver}?session=R&lap=fastest` | `/2025/Monza/braking/LEC` |
| GET | `/{year}/{gp}/compare?a=&b=&session=Q` | `/2025/Monza/compare?a=VER&b=NOR` |
| GET | `/{year}/{gp}/dominance?drivers=&session=Q` | `/2025/Monza/dominance?drivers=VER,NOR,LEC` |
| GET | `/{year}/{gp}/degradation` | `/2025/Monza/degradation` |
| GET | `/{year}/{gp}/pits` | `/2025/Monza/pits` |
| GET | `/{year}/{gp}/strategy?max_stops=2` | `/2025/Monza/strategy` |
| GET | `/{year}/{gp}/team-report?session=R` | `/2025/Monza/team-report` |
| POST | `/predict/tyre` | `[{"TyreLife":18,"LapNumber":30,"Compound":"MEDIUM","TrackTemp":42,"Circuit":"Italian Grand Prix"}]` |
| GET | `/predict/race/{year}/{gp}` | `/predict/race/2025/Monza` |

`gp` accepts an event name (`Monza`, `Italian Grand Prix`) or round number (`16`). `session`: `R`, `Q`, `S`, `SQ`, `FP1`–`FP3`.

## Race outcome predictor

Two small `HistGradientBoosting` models, not one direct finish-position regressor (an earlier version of this that predicted finish position directly lost to the grid baseline on every metric — see git history):

- **Positions-gained regressor**: predicts `finish − grid` for classified finishers only. Grid position alone already explains most of the variance in an F1 result, so the model only has to learn the (much smaller, much easier) correction on top of it — and heavy regularization (`max_leaf_nodes=7`, `min_samples_leaf=30`, early stopping) lets it shrink toward "no change" when there's genuinely nothing to add.
- **DNF classifier**: predicts P(DNF) from a small feature set (driver/team rolling DNF rate, grid, circuit type, reg-change flag).
- **Monte Carlo simulation** (10,000 runs per race): each run samples a DNF per driver from P(DNF) (classified at a position resampled from the historical distribution of where retirees actually finished) and, for finishers, `grid + predicted delta + noise` (noise ~ walk-forward CV residual std). Ranking every run's raw positions and averaging gives P(win)/P(podium)/P(points) per driver, plus an expected position for the predicted running order.

Trained on one row per driver per race (2022–2026, `data/model/race_dataset.parquet`, built by `scripts/build_race_dataset.py`) — grid, qualifying gap to pole, rolling driver/team form, team pace gap, DNF rate, circuit type. Pit-lane starts (`GridPosition == 0` in the raw data) are remapped to the back of the grid (the actual number of cars that started that race) plus a `grid_pit_lane` flag feature — an earlier version of this remap used `max(everyone else's grid) + 1`, which could overshoot the real field size when grid numbers have gaps (confirmed on 2022 round 5: it produced `grid=21` in a 20-car race). Fixing it only touched 2 of 2146 rows and moved the holdout numbers by less than the width of their own confidence intervals below — see `tests/test_build_race_dataset.py`.

**Evaluation protocol**: every design decision (features, model architecture, hyperparameters) was chosen using expanding-window walk-forward CV on 2022–2024 ("dev") only — `select_hyperparams()`/`select_dnf_hyperparams()` raise immediately if ever handed a 2025+ row, so this is enforced in code, not just convention (`tests/test_race_predictor.py::test_select_hyperparams_refuses_holdout_rows`). 2025–2026 ("holdout") was then evaluated exactly once and is reported below as-is.

**Holdout result (2025–2026, 38 races, 787 driver-rows)**, model vs **finish = grid** vs **finish = qualifying position**:

| metric | model | grid | quali |
|---|---|---|---|
| position MAE (↓) | **3.33** | 3.37 | 3.32 |
| Spearman (↑) | 0.66 | 0.65 | 0.66 |
| top-3 hit rate | **0.84** | 0.82 | 0.82 |
| winner accuracy | 0.53 | **0.66** | **0.66** |
| points F1 | 0.77 | 0.77 | **0.78** |

Probabilistic (Brier / log loss, lower is better) vs a baseline that turns grid position into a probability (empirical P(outcome \| grid slot) from 2022–2024 only):

| outcome | model Brier | baseline Brier | model log loss | baseline log loss |
|---|---|---|---|---|
| win | 0.0317 | **0.0261** | 0.105 | **0.101** |
| podium | 0.0661 | **0.0610** | 0.216 | **0.203** |
| points | **0.1614** | 0.1626 | **0.496** | 0.500 |

**How strong is that claim, really?** Bootstrap 95% CIs (2,000 resamples of whole holdout races, not rows — see `bootstrap_diff_ci()`) on model-minus-baseline:

| difference (model − baseline) | mean | 95% CI | significant? |
|---|---|---|---|
| position MAE vs grid | −0.041 | [−0.163, +0.067] | **no** — CI includes 0 |
| points Brier vs grid-probability | −0.0012 | [−0.0077, +0.0053] | **no** — CI includes 0 |
| win Brier vs grid-probability | **+0.0057** | **[+0.0016, +0.0095]** | **yes** — model is reliably worse |

**Honest result: still doesn't clearly win, and now we know exactly how honest that "closer" claim is.** The apparent edges on position MAE and points Brier are not statistically distinguishable from noise at this sample size (38 races) — the CIs straddle zero, so don't read those as real wins. The one place the data *does* support a confident claim is the opposite direction: the model is reliably worse than the simple grid-probability baseline at calibrating win probability. **Closest**: points classification (F1 ties the grid baseline, Brier/log loss point estimates edge ahead, though not significantly). **Furthest behind, and the only difference with a CI that doesn't cross zero**: win probability calibration — with ~2,100 training rows, there just isn't enough signal to beat grid position at predicting *who wins*, only some evidence (not yet significant) of an edge at predicting *who scores*. See `app/models/race_predictor.py`'s module docstring, the dashboard's "Race predictor" view (calibration chart + season track record), or `models/race_predictor_metrics.json` (regenerated by `scripts.train`-style calls, gitignored) for the full dev+holdout+per-season breakdown.

Weekly predictions are logged automatically: `scripts/predict_next_race.py` runs after qualifying (Saturday 20:00 UTC) and appends predicted position + P(win)/P(podium)/P(points) per driver to `predictions/{year}.csv`; `scripts/score_predictions.py` runs after the race (Monday) and fills in actual results. Both fail loudly (non-zero exit) if expected data is missing rather than silently doing nothing.

## Method notes and limits

- **Brake data is on/off, not pressure**, and car data is ~3.7 Hz. Brake points are accurate to roughly one sample (~20 m at speed).
- **Fuel correction** adds `0.035 s × (lap − 1)` (configurable in `app/config.py`), a public rule-of-thumb estimate.
- **Strategy model** assumes new tyres each stint and a linear deg curve per compound, fitted field-wide on green-flag laps relative to each driver's median (removes car pace). It ignores traffic, safety cars and tyre cliffs; treat totals as relative, not absolute.
- **Team report** compares public timing only. Real teams have far richer data; this finds visible symptoms, not root causes.

## Project structure

```
app/
  main.py              FastAPI routes
  config.py            constants (fuel effect, pit loss, thresholds)
  data.py              session loading, caching, lap/telemetry helpers, JSON serialisation
  analysis/            braking.py, delta.py, pits.py, team_report.py
  models/              degradation.py, strategy.py, tyre_ml.py, race_predictor.py
dashboard/streamlit_app.py
scripts/train_tyre_model.py
scripts/smoke_test.py   exercises every route's functions against a real session
scripts/build_prebuilt.py   builds data/prebuilt/ (run locally or by the update-data Action)
scripts/build_race_dataset.py   builds data/model/race_dataset.parquet (run locally, not by any Action)
scripts/predict_next_race.py   logs a prediction row per driver to predictions/{year}.csv (Action: Saturday 20:00 UTC)
scripts/score_predictions.py   fills in actual results once a race is over (Action: Monday)
tests/                 offline tests on synthetic data
data/processed/{year}/{round}.parquet   Parquet cache for live-loaded races (gitignored)
data/prebuilt/{year}/{round}/{session}/   committed prebuilt bundle: laps/results/corners/telemetry parquet + manifest.json
data/model/race_dataset.parquet   committed race-predictor training set, one row per driver per race
predictions/{year}.csv   committed public track record of weekly predictions vs actual results
.github/workflows/update-data.yml   rebuilds data/prebuilt/ twice a week; scores/predicts races on its own cron
docs/img/              README screenshots
```

## Data

FastF1 pulls from F1's live timing service. This project is unofficial and not associated with Formula 1.

The dashboard reads from three sources, in order: the **prebuilt bundle** (`data/prebuilt/`, committed to the repo and refreshed automatically — see Deploy above), the **[OpenF1](https://openf1.org) API** for a race that isn't prebuilt yet, and finally a **live FastF1 load** (only reachable when running locally; Streamlit Cloud can't reach F1's timing service). If none of the three have the session yet, the dashboard says so and suggests trying again later. Race laps loaded live are also cached to `data/processed/{year}/{round}.parquet` so a second local request skips FastF1.
