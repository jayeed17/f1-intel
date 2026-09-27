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

*(placeholders above — swap in real captures from your deployed dashboard)*

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

`gp` accepts an event name (`Monza`, `Italian Grand Prix`) or round number (`16`). `session`: `R`, `Q`, `S`, `SQ`, `FP1`–`FP3`.

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
  models/              degradation.py, strategy.py, tyre_ml.py
dashboard/streamlit_app.py
scripts/train_tyre_model.py
scripts/smoke_test.py   exercises every route's functions against a real session
scripts/build_prebuilt.py   builds data/prebuilt/ (run locally or by the update-data Action)
tests/                 offline tests on synthetic data
data/processed/{year}/{round}.parquet   Parquet cache for live-loaded races (gitignored)
data/prebuilt/{year}/{round}/{session}/   committed prebuilt bundle: laps/results/corners/telemetry parquet + manifest.json
.github/workflows/update-data.yml   rebuilds data/prebuilt/ twice a week
docs/img/              README screenshots
```

## Data

FastF1 pulls from F1's live timing service. This project is unofficial and not associated with Formula 1.

The dashboard reads from three sources, in order: the **prebuilt bundle** (`data/prebuilt/`, committed to the repo and refreshed automatically — see Deploy above), the **[OpenF1](https://openf1.org) API** for a race that isn't prebuilt yet, and finally a **live FastF1 load** (only reachable when running locally; Streamlit Cloud can't reach F1's timing service). If none of the three have the session yet, the dashboard says so and suggests trying again later. Race laps loaded live are also cached to `data/processed/{year}/{round}.parquet` so a second local request skips FastF1.
