# CLAUDE.md

F1 telemetry analytics + ML. Python 3.10+, FastF1, FastAPI, pandas, scikit-learn, Plotly, Streamlit.
Owner prefers: minimal explanation, working code, casual tone.

## Commands
- Install (full dev — API + ML + tests): `pip install -r requirements-dev.txt`
- Install (dashboard only, e.g. Streamlit Cloud): `pip install -r requirements.txt`
- API: `uvicorn app.main:app --reload` (docs at /docs)
- Dashboard: `streamlit run dashboard/streamlit_app.py`
- Tests: `pytest -q` (must stay offline and fast; needs requirements-dev.txt)
- Train ML: `python -m scripts.train_tyre_model --year 2025 --max-races 10`

## Architecture
- `app/data.py` is the only place that touches FastF1 objects directly: `load_session` (lru_cached), `get_lap`, `lap_telemetry`, `corners`, `clean_laps`, `race_laps`, `to_records`.
- `race_laps(year, gp)` reads `data/processed/{year}/{round}.parquet` if present, else loads the race session from FastF1 and writes it there for next time. Only covers the "R" session's raw laps (used by degradation/pits/strategy/team-report); telemetry-based routes (braking/compare/dominance) always load live.
- `app/analysis/*` and `app/models/*` take **plain pandas DataFrames** and return DataFrames/dicts. Keep them FastF1-free so they're testable with synthetic data. `clean_laps` and `team_report` (both in the FastF1-touching layer) accept either a live session or a plain raw-laps DataFrame, so they work from the Parquet cache too.
- `app/main.py` routes: load session → call pure functions → `to_records()` for JSON. No analysis logic in routes.
- Dashboard imports the same functions directly (does not call the API).
- `demo_data/{year}/{round}/{session}/` bundles 3 races (2025 Italian/Monaco/British GP, Q+R) built by `scripts/build_demo_data.py` for a Streamlit Cloud deploy that can't reach F1's timing API. `app/data.py::get_session()`/`race_laps()` check demo_data first (via `DemoSession`, a duck-typed stand-in for a FastF1 session), then the Parquet cache, then live FastF1. `OFFLINE_MODE=true` (env var or `st.secrets`) restricts the dashboard to demo races only and never touches FastF1, even for the schedule.

## Data conventions
- Telemetry DataFrame columns used: `Distance` (m), `Speed` (kph), `Brake` (bool), `Throttle` (0–100), `TimeS` (s from lap start), `X`, `Y`.
- Lap DataFrames: `LapTimeS` (float s) is added by `clean_laps`. Raw FastF1 timing columns are `Timedelta`; convert with `.dt.total_seconds()`.
- Compounds for strategy/ML: only `SOFT`, `MEDIUM`, `HARD`. Wet races → strategy returns 422.
- All JSON output goes through `to_records()` (handles Timedelta, NaN, inf).

## Gotchas
- FastF1 first load is slow (network). Never call FastF1 in tests; build synthetic frames (see `tests/test_core.py::fake_lap`, `fake_race`).
- `Brake` is boolean in public data. Don't add features that assume brake pressure.
- 2026 regs removed DRS; don't rely on the `DRS` column for 2026+.
- Use `laps.pick_drivers()` (plural); `pick_driver` is deprecated in FastF1 3.x.
- `session.total_laps` may be missing on older FastF1; fall back to `laps.LapNumber.max()`.
- `get_weather_data()` must be called on the same filtered `Laps` object to stay row-aligned (see `clean_laps(with_weather=True)`).
- Real-data endpoints were written against the FastF1 3.x API but only unit-tested offline. When something breaks on a real session, fix it in `app/data.py` first.
- `session.get_circuit_info()` can raise `AttributeError` instead of returning `None` when MultiViewer has no map yet for a `circuit_key` (new season, or a redesigned track — e.g. Catalunya got a new key for 2026). `app/data.py::corners()` catches this, falls back to the previous year's map, and returns an empty frame as a last resort; `assign_corners`/`compare_corners` degrade to unlabelled zones instead of crashing.
- `Session.load()` can return normally (no exception, just a logged warning) while leaving `.laps`/telemetry unloaded — e.g. when `session.f1_api_support` is `False` for that session. Any later access to `.laps`/`.car_data` then raises `fastf1.exceptions.DataNotLoadedError`. `load_session()` verifies both are actually accessible before returning, retries once, and raises `SessionLoadError` (a `DataError` subclass) otherwise — this happens *before* the return so a broken session is never cached by `lru_cache`/`st.cache_resource`. The dashboard shows a "try again / pick another race" message with a Retry button on this specific error.

## Roadmap (pick up in order)
- [x] Verify every endpoint against a real race (e.g. `2025/Monza`) and fix any FastF1 API drift
- [x] Persist processed per-race data to Parquet (`data/processed/{year}/{round}.parquet`) so the API skips FastF1 loads
- [ ] Race outcome predictor: grid, quali gap to pole, team pace rolling avg, circuit type → finish position (time-based CV, no leakage)
- [ ] Overtake probability model from gap, tyre age delta, compound delta, circuit
- [ ] Driver style clustering on throttle/brake traces (early vs late brakers)
- [ ] Safety car / VSC-aware strategy sim (probabilistic pit loss)
- [ ] Undercut/overcut detector from pit sequences
- [ ] Auto race recap: generate a written summary from team report + strategy output
- [ ] Next.js frontend consuming the API; deploy API (Render/Fly) + frontend (Vercel)
- [ ] GitHub Action: after each race weekend, refresh cache + retrain + publish predictions

## Definition of done for new features
1. Pure function in `app/analysis` or `app/models`, taking/returning DataFrames.
2. Offline test with synthetic data in `tests/`.
3. Route in `app/main.py` + view in dashboard if user-facing.
4. README features/API table updated.
