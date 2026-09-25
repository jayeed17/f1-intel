# CLAUDE.md

F1 telemetry analytics + ML. Python 3.10+, FastF1, FastAPI, pandas, scikit-learn, Plotly, Streamlit.
Owner prefers: minimal explanation, working code, casual tone.

## Commands
- Install: `pip install -r requirements.txt`
- API: `uvicorn app.main:app --reload` (docs at /docs)
- Dashboard: `streamlit run dashboard/streamlit_app.py`
- Tests: `pytest -q` (must stay offline and fast)
- Train ML: `python -m scripts.train_tyre_model --year 2025 --max-races 10`

## Architecture
- `app/data.py` is the only place that touches FastF1 objects directly: `load_session` (lru_cached), `get_lap`, `lap_telemetry`, `corners`, `clean_laps`, `to_records`.
- `app/analysis/*` and `app/models/*` take **plain pandas DataFrames** and return DataFrames/dicts. Keep them FastF1-free (except `team_report`, which takes a session) so they're testable with synthetic data.
- `app/main.py` routes: load session → call pure functions → `to_records()` for JSON. No analysis logic in routes.
- Dashboard imports the same functions directly (does not call the API).

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

## Roadmap (pick up in order)
- [ ] Verify every endpoint against a real race (e.g. `2025/Monza`) and fix any FastF1 API drift
- [ ] Persist processed per-race data to Parquet (`data/processed/{year}/{round}.parquet`) so the API skips FastF1 loads
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
