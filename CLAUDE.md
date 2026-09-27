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
- `data/prebuilt/{year}/{round}/{session}/` bundles every 2025 race and every completed 2026 race (built by `scripts/build_prebuilt.py`, refreshed by `.github/workflows/update-data.yml` — Monday 02:00 UTC plus a 14:00 UTC catch-up pass) for a Streamlit Cloud deploy that can't reach F1's timing API at all. `app/data.py::get_session()`/`race_laps()` try, in order: the prebuilt bundle (`PrebuiltSession`, a duck-typed stand-in for a FastF1 session), the Parquet cache (`race_laps` only), the [OpenF1](https://openf1.org) API (`OpenF1Session`, mapping `/v1/laps,car_data,stints,pit,drivers,sessions,location` into the same DataFrame shapes), then live FastF1. If all three fail, `SessionLoadError` — "This session isn't available yet — it's added automatically a few hours after it ends." The dashboard's race list comes straight from `data/prebuilt/manifest.json` (no FastF1 schedule call on the cloud).
- `build_prebuilt.py` has its own, separate fallback: `_load_with_diagnosis()` classifies a FastF1 failure as `RATE LIMITED` (backs off and retries — see `_RATE_LIMIT_BACKOFFS`), `DOWNLOAD FAILED`, or `NOT PUBLISHED YET` (distinguished via FastF1's own internally-logged "Failed to load X" warnings, captured through `_Fastf1WarningCapture`) before trying `OpenF1Session` directly. Confirmed live: GitHub's hosted runner currently cannot reach F1's timing API at all, and OpenF1's own `car_data` endpoint is separately broken right now — so a build can legitimately only recover laps/results. Each manifest session entry carries `session_status: {source, status, missing}`, where `status` is `"partial"` (e.g. laps/results only) or `"complete"`; `--only-missing` keeps retrying partial sessions every run until a source upgrades them. The Action fails loudly (non-zero exit, `$GITHUB_STEP_SUMMARY`) only if a race whose session ended >6h ago is missing from the manifest *entirely* — partial sessions just get a non-fatal summary warning. `--force` rebuilds a named `--race` regardless of `--only-missing`. Dashboard: a partial session's telemetry views show "Telemetry for this session isn't available yet" (`app.data.session_missing()`) instead of crashing.

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
- `app/data.py::corners()` tries three tiers before giving up: (1) this session's own MultiViewer map via `get_circuit_info()` — can raise `AttributeError` (no map published yet for a `circuit_key`, e.g. a new season or redesigned track like Catalunya's 2026 key change) or `KeyError` (malformed position data breaks the internal telemetry merge — confirmed on 2026 Monaco Race); (2) the same event's map from last season, reused only if lap length is within 1% (`_reuse_prebuilt_corners`, prebuilt-bundle-first — cheap and correct even when this session's own position data is broken, since it reuses the previous build's Distance values directly instead of re-projecting them — then a live lookup of *last year's own* circuit_key, which can differ from this year's); (3) corners estimated from the fastest lap's speed trace (`_speed_trace_corners`: `scipy.signal.find_peaks` on inverted speed, >15 kph drop, prominence-filtered), labelled `C1, C2, ...` with `Estimated=True`. `assign_corners`/`compare_corners` degrade to unlabelled zones if `corners()` still comes back empty. OpenF1-sourced sessions skip tier 1 entirely (`_openf1_corners`, no circuit-map endpoint) but still get tiers 2/3.
- `lap_telemetry()` falls back to `lap.get_car_data().add_distance()` (Distance/Speed/Throttle/Brake/TimeS, no X/Y) when `lap.get_telemetry()` itself fails — same malformed-position-data cause as above, but `get_car_data()` never touches position data so it's unaffected. Check `has_position_data(tel)` before anything that needs a track map (Track dominance shows "unavailable" instead of crashing).
- `get_lap(..., "fastest")` falls back to `laps.pick_fastest(only_by_time=True)` (lowest non-null `LapTime`, ignoring the personal-best flag) when `pick_fastest()` returns `None` — e.g. a driver whose only laps were deleted for track limits. If a driver has zero non-null lap times at all (e.g. a lap-1 retirement — confirmed real case: VER at 2026 Monaco Race), this still raises `DataError`; there's genuinely nothing to recover.
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
