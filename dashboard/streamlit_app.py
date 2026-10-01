"""Streamlit dashboard. Run from repo root: streamlit run dashboard/streamlit_app.py"""
from __future__ import annotations

import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fastf1.plotting  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

from app.analysis.braking import assign_corners, braking_zones, compare_corners  # noqa: E402
from app.analysis.delta import lap_delta, minisector_dominance  # noqa: E402
from app.analysis.pits import estimate_pit_loss, pit_stops  # noqa: E402
from app.analysis.team_report import team_report  # noqa: E402
from app.config import RACE_DATASET_PATH  # noqa: E402
from app.data import (DataError, SessionLoadError, clean_laps, corners,  # noqa: E402
                      forecast_features, get_lap, get_session, has_position_data, lap_telemetry,
                      load_session, prebuilt_built_at, prebuilt_races, race_laps,
                      race_prediction_features, race_stage_for, session_missing)
from app.models.degradation import compound_model, stint_degradation  # noqa: E402
from app.models.race_predictor import DEFAULT_DELTA_PARAMS, DEFAULT_DNF_PARAMS  # noqa: E402
from app.models.race_predictor import FEATURES_DELTA, FEATURES_DNF, TARGET_DELTA, TARGET_DNF  # noqa: E402
from app.models.race_predictor import _prep_delta, _prep_dnf  # noqa: E402
from app.models.race_predictor import ensure_trained, predict_race, run_full_evaluation  # noqa: E402
from app.models.race_predictor import compute_live_track_record, frozen_model_spec  # noqa: E402
from app.models.quali_predictor import ensure_trained as ensure_trained_quali  # noqa: E402
from app.models.quali_predictor import predict_quali  # noqa: E402
from app.models.quali_predictor import frozen_model_spec as frozen_quali_spec  # noqa: E402
from app.models.quali_predictor import compute_live_track_record as compute_live_track_record_quali  # noqa: E402
from app.models.race_predictor_v2 import ensure_trained as ensure_trained_race_v2  # noqa: E402
from app.models.race_predictor_v2 import predict_race_v2  # noqa: E402
from app.models.race_predictor_v2 import frozen_model_spec as frozen_race_v2_spec  # noqa: E402
from app.models.race_predictor_v2 import compute_live_track_record as compute_live_track_record_v2  # noqa: E402
from app.models.strategy import compare_actual, simulate  # noqa: E402

st.set_page_config(page_title="F1 Intel", layout="wide")
TEMPLATE = "plotly_dark"
COMPOUND_COLORS = {"SOFT": "#E8002D", "MEDIUM": "#FFD12E", "HARD": "#F0F0EC"}

# These three need per-lap car telemetry (heavy); the rest only need lap timing.
TELEMETRY_VIEWS = ["Braking", "Head to head", "Track dominance"]
PARQUET_VIEWS = ["Tyre degradation", "Strategy", "Team report"]
PREDICTOR_VIEWS = ["Predictions"]
_SESSION_ORDER = ["FP1", "FP2", "FP3", "SQ", "S", "Q", "R"]


def drv_color(drv: str, s) -> str | None:
    try:
        return fastf1.plotting.get_driver_color(drv, session=s)
    except Exception:
        return None


@st.cache_resource(show_spinner="Loading session...", max_entries=2, ttl=3600)
def session_full(year: int, gp: str, kind: str):
    """Full session with car telemetry: prebuilt bundle -> OpenF1 -> live FastF1."""
    return get_session(year, gp, kind, telemetry=True)


@st.cache_data(show_spinner="Loading race data...")
def cached_race_laps(year: int, gp: str) -> pd.DataFrame:
    return race_laps(year, gp)


@st.cache_resource(show_spinner="Training race predictor...")
def cached_predictor_models():
    return ensure_trained(RACE_DATASET_PATH)


@st.cache_data(show_spinner="Evaluating race predictor (walk-forward CV + Monte Carlo)...")
def cached_predictor_report() -> dict:
    return run_full_evaluation(pd.read_parquet(RACE_DATASET_PATH), DEFAULT_DELTA_PARAMS, DEFAULT_DNF_PARAMS)


@st.cache_data(show_spinner="Computing feature importance...")
def cached_feature_importance(_delta_pipe, _dnf_pipe) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.inspection import permutation_importance
    df = pd.read_parquet(RACE_DATASET_PATH)
    delta_df = df[df[TARGET_DNF] == 0].dropna(subset=[TARGET_DELTA])
    r_delta = permutation_importance(_delta_pipe, _prep_delta(delta_df), delta_df[TARGET_DELTA],
                                     n_repeats=5, random_state=42, scoring="neg_mean_absolute_error")
    dnf_df = df.dropna(subset=[TARGET_DNF])
    r_dnf = permutation_importance(_dnf_pipe, _prep_dnf(dnf_df), dnf_df[TARGET_DNF],
                                   n_repeats=5, random_state=42, scoring="neg_log_loss")
    imp_delta = pd.DataFrame({"feature": FEATURES_DELTA, "importance": r_delta.importances_mean}
                            ).sort_values("importance")
    imp_dnf = pd.DataFrame({"feature": FEATURES_DNF, "importance": r_dnf.importances_mean}
                          ).sort_values("importance")
    return imp_delta, imp_dnf


def _race_options() -> tuple[list[str], dict[str, tuple[int, str]]]:
    """Race dropdown labels, newest first, straight from the prebuilt manifest."""
    races = prebuilt_races()
    labels = [f"{r['year']} {r['name']}" for r in races]
    mapping = {f"{r['year']} {r['name']}": (r["year"], r["name"]) for r in races}
    return labels, mapping


def _predictions_race_options() -> tuple[list[str], dict[str, dict]]:
    """2026 race dropdown, oldest round first: every prebuilt 2026 race,
    plus -- best effort -- the next round not in the manifest yet, so a
    genuine pre-weekend "Forecast" entry can show up. The schedule lookup
    is wrapped in its own try/except and silently skipped if it fails
    (e.g. on Streamlit Cloud, which can't reach FastF1 at all) so the rest
    of the dropdown still works from the prebuilt bundle alone."""
    races = sorted((r for r in prebuilt_races() if r["year"] == 2026), key=lambda r: r["round"])
    options = {f"{r['name']} {r['year']}": {"year": r["year"], "name": r["name"], "round": r["round"]}
              for r in races}
    built_rounds = {r["round"] for r in races}
    try:
        sch = fastf1.get_event_schedule(2026, include_testing=False)
        upcoming = sch[~sch["RoundNumber"].isin(built_rounds)].sort_values("RoundNumber")
        if not upcoming.empty:
            nxt = upcoming.iloc[0]
            options[f"{nxt['EventName']} 2026"] = {"year": 2026, "name": nxt["EventName"],
                                                   "round": int(nxt["RoundNumber"])}
    except Exception:
        pass
    return list(options.keys()), options


def _default_predictions_index(options: dict[str, dict]) -> int:
    """The next race whose result isn't in yet, or the most recent one if
    the whole season (so far) is already classified."""
    labels = list(options.keys())
    for i, label in enumerate(labels):
        info = options[label]
        if race_stage_for(info["year"], info["name"]) != "Result":
            return i
    return max(len(labels) - 1, 0)


@st.cache_resource(show_spinner="Training qualifying predictor...")
def cached_quali_models():
    return ensure_trained_quali(RACE_DATASET_PATH)


@st.cache_resource(show_spinner="Training race predictor v2...")
def cached_race_v2_models():
    return ensure_trained_race_v2(RACE_DATASET_PATH)


def _track_history_table(race_feats: pd.DataFrame) -> pd.DataFrame:
    """Each driver's and their team's history at this circuit (last up to 3
    prior editions), already computed with a correct season cutoff by
    app.data's feature-assembly functions -- just reshaped for display.
    Empty if nobody has prior editions here (new or layout-changed venue)."""
    if race_feats.empty or not (race_feats["driver_circuit_races_here"].fillna(0) > 0).any():
        return pd.DataFrame()
    cols = ["driver", "team_id", "driver_circuit_avg_quali_3", "driver_circuit_avg_finish_3",
           "driver_circuit_races_here", "team_circuit_avg_quali_3", "team_circuit_avg_finish_3",
           "team_circuit_races_here"]
    hist = race_feats[cols].rename(columns={
        "driver": "Driver", "team_id": "Team",
        "driver_circuit_avg_quali_3": "Driver avg quali (last 3)",
        "driver_circuit_avg_finish_3": "Driver avg finish (last 3)",
        "driver_circuit_races_here": "Driver races here",
        "team_circuit_avg_quali_3": "Team avg quali (last 3)",
        "team_circuit_avg_finish_3": "Team avg finish (last 3)",
        "team_circuit_races_here": "Team races here",
    })
    return hist.sort_values("Driver races here", ascending=False)


with st.sidebar:
    race_labels, race_map = _race_options()
    if not race_labels:
        st.error("No prebuilt race data available. Run scripts/build_prebuilt.py.")
        st.stop()
    race_label = st.selectbox("Race", race_labels, index=0)
    year, gp = race_map[race_label]

    view = st.radio("View", TELEMETRY_VIEWS + PARQUET_VIEWS + PREDICTOR_VIEWS)
    if view in TELEMETRY_VIEWS:
        kind = st.selectbox("Session", _SESSION_ORDER, index=_SESSION_ORDER.index("Q"))
    elif view in PREDICTOR_VIEWS:
        kind = "Q"
        st.caption("Has its own 2026 race picker below -- the sidebar race selection doesn't apply here.")
    else:
        kind = "R"
        st.caption("Uses Race session data.")

_race = (int(year), gp)
if st.session_state.get("_prev_race") not in (None, _race):
    gc.collect()
st.session_state["_prev_race"] = _race

st.title(f"{gp} {year} · {kind}")
built_at = prebuilt_built_at(year, gp)
if built_at:
    st.caption(f"Data updated: {built_at}")

_TELEMETRY_UNAVAILABLE_MSG = "Telemetry for this session isn't available yet — it'll be added automatically."


def render_view() -> None:
    if view in TELEMETRY_VIEWS and "telemetry" in session_missing(int(year), gp, kind):
        st.info(_TELEMETRY_UNAVAILABLE_MSG)
        return

    if view == "Braking":
        s = session_full(int(year), gp, kind)
        drivers = sorted(s.laps["Driver"].dropna().unique())
        if not drivers:
            st.warning("No lap data for this session.")
            return
        drv = st.selectbox("Driver", drivers)
        lap = get_lap(s, drv)
        tel = lap_telemetry(lap)
        cn = corners(s)
        zones = assign_corners(braking_zones(tel), cn)
        fig = go.Figure()
        fig.add_scatter(x=tel["Distance"], y=tel["Speed"], name="Speed", line=dict(color=drv_color(drv, s)))
        for _, z in zones.iterrows():
            fig.add_vrect(x0=z["brake_start_m"], x1=z["brake_end_m"], fillcolor="red", opacity=0.2, line_width=0)
        for _, c in cn.iterrows():
            label = c["Label"] + (" (est.)" if c.get("Estimated") else "")
            fig.add_annotation(x=c["Distance"], y=tel["Speed"].max() + 10, text=label, showarrow=False, font_size=10)
        fig.update_layout(template=TEMPLATE, height=450, xaxis_title="Distance (m)", yaxis_title="kph",
                          title=f"{drv} lap {int(lap['LapNumber'])} ({lap['LapTime'].total_seconds():.3f}s) · red = braking")
        st.plotly_chart(fig, width="stretch")
        if not has_position_data(tel):
            st.caption("No track position data for this session — corner labels are placed by distance only.")
        if cn.get("Estimated", pd.Series(dtype=bool)).any():
            st.caption("Corners marked (est.) are estimated from the speed trace, not an official track map.")
        st.dataframe(zones, width="stretch", hide_index=True)

    elif view == "Head to head":
        s = session_full(int(year), gp, kind)
        drivers = sorted(s.laps["Driver"].dropna().unique())
        if len(drivers) < 2:
            st.warning("Not enough drivers with lap data in this session.")
            return
        c1, c2 = st.columns(2)
        a = c1.selectbox("Driver A (reference)", drivers, 0)
        b = c2.selectbox("Driver B", drivers, min(1, len(drivers) - 1))
        ta, tb = lap_telemetry(get_lap(s, a)), lap_telemetry(get_lap(s, b))
        d = lap_delta(ta, tb)
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.25, 0.25], vertical_spacing=0.04)
        fig.add_scatter(x=ta["Distance"], y=ta["Speed"], name=a, line=dict(color=drv_color(a, s)), row=1, col=1)
        fig.add_scatter(x=tb["Distance"], y=tb["Speed"], name=b, line=dict(color=drv_color(b, s), dash="dot"), row=1, col=1)
        fig.add_scatter(x=ta["Distance"], y=ta["Throttle"], name=f"{a} throttle", line=dict(color=drv_color(a, s)), row=2, col=1)
        fig.add_scatter(x=tb["Distance"], y=tb["Throttle"], name=f"{b} throttle", line=dict(color=drv_color(b, s), dash="dot"), row=2, col=1)
        fig.add_scatter(x=d["distance_m"], y=d["delta_s"], name=f"{b} gap to {a}", line=dict(color="#9aa0a6"), row=3, col=1)
        fig.update_layout(template=TEMPLATE, height=700)
        fig.update_yaxes(title_text="kph", row=1, col=1)
        fig.update_yaxes(title_text="throttle %", row=2, col=1)
        fig.update_yaxes(title_text="gap (s)", row=3, col=1)
        st.plotly_chart(fig, width="stretch")
        st.subheader("Corner by corner (positive diff = B higher / brakes later)")
        cn = corners(s)
        comp = compare_corners(ta, tb, cn)
        if "Estimated" in cn.columns and not comp.empty:
            comp = comp.merge(cn[["Label", "Estimated"]].rename(columns={"Label": "corner"}), on="corner", how="left")
        if cn.get("Estimated", pd.Series(dtype=bool)).any():
            st.caption("Corners marked Estimated=True are estimated from the speed trace, not an official track map.")
        st.dataframe(comp, width="stretch", hide_index=True)

    elif view == "Track dominance":
        s = session_full(int(year), gp, kind)
        drivers = sorted(s.laps["Driver"].dropna().unique())
        picks = st.multiselect("Drivers", drivers, drivers[:3])
        n = st.slider("Minisectors", 10, 50, 25)
        if len(picks) >= 2:
            tels = {p: lap_telemetry(get_lap(s, p)) for p in picks}
            if not all(has_position_data(t) for t in tels.values()):
                st.warning("Track map unavailable for this session.")
                return
            pts = minisector_dominance(tels, n)
            cmap = {p: drv_color(p, s) or None for p in picks}
            fig = px.scatter(pts, x="X", y="Y", color="Winner", color_discrete_map=cmap, template=TEMPLATE)
            fig.update_traces(marker_size=6)
            fig.update_yaxes(scaleanchor="x", visible=False)
            fig.update_xaxes(visible=False)
            fig.update_layout(height=600)
            st.plotly_chart(fig, width="stretch")
            st.write(pts.groupby("Winner")["Minisector"].nunique().rename("minisectors won"))
        else:
            st.info("Pick at least 2 drivers to compare.")

    elif view == "Tyre degradation":
        laps = cached_race_laps(int(year), gp)
        cl = clean_laps(laps)
        if cl.empty:
            st.warning("No representative dry-compound laps for this race (wet race?).")
            return
        fig = px.scatter(cl, x="TyreLife", y="LapTimeS", color="Compound", hover_data=["Driver", "LapNumber"],
                         color_discrete_map=COMPOUND_COLORS, template=TEMPLATE)
        fig.update_layout(height=450, yaxis_title="lap time (s)")
        st.plotly_chart(fig, width="stretch")
        st.subheader("Compound model (fuel-corrected, relative to driver median)")
        st.json(compound_model(cl))
        st.subheader("Per stint")
        st.dataframe(stint_degradation(cl).sort_values("DegPerLap"), width="stretch", hide_index=True)

    elif view == "Strategy":
        laps = cached_race_laps(int(year), gp)
        cl = clean_laps(laps)
        model = compound_model(cl)
        stops = pit_stops(laps)
        loss = st.number_input("Pit loss (s)", 5.0, 60.0, estimate_pit_loss(stops))
        total = int(laps["LapNumber"].max())
        if len(model) < 2:
            st.warning("Need two dry compounds with enough clean laps (wet race or sprint?).")
        else:
            sims = simulate(total, model, loss)
            st.subheader("Model-optimal strategies")
            st.dataframe(sims, width="stretch", hide_index=True)
            act = compare_actual(laps, total, model, loss, float(sims["total_s"].iloc[0]))
            if not act.empty:
                fig = px.bar(act, x="Driver", y="lost_vs_optimal_s", color="Team", hover_data=["plan"], template=TEMPLATE)
                fig.update_layout(height=400, yaxis_title="seconds lost vs optimal (model)")
                st.plotly_chart(fig, width="stretch")
            st.subheader("Pit stops")
            st.dataframe(stops, width="stretch", hide_index=True)

    elif view == "Team report":
        laps = cached_race_laps(int(year), gp)
        rep = team_report(laps)
        if rep.empty:
            st.warning("No team data for this race.")
            return
        st.dataframe(rep, width="stretch", hide_index=True)
        gap_cols = [c for c in ["s1_gap", "s2_gap", "s3_gap"] if c in rep]
        long = rep.melt(id_vars="Team", value_vars=gap_cols, var_name="sector", value_name="gap_s")
        fig = px.bar(long, x="Team", y="gap_s", color="sector", barmode="group", template=TEMPLATE)
        fig.update_layout(height=420, yaxis_title="gap to best sector (s)")
        st.plotly_chart(fig, width="stretch")

    elif view == "Predictions":
        if not RACE_DATASET_PATH.exists():
            st.warning("Race dataset not built yet -- run `python -m scripts.build_race_dataset`.")
            return

        labels, options = _predictions_race_options()
        if not labels:
            st.warning("No 2026 race data available yet.")
            return
        pred_label = st.selectbox("2026 race", labels, index=_default_predictions_index(options),
                                  key="predictions_race")
        race_info = options[pred_label]
        pred_year, pred_gp, pred_round = race_info["year"], race_info["name"], race_info["round"]

        stage = race_stage_for(pred_year, pred_gp)
        badge = {"Forecast": "🔵 Forecast", "Post-practice": "🟡 Post-practice",
                "Post-quali": "🟠 Post-quali", "Result": "🟢 Result"}[stage]
        st.markdown(f"### {pred_gp} {pred_year} -- {badge}")

        dataset = pd.read_parquet(RACE_DATASET_PATH)
        circuit_row = dataset[(dataset["season"] == pred_year) & (dataset["round"] == pred_round)]
        circuit_id = circuit_row["circuit_id"].iloc[0] if not circuit_row.empty else None

        try:
            race_feats = (forecast_features(pred_year, circuit_id) if stage in ("Forecast", "Post-practice")
                         else race_prediction_features(pred_year, pred_gp, circuit_id))
        except Exception as e:
            st.error(f"Could not load data for this race: {e}")
            return

        quali_pipe, quali_meta = cached_quali_models()
        quali_preds = predict_quali(quali_pipe, race_feats, quali_meta)

        race_delta_pipe, race_dnf_pipe, race_meta = cached_race_v2_models()
        if stage in ("Forecast", "Post-practice"):
            race_preds = predict_race_v2(race_delta_pipe, race_dnf_pipe, race_feats, race_meta,
                                        quali_pipe=quali_pipe, quali_meta=quali_meta, quali_features=race_feats)
            st.caption("Grid not known yet -- each of the race model's 10k Monte Carlo runs samples its own "
                      "simulated qualifying order from the quali model, instead of using one fixed projection.")
        else:
            race_preds = predict_race_v2(race_delta_pipe, race_dnf_pipe, race_feats, race_meta)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Qualifying**")
            q_show = quali_preds[["predicted_quali_position", "driver", "team_id", "pole_probability",
                                  "top3_probability", "q3_probability", "expected_quali_position"]].copy()
            for c in ("pole_probability", "top3_probability", "q3_probability"):
                q_show[c] = q_show[c] * 100
            q_show = q_show.rename(columns={"predicted_quali_position": "Predicted", "driver": "Driver",
                                            "team_id": "Team", "pole_probability": "P(pole)",
                                            "top3_probability": "P(top 3)", "q3_probability": "P(Q3)",
                                            "expected_quali_position": "Expected pos."})
            st.dataframe(q_show, width="stretch", hide_index=True,
                        column_config={c: st.column_config.NumberColumn(format="%.1f%%")
                                      for c in ("P(pole)", "P(top 3)", "P(Q3)")})
        with col2:
            st.markdown("**Race**")
            r_show = race_preds[["predicted_position", "driver", "team_id", "win_probability",
                                 "podium_probability", "points_probability"]].copy()
            for c in ("win_probability", "podium_probability", "points_probability"):
                r_show[c] = r_show[c] * 100
            r_show = r_show.rename(columns={"predicted_position": "Predicted", "driver": "Driver",
                                            "team_id": "Team", "win_probability": "P(win)",
                                            "podium_probability": "P(podium)",
                                            "points_probability": "P(points)"})
            st.dataframe(r_show, width="stretch", hide_index=True,
                        column_config={c: st.column_config.NumberColumn(format="%.1f%%")
                                      for c in ("P(win)", "P(podium)", "P(points)")})

        st.subheader("Track history")
        hist = _track_history_table(race_feats)
        if hist.empty:
            st.caption("No prior editions of this circuit in the dataset (new or recently changed venue).")
        else:
            st.dataframe(hist, width="stretch", hide_index=True)

        if stage == "Result":
            st.subheader("Predicted vs actual")
            actual = circuit_row[["driver", "quali_position", "target_finish_pos"]].rename(
                columns={"quali_position": "Actual quali", "target_finish_pos": "Actual finish"})
            pred_vs_actual = quali_preds[["driver", "predicted_quali_position"]].rename(
                columns={"predicted_quali_position": "Predicted quali"})
            pred_vs_actual = pred_vs_actual.merge(
                race_preds[["driver", "predicted_position"]].rename(
                    columns={"predicted_position": "Predicted finish"}), on="driver")
            pred_vs_actual = pred_vs_actual.merge(actual, on="driver").sort_values("Actual finish")
            st.dataframe(pred_vs_actual[["driver", "Predicted quali", "Actual quali",
                                        "Predicted finish", "Actual finish"]],
                        width="stretch", hide_index=True)

        with st.expander("Live track record by stage (quali vs baselines, race v1 vs v2 vs grid)"):
            q_spec, v1_spec, v2_spec = frozen_quali_spec(), frozen_model_spec(), frozen_race_v2_spec()
            st.caption("Each stage is scored only from predictions logged at or after its model's own "
                      "freeze date, using scripts/predict_staged.py's per-stage prediction files -- a "
                      "genuinely prospective test, not another look at CV data.")

            st.markdown("**Qualifying model**")
            if q_spec:
                qcols = st.columns(2)
                for qcol, qstage, qlabel in zip(qcols, ("forecast", "post_practice"),
                                                ("Forecast", "Post-practice")):
                    live_q = compute_live_track_record_quali(q_spec["frozen_at"], qstage)
                    with qcol:
                        st.caption(qlabel)
                        if live_q is None:
                            st.info("No predictions logged yet at this stage since the freeze.")
                        elif live_q["n_races_scored"] == 0:
                            st.info(f"{live_q['n_races_predicted']} race(s) predicted, none scored yet.")
                        else:
                            st.metric("Model MAE", live_q["model_mae"])
                            if "baseline_rolling_mae" in live_q:
                                st.caption(f"vs rolling-5 baseline: {live_q['baseline_rolling_mae']}")
                            if "baseline_last_year_mae" in live_q:
                                st.caption(f"vs last-year-here baseline: {live_q['baseline_last_year_mae']}")
                            if live_q.get("pole_brier") is not None:
                                st.caption(f"Pole Brier: {live_q['pole_brier']}")
            else:
                st.caption("Quali model not frozen yet.")

            st.markdown("**Race model (v1 vs v2 vs grid)**")
            if v1_spec:
                rcols = st.columns(3)
                for rcol, rstage, rlabel in zip(rcols, ("forecast", "post_practice", "post_quali"),
                                                ("Forecast", "Post-practice", "Post-quali")):
                    live_v2 = compute_live_track_record_v2(v2_spec["frozen_at"], rstage) if v2_spec else None
                    with rcol:
                        st.caption(rlabel)
                        if live_v2 is None:
                            st.info("No predictions logged yet at this stage since v2's freeze.")
                        elif live_v2["n_races_scored"] == 0:
                            st.info(f"{live_v2['n_races_predicted']} race(s) predicted, none scored yet.")
                        else:
                            st.metric("v2 MAE", live_v2["model_mae"])
                            if "grid_mae" in live_v2:
                                st.caption(f"vs grid baseline: {live_v2['grid_mae']}")
                            if "v2_mae_vs_v1_pairs" in live_v2:
                                st.caption(f"vs v1 (same races): v2 {live_v2['v2_mae_vs_v1_pairs']} / "
                                          f"v1 {live_v2['v1_mae_vs_v1_pairs']}")
                            if live_v2.get("win_brier") is not None:
                                st.caption(f"Win Brier: {live_v2['win_brier']}")
            else:
                st.caption("Race v2 model not frozen yet.")

        with st.expander("Race v1 model diagnostics (feature importance, track record, calibration)"):
            delta_pipe, dnf_pipe, _meta = cached_predictor_models()
            st.caption("Feature importance (permutation, full dataset)")
            imp_delta, imp_dnf = cached_feature_importance(delta_pipe, dnf_pipe)
            ecol1, ecol2 = st.columns(2)
            with ecol1:
                st.caption("Positions-gained (delta) model -- MAE increase when shuffled")
                fig = px.bar(imp_delta, x="importance", y="feature", orientation="h", template=TEMPLATE)
                fig.update_layout(height=350, xaxis_title="importance", yaxis_title="")
                st.plotly_chart(fig, width="stretch")
            with ecol2:
                st.caption("DNF model -- log-loss increase when shuffled")
                fig = px.bar(imp_dnf, x="importance", y="feature", orientation="h", template=TEMPLATE)
                fig.update_layout(height=350, xaxis_title="importance", yaxis_title="")
                st.plotly_chart(fig, width="stretch")

            report = cached_predictor_report()

            st.subheader("Season track record: model vs grid baseline (position MAE)")
            rows = [{"season": season, "method": method,
                    "position_mae": m.get("point_metrics", {}).get(method, {}).get("position_mae")}
                   for season, m in report.get("by_season", {}).items() for method in ("model", "baseline_grid")]
            record = pd.DataFrame(rows).dropna(subset=["position_mae"])
            fig = px.bar(record, x="season", y="position_mae", color="method", barmode="group", template=TEMPLATE)
            fig.update_layout(height=380, yaxis_title="position MAE (lower is better)")
            st.plotly_chart(fig, width="stretch")

            st.subheader("Holdout (2025-2026) calibration: predicted P(points) vs observed frequency")
            cal = report.get("holdout", {}).get("calibration", {}).get("points")
            if cal:
                cal_df = pd.DataFrame(cal)
                fig = px.line(cal_df, x="predicted", y="observed", markers=True, template=TEMPLATE)
                fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                             line=dict(dash="dash", color="gray"))
                fig.update_layout(height=380, xaxis_title="predicted P(points)",
                                 yaxis_title="observed frequency", xaxis_range=[0, 1], yaxis_range=[0, 1])
                st.plotly_chart(fig, width="stretch")
            else:
                st.caption("Not enough holdout rows yet for a calibration curve.")

            holdout_model = report.get("holdout", {}).get("point_metrics", {}).get("model", {})
            holdout_grid = report.get("holdout", {}).get("point_metrics", {}).get("baseline_grid", {})
            if holdout_model and holdout_grid:
                st.caption(f"Holdout (2025-2026, {report['holdout']['n_races']} races): model position MAE "
                          f"{holdout_model['position_mae']} vs grid baseline {holdout_grid['position_mae']} -- "
                          "see the README's \"Race outcome predictor\" section for the full honest comparison "
                          "(point + probabilistic metrics) against both baselines.")

            spec = frozen_model_spec()
            if spec:
                frozen_at = spec["frozen_at"]
                st.subheader(f"Live track record since {frozen_at}")
                st.caption(f"Model v{spec['version']}, frozen {frozen_at} -- no further tuning since. "
                          "Only predictions logged at or after this date count, so this is a genuinely "
                          "prospective test, not another look at CV data.")
                live = compute_live_track_record(frozen_at)
                if live is None:
                    st.info("No races predicted since the freeze yet -- this fills in as the weekly "
                           "Action logs and scores predictions going forward.")
                elif live["n_races_scored"] == 0:
                    st.info(f"{live['n_races_predicted']} race(s) predicted since the freeze, "
                           "none scored yet (race hasn't happened / results not in yet).")
                else:
                    lcols = st.columns(4)
                    lcols[0].metric("Races scored", live["n_races_scored"])
                    lcols[1].metric("Model MAE", live["model_mae"],
                                   delta=round(live["model_mae"] - live["grid_mae"], 3), delta_color="inverse")
                    lcols[2].metric("Grid MAE", live["grid_mae"])
                    if live.get("model_winner_accuracy") is not None:
                        lcols[3].metric("Winner accuracy", f"{live['model_winner_accuracy']:.0%}")
                    brier_bits = [f"win {live['win_brier']}" if "win_brier" in live else None,
                                f"podium {live['podium_brier']}" if "podium_brier" in live else None,
                                f"points {live['points_brier']}" if "points_brier" in live else None]
                    brier_bits = [b for b in brier_bits if b]
                    if brier_bits:
                        st.caption("Brier (lower is better): " + ", ".join(brier_bits))


try:
    render_view()
except SessionLoadError:
    st.error("This session isn't available yet — it's added automatically a few hours after it ends.")
    if st.button("Retry"):
        session_full.clear()
        cached_race_laps.clear()
        load_session.cache_clear()
        st.rerun()
except DataError as e:
    st.error(f"Data not available: {e}")
except Exception as e:
    st.error(f"Couldn't load this view: {e}")
