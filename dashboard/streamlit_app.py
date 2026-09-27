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
                      get_lap, get_session, has_position_data, lap_telemetry,
                      load_session, prebuilt_built_at, prebuilt_races, race_laps,
                      race_prediction_features, session_missing)
from app.models.degradation import compound_model, stint_degradation  # noqa: E402
from app.models.race_predictor import FEATURES, TARGET_POS, _prep  # noqa: E402
from app.models.race_predictor import ensure_trained, evaluate, predict_race  # noqa: E402
from app.models.strategy import compare_actual, simulate  # noqa: E402

st.set_page_config(page_title="F1 Intel", layout="wide")
TEMPLATE = "plotly_dark"
COMPOUND_COLORS = {"SOFT": "#E8002D", "MEDIUM": "#FFD12E", "HARD": "#F0F0EC"}

# These three need per-lap car telemetry (heavy); the rest only need lap timing.
TELEMETRY_VIEWS = ["Braking", "Head to head", "Track dominance"]
PARQUET_VIEWS = ["Tyre degradation", "Strategy", "Team report"]
PREDICTOR_VIEWS = ["Race predictor"]
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


@st.cache_data(show_spinner="Evaluating race predictor (walk-forward CV)...")
def cached_predictor_metrics() -> dict:
    _, metrics = evaluate(pd.read_parquet(RACE_DATASET_PATH))
    return metrics


@st.cache_data(show_spinner="Computing feature importance...")
def cached_feature_importance(_pos_pipe) -> pd.DataFrame:
    from sklearn.inspection import permutation_importance
    df = pd.read_parquet(RACE_DATASET_PATH).dropna(subset=[TARGET_POS])
    X, y = _prep(df), df[TARGET_POS]
    r = permutation_importance(_pos_pipe, X, y, n_repeats=5, random_state=42,
                               scoring="neg_mean_absolute_error")
    return pd.DataFrame({"feature": FEATURES, "importance": r.importances_mean}).sort_values("importance")


def _race_options() -> tuple[list[str], dict[str, tuple[int, str]]]:
    """Race dropdown labels, newest first, straight from the prebuilt manifest."""
    races = prebuilt_races()
    labels = [f"{r['year']} {r['name']}" for r in races]
    mapping = {f"{r['year']} {r['name']}": (r["year"], r["name"]) for r in races}
    return labels, mapping


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
        st.caption("Predicts from this race's Qualifying results.")
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

    elif view == "Race predictor":
        if not RACE_DATASET_PATH.exists():
            st.warning("Race dataset not built yet -- run `python -m scripts.build_race_dataset`.")
            return
        pos_pipe, pts_pipe = cached_predictor_models()
        try:
            features = race_prediction_features(int(year), gp)
        except Exception as e:
            st.error(f"Could not load qualifying for this race: {e}")
            return

        preds = predict_race(pos_pipe, pts_pipe, features)
        st.caption("Predicted from this race's qualifying results, as if predicting before the race "
                  "(grid assumed = qualifying classification). Model does not currently beat the "
                  "grid/quali baselines -- see the track record below.")
        show = preds[["predicted_position", "driver", "team_id", "grid", "points_probability"]].rename(
            columns={"predicted_position": "Predicted", "driver": "Driver", "team_id": "Team",
                    "grid": "Grid (quali)", "points_probability": "P(points)"})
        st.dataframe(show, width="stretch", hide_index=True)

        st.subheader("Feature importance (permutation, full dataset)")
        imp = cached_feature_importance(pos_pipe)
        fig = px.bar(imp, x="importance", y="feature", orientation="h", template=TEMPLATE)
        fig.update_layout(height=400, xaxis_title="importance (MAE increase when shuffled)", yaxis_title="")
        st.plotly_chart(fig, width="stretch")

        st.subheader("Season track record: model vs grid baseline")
        metrics = cached_predictor_metrics()
        rows = [{"season": season, "method": method, "position_mae": m[method]["position_mae"]}
               for season, m in metrics.get("by_season", {}).items() for method in ("model", "baseline_grid")]
        record = pd.DataFrame(rows)
        fig = px.bar(record, x="season", y="position_mae", color="method", barmode="group", template=TEMPLATE)
        fig.update_layout(height=400, yaxis_title="position MAE (lower is better)")
        st.plotly_chart(fig, width="stretch")


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
