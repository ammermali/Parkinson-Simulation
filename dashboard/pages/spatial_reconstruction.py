from __future__ import annotations
import pandas as pd
import plotly.express as px
import streamlit as st
from dashboard.services.spatial_reconstruction_service import SpatialReconstructionService, SpatialSnapshotError

AGENT_SYMBOL_MAP = {"Neuron": "circle","Microglia": "diamond","Astrocyte": "square","AlphaSynuclein": "cross","AlphaAggregate": "x","Mitochondrion": "triangle-up","Lysosome": "triangle-down"}
DEFAULT_AGENT_SYMBOL = "circle-open"
STATE_COLOR_MAP = {"Healthy": "#2ca02c","Resting": "#1f77b4","Active": "#ff7f0e","Activated": "#ff7f0e","Compromised": "#d62728","Apoptotic": "#9467bd","Ruptured": "#8c564b","Free": "#17becf","Misfolded": "#bcbd22","Oligomer": "#e377c2","Aggregate": "#7f7f7f","LewyBody": "#000000","Cleared": "#c7c7c7"}
DEFAULT_STATE_COLOR = "#9edae5"
PLOT_SYMBOL_MAP = {**AGENT_SYMBOL_MAP,"Other": DEFAULT_AGENT_SYMBOL}
PLOT_COLOR_MAP = {**STATE_COLOR_MAP,"Other": DEFAULT_STATE_COLOR}
PLAYBACK_INTERVAL = "500ms"

def with_stable_visual_categories(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "agent_class" in frame.columns:
        frame["plot_agent_class"] = (frame["agent_class"].astype(str).where(frame["agent_class"].astype(str).isin(AGENT_SYMBOL_MAP),"Other"))
    else:
        frame["plot_agent_class"] = "Other"
    if "state" in frame.columns:
        frame["plot_state"] = (frame["state"].astype(str).where(frame["state"].astype(str).isin(STATE_COLOR_MAP),"Other"))
    else:
        frame["plot_state"] = "Other"
    return frame

def initialize_playback_state(ticks: list[int]) -> None:
    if "spatial_playing" not in st.session_state:
        st.session_state.spatial_playing = False
    if "spatial_selected_tick" not in st.session_state:
        st.session_state.spatial_selected_tick = ticks[0]
    if st.session_state.spatial_selected_tick not in ticks:
        st.session_state.spatial_selected_tick = ticks[0]

def start_playback() -> None:
    st.session_state.spatial_playing = True

def stop_playback() -> None:
    st.session_state.spatial_playing = False

def advance_tick_if_playing(ticks: list[int]) -> None:
    if not st.session_state.spatial_playing:
        return
    current_tick = st.session_state.spatial_selected_tick
    try:
        current_index = ticks.index(current_tick)
    except ValueError:
        current_index = 0
    next_index = (current_index + 1) % len(ticks)
    st.session_state.spatial_selected_tick = ticks[next_index]

def render_filters(service: SpatialReconstructionService,snapshots: pd.DataFrame) -> tuple[list[str], list[str]]:
    agent_classes = service.available_agent_classes(snapshots)
    states = service.available_states(snapshots)
    col_classes, col_states = st.columns(2)
    with col_classes:
        selected_classes = st.multiselect("Agent classes",options=agent_classes,default=agent_classes)
    with col_states:
        selected_states = st.multiselect("States",options=states,default=states,disabled=not states)
    return selected_classes, selected_states

def render_tick_controls(ticks: list[int]) -> int:
    advance_tick_if_playing(ticks)
    col_tick, col_play, col_stop = st.columns([8, 1, 1], vertical_alignment="bottom")
    with col_tick:
        selected_tick = st.select_slider("Tick",options=ticks,key="spatial_selected_tick")
    with col_play:
        st.button(
            "▶ Play",
            width="stretch",
            disabled=st.session_state.spatial_playing,
            on_click=start_playback,
        )
    with col_stop:
        st.button("⏹ Stop",width="stretch",disabled=not st.session_state.spatial_playing,on_click=stop_playback)
    return int(selected_tick)

def render_spatial_frame(*,service: SpatialReconstructionService,snapshots: pd.DataFrame,frame: pd.DataFrame,selected_tick: int) -> None:
    x_min, x_max, y_min, y_max = service.grid_bounds(snapshots)
    hover_columns = [
        column
        for column in ("uid","agent_class","state","rank","compartment", "owner_uid","aggregate_id","x","y")
        if column in frame.columns]
    color_column = "state" if "state" in frame.columns and frame["state"].notna().any() else "agent_class"
    plot_frame = with_stable_visual_categories(frame)
    figure = px.scatter(
        plot_frame,x="x",y="y",color="plot_state",
        symbol="plot_agent_class",color_discrete_map=PLOT_COLOR_MAP,
        symbol_map=PLOT_SYMBOL_MAP,hover_data=hover_columns,
        title=f"Spatial reconstruction — tick {selected_tick}")
    figure.update_traces(marker={"size": 11,"opacity": 0.8,"line": {"width": 0.5}})
    figure.update_xaxes(title="X",range=[x_min - 0.5, x_max + 0.5],tickmode="linear",tick0=x_min,dtick=1,showgrid=True,zeroline=False)
    figure.update_yaxes(title="Y",range=[y_min - 0.5, y_max + 0.5],tickmode="linear",tick0=y_min,dtick=1,showgrid=True,zeroline=False,scaleanchor="x",scaleratio=1)
    figure.update_layout(height=700,legend_title_text="State / agent class",margin={ "l": 30, "r": 30, "t": 60, "b": 30})
    st.plotly_chart(figure,width="stretch",key=f"spatial-reconstruction-{selected_tick}")
    metric_columns = st.columns(4)
    metric_columns[0].metric("Tick", selected_tick)
    metric_columns[1].metric("Visible agents", len(frame))
    metric_columns[2].metric("Agent classes", frame["agent_class"].nunique())
    metric_columns[3].metric("Occupied cells", frame[["x", "y"]].drop_duplicates().shape[0])
    with st.expander("Snapshot rows"):
        st.dataframe(frame, width="stretch", hide_index=True)

@st.cache_data(show_spinner=False)
def load_snapshots(snapshot_path: str, modified_time: float) -> pd.DataFrame:
    del modified_time
    loader = SpatialReconstructionService(snapshot_path)
    return loader.load()

@st.fragment(run_every=PLAYBACK_INTERVAL)
def render_reconstruction_fragment(*,service: SpatialReconstructionService,snapshots: pd.DataFrame,ticks: list[int],selected_classes: list[str],selected_states: list[str]) -> None:
    selected_tick = render_tick_controls(ticks)
    frame = service.frame_at(snapshots,selected_tick,agent_classes=selected_classes,states=selected_states)
    if frame.empty:
        st.warning("No agents match the selected filters.")
        return
    render_spatial_frame(service=service,snapshots=snapshots,frame=frame,selected_tick=selected_tick)
    if st.session_state.spatial_playing:
        st.caption("Playback is running.")
    else:
        st.caption("Playback is stopped.")

st.title("Spatial reconstruction")

st.caption("Post-run reconstruction of the latest simulation on the discrete 2D grid.")
service = SpatialReconstructionService()
if not service.has_simulation():
    st.info("No simulation to explore now.")
    st.stop()

try:
    snapshots = load_snapshots(str(service.snapshot_path),service.snapshot_path.stat().st_mtime)
except SpatialSnapshotError as exc:
    st.error("The spatial snapshot log is invalid.")
    st.code(str(exc), language="text")
    st.stop()
except OSError as exc:
    st.error("The spatial snapshot log could not be read.")
    st.exception(exc)
    st.stop()

if snapshots.empty:
    st.info("No simulation to explore now.")
    st.stop()
ticks = service.available_ticks(snapshots)
if not ticks:
    st.info("No simulation to explore now.")
    st.stop()

initialize_playback_state(ticks)
selected_classes, selected_states = render_filters(service, snapshots)
render_reconstruction_fragment(service=service,snapshots=snapshots,ticks=ticks,selected_classes=selected_classes,selected_states=selected_states)
