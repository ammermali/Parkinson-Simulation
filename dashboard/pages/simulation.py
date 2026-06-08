from __future__ import annotations
import pandas as pd
import streamlit as st
from dashboard.services.simulation_service import SimulationService

st.title("Simulation")
service = SimulationService(ranks=4,mode="rule")
status = service.status()
col_start, col_status = st.columns([1, 2], vertical_alignment="center")
with col_start:
    start_clicked = st.button("Start Simulation",type="primary",disabled=status.running,width="stretch")
with col_status:
    if status.running:
        st.success(f"Simulation running — MPI process PID {status.pid}")
    elif status.pid is not None:
        st.info("The latest simulation process is no longer running.")
    else:
        st.info("No simulation is currently running.")
if start_clicked:
    try:
        pid = service.start()
    except Exception as exc:
        st.error("Unable to start the MPI simulation.")
        st.exception(exc)
    else:
        st.success(f"Simulation started — PID {pid}")
        st.rerun()

@st.fragment(run_every="1s")
def render_simulation_monitor() -> None:
    current_status = service.status()
    metrics = service.read_tick_metrics()
    st.subheader("Per-tick environment evolution")
    render_environment_metrics(metrics)
    with st.expander("Simulation console", expanded=current_status.running):
        console = service.read_console_tail()
        if console:
            st.code(console, language="text", wrap_lines=True)
        else:
            st.info("No console output available yet.")
    if current_status.running:
        st.caption("The view refreshes automatically every second.")
    elif not metrics.empty:
        st.success("Simulation output available.")


def render_environment_metrics(metrics: pd.DataFrame) -> None:
    required = [column for column in ("debris","inflammation","dopamine") if column in metrics.columns]
    if metrics.empty:
        st.info("Waiting for the first tick metrics...")
        return
    if "tick" not in metrics.columns:
        st.warning("`tick_metrics.csv` does not contain the `tick` column.")
        return
    if not required:
        st.warning("None of the expected metric columns were found: `debris`, `inflammation`, `dopamine`.")
        return
    chart_data = metrics[["tick", *required]].copy()
    for column in required:
        chart_data[column] = pd.to_numeric(chart_data[column], errors="coerce")
    st.line_chart(chart_data.set_index("tick"), width="stretch")
    if not chart_data.empty:
        latest = chart_data.iloc[-1]
        columns = st.columns(3)
        for widget, metric_name in zip(columns, ("debris", "inflammation", "dopamine")):
            value = latest.get(metric_name)
            widget.metric(metric_name.capitalize(),"—" if pd.isna(value) else f"{value:.3f}")

render_simulation_monitor()
