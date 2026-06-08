from __future__ import annotations
from typing import Any
import pandas as pd
import plotly.express as px
import streamlit as st
from dashboard.services.metrics_service import MetricsDataError, MetricsService

ENVIRONMENT_METRICS = ["debris","inflammation","dopamine"]
NEURON_METRICS = ["neurons_healthy","neurons_compromised","neurons_apoptotic","neurons_ruptures"]
ALPHA_METRICS = ["free_alpha","alpha_aggregate"]

def format_metric_value(value: Any) -> str:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return "—"
    return f"{float(number):.5g}"

def display_optional_integer(value: Any) -> str:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return "—"
    return str(int(number))

def render_latest_tick_metrics(metrics: pd.DataFrame) -> None:
    if metrics.empty:
        return
    latest = metrics.iloc[-1]
    cards = st.columns(4)
    cards[0].metric("Latest tick", int(latest["tick"]))
    cards[1].metric("Debris", format_metric_value(latest.get("debris")))
    cards[2].metric("Inflammation", format_metric_value(latest.get("inflammation")))
    cards[3].metric("Dopamine", format_metric_value(latest.get("dopamine")))

def render_tick_chart(*, metrics: pd.DataFrame,columns: list[str],empty_message: str) -> None:
    available = [column for column in columns if column in metrics.columns]
    if not available:
        st.info(empty_message)
        return
    st.line_chart(metrics.set_index("tick")[available],width="stretch")

def render_tick_metrics(service: MetricsService) -> None:
    if not service.has_tick_metrics():
        st.info("No per-tick metrics are available.")
        st.caption("Expected file: `output/metrics/tick_metrics.csv`.")
        return
    try:
        metrics = service.load_tick_metrics()
    except MetricsDataError as exc:
        st.error("The per-tick metrics file is invalid.")
        st.code(str(exc),language="text")
        return
    if metrics.empty:
        st.info("The per-tick metrics file is empty.")
        return
    render_latest_tick_metrics(metrics)
    st.divider()
    (tab_environment,tab_neurons,tab_alpha,tab_custom,tab_data) = st.tabs(["Environment","Neurons","Alpha-synuclein","Custom chart","Raw data"])
    with tab_environment:
        render_tick_chart(metrics=metrics,columns=ENVIRONMENT_METRICS,empty_message=("No environment metrics are available."))
    with tab_neurons:
        render_tick_chart(metrics=metrics,columns=NEURON_METRICS,empty_message=("No neuron metrics are available."))
    with tab_alpha:
        render_tick_chart(metrics=metrics,columns=ALPHA_METRICS,empty_message=("No alpha-synuclein metrics are available."))
    with tab_custom:
        available = service.tick_metric_columns(metrics)
        selected = st.multiselect("Metrics", options=available, default=[metric for metric in ENVIRONMENT_METRICS if metric in available],key="custom-tick-metrics")
        if selected:
            st.line_chart(metrics.set_index("tick")[selected],width="stretch")
        else:
            st.info("Select at least one metric.")
    with tab_data:
        st.dataframe(metrics, width="stretch", hide_index=True)

def render_mechanism_overview(*,service: MetricsService,run_report: dict[str, Any]) -> None:
    counts = service.mechanism_counts_frame(run_report)
    total_edges = run_report.get("total_edges", 0)
    columns = st.columns(3)
    columns[0].metric("Total event edges", total_edges)
    columns[1].metric("Mechanisms", len(counts))
    columns[2].metric("Mechanism occurrences", int(counts["count"].sum()) if not counts.empty else 0)
    if counts.empty:
        st.info("The report contains no mechanism counts.")
        return
    maximum = min(len(counts), 30)
    visible = st.slider("Mechanisms to display",min_value=1,max_value=maximum,value=min(15, maximum),key="mechanism-count-limit")
    chart_data = counts.head(visible)
    figure = px.bar(chart_data,x="count",y="mechanism",orientation="h",title="Mechanism occurrences")
    figure.update_layout(height=max(400, visible * 28),xaxis_title="Occurrences",
        yaxis_title="Mechanism",yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(figure, width="stretch")
    st.dataframe(counts, width="stretch", hide_index=True)

def render_selected_mechanism(*,service: MetricsService,run_report: dict[str, Any]) -> None:
    counts = service.mechanism_counts_frame(run_report)
    if counts.empty:
        st.info("No mechanisms are available.")
        return
    mechanisms = counts["mechanism"].astype(str).tolist()
    selected = st.selectbox("Mechanism",options=mechanisms,key="selected-mechanism-detail")
    details = service.mechanism_details(run_report, selected)
    profile = service.mechanism_profile_frame(run_report, selected)
    total_occurrences = int(details.get("count", 0) or 0)
    active_profile = profile[profile["count"] > 0] if not profile.empty else pd.DataFrame()
    active_ticks = len(active_profile)
    peak_count = int(profile["count"].max()) if not profile.empty else 0
    peak_tick = "—"
    if not profile.empty and peak_count > 0:
        peak_row = profile.loc[profile["count"].idxmax()]
        peak_tick = str(int(peak_row["tick"]))
    cards = st.columns(6)
    cards[0].metric("Occurrences", total_occurrences)
    cards[1].metric("First tick", display_optional_integer(details.get("first_tick")))
    cards[2].metric("Last tick", display_optional_integer(details.get("last_tick")))
    cards[3].metric("Active ticks", active_ticks)
    cards[4].metric("Peak occurrences", peak_count)
    cards[5].metric("Peak tick", peak_tick)
    if profile.empty:
        st.info("The report does not contain per-tick data for this mechanism.")
        return
    chart_mode = st.radio("Display",options=["Occurrences per tick","Share of total occurrences"],horizontal=True,key="mechanism-profile-mode")
    if chart_mode == "Occurrences per tick":
        figure = px.bar(profile,x="tick",y="count",title=(f"{selected}: occurrences per tick"))
        figure.update_layout(xaxis_title="Tick",yaxis_title="Occurrences",height=480)
    else:
        figure = px.area(profile,x="tick",y="share_percent",title=f"{selected}: concentration by tick")
        figure.update_layout(xaxis_title="Tick",yaxis_title="Share of total occurrences (%)",height=480)
    first_tick = pd.to_numeric(details.get("first_tick"), errors="coerce")
    last_tick = pd.to_numeric(details.get("last_tick"), errors="coerce")
    if not pd.isna(first_tick):
        figure.add_vline(x=float(first_tick), line_dash="dash",annotation_text="First",annotation_position="top left")
    if not pd.isna(last_tick) and last_tick != first_tick:
        figure.add_vline(x=float(last_tick), line_dash="dash", annotation_text="Last", annotation_position="top right")
    st.plotly_chart(figure,width="stretch",key=f"mechanism-profile-{selected}")
    if total_occurrences > 0:
        mean_per_active_tick = total_occurrences / active_ticks if active_ticks else 0
        st.caption(f"`{selected}` appears {total_occurrences} times over {active_ticks} active ticks, with an average of {mean_per_active_tick:.3g} occurrences per active tick.")
    with st.expander("Per-tick values",expanded=False,):
        st.dataframe(
            profile,
            width="stretch",
            hide_index=True,
            column_config={
                "tick": st.column_config.NumberColumn("Tick",format="%d"),
                "count": st.column_config.NumberColumn("Occurrences",format="%d"),
                "share_percent": (st.column_config.NumberColumn("Share",format="%.3f%%"))})
    probability_summary = details.get("probability_summary", {})
    rng_summary = details.get("rng_summary", {})
    if probability_summary or rng_summary:
        with st.expander("Stochastic summaries",expanded=False):
            stochastic_rows = []
            if probability_summary:
                stochastic_rows.append({"measure": "Probability",**probability_summary})
            if rng_summary:
                stochastic_rows.append({"measure": "RNG value",**rng_summary})
            st.dataframe(pd.DataFrame(stochastic_rows),width="stretch",hide_index=True)

def render_selected_biology(*,service: MetricsService,run_report: dict[str, Any]) -> None:
    frame = service.selected_mechanism_summary_frame(run_report)
    if frame.empty:
        st.info("No biological summaries are available.")
        return
    groups = sorted(frame["group"].dropna().astype(str).unique().tolist())
    selected_group = st.selectbox("Biological group",options=groups,key="biological-group")
    group_frame = frame[frame["group"].astype(str) == selected_group]
    st.dataframe(group_frame[["metric", "value"]], width="stretch", hide_index=True)

def render_lifecycle(*,service: MetricsService,run_report: dict[str, Any]) -> None:
    lifecycle = service.mechanism_lifecycle_frame(run_report)
    if lifecycle.empty:
        st.info("No mechanism lifecycle information is available.")
        return
    st.dataframe(
        lifecycle,
        width="stretch",
        hide_index=True,
        column_config={
            "mechanism": st.column_config.TextColumn("Mechanism",width="large"),
            "count": st.column_config.NumberColumn("Count",format="%d"),
            "first_tick": st.column_config.NumberColumn("First tick",format="%d"),
            "last_tick": st.column_config.NumberColumn("Last tick",format="%d")})

def run_mechanism_computation(*, service: MetricsService, include_by_tick: bool) -> bool:
    try:
        with st.spinner("Computing mechanism metrics...",show_time=True):
            service.compute_mechanism_metrics(include_by_tick=include_by_tick)
    except Exception as exc:
        st.error("Mechanism metrics computation failed.")
        st.exception(exc)
        return False
    st.success("Mechanism metrics computed successfully.")
    return True


def render_mechanism_metrics(service: MetricsService) -> None:
    if not service.has_mechanism_metrics():
        st.info("Mechanism metrics have not been computed yet.")
        st.caption("Expected output: `output/metrics/mechanism_metrics_latest.json`.")
        if not service.has_event_logs():
            st.warning("The event log is missing.")
            st.caption("Expected input: `output/run_logs/events.jsonl`.")
            return
        include_by_tick = st.checkbox("Include per-tick mechanism counts",value=True,key="include-by-tick")
        if st.button("Compute mechanism metrics",type="primary",width="stretch"):
            if run_mechanism_computation(service=service,include_by_tick=include_by_tick):
                st.rerun()
        return
    try:
        report = service.load_mechanism_metrics()
    except MetricsDataError as exc:
        st.error("The mechanism metrics report is invalid.")
        st.code(str(exc),language="text")
        return
    run_report = service.current_run_report(report)
    if not run_report:
        st.warning("The mechanism metrics report does not contain a valid run.")
        return
    status_column, recompute_column = st.columns([4, 1], vertical_alignment="center")
    with status_column:
        st.success("Loaded `output/metrics/mechanism_metrics_latest.json`.")
    with recompute_column:
        recompute = st.button("Recompute",width="stretch")
    if recompute:
        if not service.has_event_logs():
            st.error("Cannot recompute: `output/run_logs/events.jsonl` is missing.")
            return
        if run_mechanism_computation(service=service,include_by_tick=True):
            st.rerun()
        return
    (tab_overview,tab_explorer,tab_biology,tab_lifecycle,tab_raw) = st.tabs(["Overview", "Mechanism explorer", "Biological summaries", "Lifecycle","Raw report"])
    with tab_overview:
        render_mechanism_overview(service=service,run_report=run_report)
    with tab_explorer:
        render_selected_mechanism(service=service,run_report=run_report)
    with tab_biology:
        render_selected_biology(service=service,run_report=run_report)
    with tab_lifecycle:
        render_lifecycle(service=service,run_report=run_report)
    with tab_raw:
        st.json(report)

st.title("Post-run metrics")
st.caption("Explore per-tick simulation metrics and mechanism-level summaries.")
service = MetricsService()
if not service.has_simulation_output():
    st.info("No simulation to explore now.")
    st.stop()
tab_tick, tab_mechanisms = st.tabs(["Per-tick metrics","Mechanism metrics"])
with tab_tick:
    render_tick_metrics(service)
with tab_mechanisms:
    render_mechanism_metrics(service)
