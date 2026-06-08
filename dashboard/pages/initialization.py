from __future__ import annotations
from collections.abc import Mapping
from typing import Any
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml
from scipy.stats import gaussian_kde
from dashboard.services.initialization_service import InitializationService, InitializationDataError

def display_mapping(value: Any) -> None:
    if value is None:
        st.caption("Not available.")
        return
    if isinstance(value, dict):
        text = yaml.safe_dump(value, sort_keys=False, allow_unicode=True)
        st.code(text, language="yaml", wrap_lines=True)
        st.write(value)

def display_scalar(value: Any, fallback: str = "-") -> Any:
    if value is None:
        return fallback
    try:
        if pd.isna(value):
            return fallback
    except (TypeError, ValueError):
        pass
    return value

def render_summary(service: InitializationService, summary: dict[str, Any]) -> None:
    if not summary:
        st.info("Initialization summary not available.")
        return
    counts_by_class = service.summary_counts_by_class(summary)
    counts_by_rank = service.summary_counts_by_rank(summary)
    metric_columns = st.columns(3)
    metric_columns[0].metric("Total agents", summary.get("total_agents", "-"))
    metric_columns[1].metric("Agent classes", len(counts_by_class))
    metric_columns[2].metric("MPI ranks", counts_by_rank["rank"].nunique() if not counts_by_rank.empty else "-")
    tab_classes, tab_ranks = st.tabs(["Classes", "MPI distribution"])
    with tab_classes:
        if counts_by_class.empty:
            st.info("No class counts available.")
        else:
            st.bar_chart(counts_by_class, x="agent_class", y="count")
            st.dataframe(counts_by_class, width="stretch", hide_index=True)
    with tab_ranks:
        if counts_by_rank.empty:
            st.info("No MPI rank counts available.")
        else:
            pivot = counts_by_rank.pivot_table(index="rank", columns="agent_class", values="count", fill_value=0, aggfunc="sum")
            st.bar_chart(pivot, width="stretch")
            st.dataframe(pivot, width="stretch")

def threshold_key_matches(key: str) -> bool:
    return str(key).lower().endswith("threshold")

def collect_config_thresholds(*, value: Any, prefix: str = "", result: dict[str, float] | None = None) -> dict[str, float]:
    if result is None:
        result = {}
    if not isinstance(value, Mapping):
        return result
    for raw_key, child in value.items():
        key = str(raw_key)
        path = f"{prefix}.{key}" if prefix else key
        key = str(raw_key)
        if isinstance(child, Mapping):
            collect_config_thresholds(value=child, prefix=path, result=result)
            continue
        if not threshold_key_matches(key):
            continue
        if isinstance(child, bool):
            continue
        if isinstance(child, (int, float)):
            result[path] = float(child)
    return result

def threshold_frame_from_agents(agents: pd.DataFrame) -> pd.DataFrame:
    columns = ["uid", "agent_class", "rank", "owner_uid", "threshold", "value"]
    if agents.empty or "config" not in agents.columns:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for _, agent in agents.iterrows():
        config = agent.get("config")
        if not isinstance(config, Mapping):
            continue
        extracted = collect_config_thresholds(value=config)
        for threshold_name, threshold_value in extracted.items():
            rows.append(
                {
                    "uid": str(agent.get("uid", "")),
                    "agent_class": str(agent.get("agent_class", "")),
                    "rank": agent.get("rank"),
                    "owner_uid": agent.get("owner_uid"),
                    "threshold": threshold_name,
                    "value": threshold_value,
                }
            )
    if not rows:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(rows, columns=columns)
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["uid", "agent_class", "threshold", "value"])
    return frame.sort_values(["agent_class", "threshold", "uid"], kind="stable").reset_index(drop=True)


def short_threshold_name(path: str) -> str:
    parts = str(path).split(".")
    if len(parts) <= 2:
        return str(path)
    return ".".join(parts[-2:])


def available_thresholds_for_class(thresholds: pd.DataFrame, *, agent_class: str) -> list[str]:
    if thresholds.empty:
        return []
    frame = thresholds[thresholds["agent_class"].astype(str) == str(agent_class)]
    return sorted(frame["threshold"].dropna().astype(str).unique().tolist())

def threshold_distribution_for_class(thresholds: pd.DataFrame, *, agent_class: str, threshold: str) -> pd.DataFrame:
    if thresholds.empty:
        return thresholds.copy()
    frame = thresholds[
        (thresholds["agent_class"].astype(str) == str(agent_class))
        & (thresholds["threshold"].astype(str) == str(threshold))
    ].copy()
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    return frame.dropna(subset=["value"]).reset_index(drop=True)


def threshold_value_for_agent(thresholds: pd.DataFrame, *, uid: str, threshold: str) -> float | None:
    if thresholds.empty:
        return None
    matches = thresholds[
        (thresholds["uid"].astype(str) == str(uid))
        & (thresholds["threshold"].astype(str) == str(threshold))
    ]
    if matches.empty:
        return None
    value = pd.to_numeric(matches.iloc[0]["value"], errors="coerce")
    if pd.isna(value):
        return None
    return float(value)

def format_threshold_summary(group: pd.DataFrame, *, limit: int = 3) -> str:
    if group.empty:
        return ""
    ordered = group.sort_values("threshold", kind="stable")
    parts = [f"{short_threshold_name(row.threshold)}={row.value:.4g}" for row in ordered.head(limit).itertuples()]
    remaining = len(ordered) - limit
    if remaining > 0:
        parts.append(f"+{remaining} more")
    return "; ".join(parts)


def add_threshold_summary_to_agents(agents: pd.DataFrame, thresholds: pd.DataFrame) -> pd.DataFrame:
    frame = agents.copy()
    frame = frame.drop(columns=["threshold_count", "threshold_summary"], errors="ignore")
    if frame.empty:
        return frame
    if thresholds.empty:
        frame["threshold_count"] = 0
        frame["threshold_summary"] = ""
        return frame
    counts = thresholds.groupby("uid").size().rename("threshold_count")
    summaries = thresholds.groupby("uid").apply(format_threshold_summary).rename("threshold_summary")
    frame["uid"] = frame["uid"].astype(str)
    frame = frame.merge(counts, left_on="uid", right_index=True, how="left")
    frame = frame.merge(summaries, left_on="uid", right_index=True, how="left")
    frame["threshold_count"] = frame["threshold_count"].fillna(0).astype(int)
    frame["threshold_summary"] = frame["threshold_summary"].fillna("")
    return frame


def compact_agent_table_with_thresholds(service: InitializationService, agents: pd.DataFrame, thresholds: pd.DataFrame) -> pd.DataFrame:
    agents_with_thresholds = add_threshold_summary_to_agents(agents, thresholds)
    table = service.compact_agent_table(agents_with_thresholds)
    table = table.drop(columns=["initial_state"], errors="ignore")
    preferred_columns = ["uid", "display_label", "agent_class", "compartment", "rank", "owner_uid", "position_x", "position_y", "threshold_count", "threshold_summary"]
    existing_columns = [column for column in preferred_columns if column in table.columns]
    remaining_columns = [column for column in table.columns if column not in existing_columns]
    return table[[*existing_columns, *remaining_columns]]

def build_threshold_density_figure(*, values: np.ndarray, selected_value: float | None, selected_uid: str, agent_class: str, threshold: str) -> go.Figure:
    finite_values = values[np.isfinite(values)]
    figure = go.Figure()
    if finite_values.size == 0:
        return figure
    unique_values = np.unique(finite_values)
    minimum = float(finite_values.min())
    maximum = float(finite_values.max())
    density_created = False
    selected_density = 0.0
    if gaussian_kde is not None and finite_values.size >= 2 and unique_values.size >= 2 and not np.isclose(minimum, maximum):
        value_range = maximum - minimum
        padding = max(value_range * 0.15, abs(maximum) * 0.01, 1e-9)
        x_values = np.linspace(minimum - padding, maximum + padding, 400)
        try:
            density = gaussian_kde(finite_values)
            y_values = density(x_values)
        except (ValueError, np.linalg.LinAlgError):
            density_created = False
        else:
            density_created = True
            figure.add_trace(go.Scatter(x=x_values, y=y_values, mode="lines", fill="tozeroy", name="Estimated density", hovertemplate="Threshold: %{x:.6g}<br>Density: %{y:.6g}<extra></extra>"))
            if selected_value is not None:
                selected_density = float(density([selected_value])[0])
    figure.add_trace(go.Scatter(x=finite_values, y=np.zeros(finite_values.size), mode="markers", name="Agents", marker={"size": 7, "symbol": "line-ns-open"}, hovertemplate="Threshold: %{x:.6g}<extra></extra>"))
    if selected_value is not None:
        figure.add_trace(
            go.Scatter(
                x=[selected_value],
                y=[selected_density if density_created else 0.0],
                mode="markers",
                name="Selected agent",
                marker={"size": 14, "symbol": "diamond"},
                customdata=[selected_uid],
                hovertemplate="Selected agent: %{customdata}<br>Threshold: %{x:.6g}<extra></extra>"))
        figure.add_vline(
            x=selected_value,
            line_width=2,
            line_dash="dash",
            annotation_text=f"{selected_uid}: {selected_value:.6g}",
            annotation_position="top")
    if not density_created:
        if gaussian_kde is None:
            note = "Install scipy to show a continuous KDE. Showing sampled values only."
        elif unique_values.size == 1:
            note = "All agents share the same threshold value."
        else:
            note = "Not enough variability to estimate a continuous density."
        figure.add_annotation(text=note, xref="paper", yref="paper", x=0.5, y=0.92, showarrow=False)
    figure.update_layout(
        title=f"{short_threshold_name(threshold)} distribution — {agent_class}",
        xaxis_title="Threshold value", yaxis_title="Estimated density",
        height=500, hovermode="closest", legend_title_text="",
        margin={"l": 40, "r": 30, "t": 70, "b": 40}
    )
    figure.update_xaxes(showgrid=True, zeroline=False)
    figure.update_yaxes(showgrid=True, zeroline=False, rangemode="tozero")
    return figure


def render_threshold_distribution(*, thresholds: pd.DataFrame, selected_agent: dict[str, Any]) -> None:
    st.markdown("#### Threshold distribution")
    if thresholds.empty:
        st.warning("No numeric per-agent thresholds were found inside the initialization configurations.")
        st.caption("Expected numeric config keys ending with `threshold`, for example `compromised_threshold` or `oxidative_stress_high_threshold`.")
        return
    uid = str(selected_agent.get("uid", ""))
    agent_class = str(selected_agent.get("agent_class", ""))
    available = available_thresholds_for_class(thresholds, agent_class=agent_class)
    if not available:
        st.info(f"No threshold values are available for `{agent_class}` agents.")
        return
    selected_threshold = st.selectbox("Threshold", options=available, format_func=short_threshold_name, key=f"threshold-distribution-{agent_class}")
    distribution = threshold_distribution_for_class(thresholds,agent_class=agent_class,threshold=selected_threshold)
    if distribution.empty:
        st.info("No values are available for this threshold.")
        return
    selected_value = threshold_value_for_agent(thresholds,uid=uid,threshold=selected_threshold)
    values = distribution["value"].to_numpy(dtype=float)
    st.caption(f"Distribution over all `{agent_class}` agents. Selected agent: `{uid}`.")
    figure = build_threshold_density_figure(values=values,selected_value=selected_value,selected_uid=uid,agent_class=agent_class,threshold=selected_threshold)
    st.plotly_chart(figure,width="stretch",key=f"threshold-density-{agent_class}-{selected_threshold}-{uid}")
    statistics = distribution["value"].describe()
    metric_columns = st.columns(6)
    metric_columns[0].metric("Agents", len(distribution))
    metric_columns[1].metric("Mean", f"{statistics['mean']:.5g}")
    metric_columns[2].metric("Median", f"{distribution['value'].median():.5g}")
    metric_columns[3].metric("Std",f"{statistics['std']:.5g}" if not pd.isna(statistics["std"]) else "—")
    metric_columns[4].metric("Minimum", f"{statistics['min']:.5g}")
    metric_columns[5].metric("Maximum", f"{statistics['max']:.5g}")
    if selected_value is None:
        st.warning("The selected agent does not expose this threshold.")
    else:
        percentile = distribution["value"].le(selected_value).mean() * 100
        st.info(f"Selected agent value: `{selected_value:.6g}` — approximately percentile `{percentile:.1f}` within `{agent_class}`.")
    with st.expander("Threshold values by agent",expanded=False):
        table_columns = [
            column
            for column in ("uid","rank","owner_uid","threshold","value")
            if column in distribution.columns]
        display_table = distribution[table_columns].copy()
        display_table["selected"] = display_table["uid"].astype(str) == uid
        st.dataframe(display_table,width="stretch",hide_index=True,
            column_config={
                "threshold": st.column_config.TextColumn("Threshold",width="large"),
                "value": st.column_config.NumberColumn("Value",format="%.6g"),
                "selected": st.column_config.CheckboxColumn("Selected agent")})

def render_agent_detail(*, service: InitializationService, agents: pd.DataFrame, manifest: dict[str, Any], thresholds: pd.DataFrame,agent: dict[str, Any]) -> None:
    uid = str(agent.get("uid", ""))
    title = display_scalar(agent.get("display_label"), fallback="") or f"{agent.get('agent_class', 'Agent')} {uid}"
    st.subheader(str(title))
    agent_thresholds = thresholds[thresholds["uid"].astype(str) == uid] if not thresholds.empty else pd.DataFrame()
    identity_columns = st.columns(5)
    identity_columns[0].metric("Class", display_scalar(agent.get("agent_class")))
    identity_columns[1].metric("Initial state", display_scalar(agent.get("initial_state")))
    identity_columns[2].metric("Rank", display_scalar(agent.get("rank")))
    identity_columns[3].metric("Compartment",display_scalar(agent.get("compartment"), fallback="None"))
    identity_columns[4].metric("Thresholds", len(agent_thresholds))
    col_identity, col_relationships = st.columns(2)
    with col_identity:
        st.markdown("#### Identity")
        identity_table = pd.DataFrame(
            [{"field": "UID", "value": uid},
                {"field": "Local ID","value": agent.get("local_id")},
                {"field": "Type ID","value": agent.get("type_id")},
                {"field": "Display label","value": agent.get("display_label")},
                {"field": "Display group","value": agent.get("display_group")},
                {"field": "Visual level","value": agent.get("visual_level")}])
        st.dataframe(identity_table,width="stretch",hide_index=True)
    with col_relationships:
        st.markdown("#### Relationships and position")
        relationship_table = pd.DataFrame(
            [{"field": "Owner UID","value": agent.get("owner_uid")},
                {"field": "Owner label","value": agent.get("owner_label")},
                {"field": "Target UID","value": agent.get("target_uid")},
                {"field": "Target class","value": agent.get("target_class")},
                {"field": "Position X","value": agent.get("position_x")},
                {"field": "Position Y","value": agent.get("position_y")},
                {"field": "Aggregate ID","value": agent.get("aggregate_id")}])
        st.dataframe(relationship_table, width="stretch", hide_index=True)
    owner = service.owner_agent(agents, agent)
    if owner is not None:
        with st.expander("Owner agent",expanded=False):
            owner_columns = [column for column in ("uid", "display_label", "agent_class", "rank") if column in owner]
            st.dataframe(
                pd.DataFrame([{column: owner.get(column)
                            for column in owner_columns}]),
                width="stretch",hide_index=True)
    children = service.child_agents(agents, uid)
    if not children.empty:
        with st.expander(f"Contained agents ({len(children)})",expanded=False):
            child_table = compact_agent_table_with_thresholds(service, children, thresholds)
            st.dataframe(child_table,width="stretch",hide_index=True)
    neuron_manifest = service.neuron_manifest_entry(manifest, uid)
    if neuron_manifest is not None:
        with st.expander("Neuron initialization hierarchy",expanded=False):
            internal_agents = neuron_manifest.get("internal_agents", {})
            if isinstance(internal_agents, dict):
                hierarchy_rows = [
                    {"agent_class": child_class,"count": len(child_uids) if isinstance(child_uids, list) else 0}
                    for child_class, child_uids
                    in internal_agents.items()]
                if hierarchy_rows:
                    st.dataframe(pd.DataFrame(hierarchy_rows), width="stretch", hide_index=True)
            display_mapping(neuron_manifest)
    if not agent_thresholds.empty:
        with st.expander("Agent config thresholds",expanded=False):
            st.dataframe(
                agent_thresholds[["threshold","value"]],
                width="stretch",
                hide_index=True,
                column_config={
                    "threshold": st.column_config.TextColumn("Threshold",width="large"),
                    "value": st.column_config.NumberColumn("Value",format="%.6g")})
    tab_config, tab_scalars, tab_buffers, tab_raw = st.tabs(["Configuration", "Initial scalars", "Initial buffers","Raw record"])
    with tab_config:
        display_mapping(agent.get("config"))
    with tab_scalars:
        st.markdown("#### Environment scalars")
        display_mapping(agent.get("initial_scalars"))
        st.markdown("#### Internal scalars")
        display_mapping(agent.get("initial_internal_scalars"))
    with tab_buffers:
        display_mapping(agent.get("initial_buffers"))
    with tab_raw:
        raw_fields = {key: value for key, value in agent.items() if not str(key).startswith("_")}
        display_mapping(raw_fields)

@st.cache_data(show_spinner=False)
def load_initialization_data(agents_path: str, agents_mtime: float, manifest_path: str, manifest_mtime: float | None, summary_path: str, summary_mtime: float | None) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    del agents_mtime
    del manifest_mtime
    del summary_mtime
    loader = InitializationService(agents_path=agents_path,manifest_path=manifest_path, summary_path=summary_path)
    return loader.load_agents(),loader.load_manifest(),loader.load_summary()

st.title("Agent initialization")
st.caption("Inspect every agent created at simulation startup, including configuration, ownership, config thresholds and initial values.")
service = InitializationService()
if not service.has_initialization_data():
    st.info("No simulation to explore now.")
    st.stop()
try:
    agents, manifest, summary = load_initialization_data(
        str(service.agents_path),
        service.agents_path.stat().st_mtime,
        str(service.manifest_path),
        (service.manifest_path.stat().st_mtime
        if service.manifest_path.exists()
        else None),
        str(service.summary_path),
        (service.summary_path.stat().st_mtime
        if service.summary_path.exists()
        else None))
except InitializationDataError as exc:
    st.error("The initialization log is invalid.")
    st.code(str(exc), language="text")
    st.stop()
except OSError as exc:
    st.error("Initialization data could not be read.")
    st.exception(exc)
    st.stop()
if agents.empty:
    st.info("No simulation to explore now.")
    st.stop()
thresholds = threshold_frame_from_agents(agents)
if thresholds.empty:
    st.warning("No numeric config thresholds were found. The page expects numeric keys in `config` whose name ends with `threshold`, such as `compromised_threshold`.")
with st.expander("Initialization overview", expanded=True):
    render_summary(service, summary)
st.divider()
st.subheader("Agent explorer")
search = st.text_input("Search",placeholder=("Search by UID, class, owner, target or display label..."))
filter_row = st.columns(3)
with filter_row[0]:
    available_classes = service.available_classes(agents)
    selected_classes = st.multiselect("Agent classes",options=available_classes,default=available_classes)
with filter_row[1]:
    available_ranks = service.available_ranks(agents)
    selected_ranks = st.multiselect("MPI ranks",options=available_ranks,default=available_ranks)
with filter_row[2]:
    available_compartments = service.available_compartments(agents)
    selected_compartments = st.multiselect("Compartments",options=available_compartments,default=available_compartments)
available_owners = service.available_owner_uids(agents)
selected_owner = st.selectbox("Owner neuron",options=["All",*available_owners])
filtered_agents = service.filter_agents(agents,search=search,agent_classes=selected_classes,compartments=selected_compartments,ranks=selected_ranks,owner_uid=(None if selected_owner == "All" else selected_owner))
if filtered_agents.empty:
    st.warning("No agents match the selected filters.")
    st.stop()
summary_columns = st.columns(4)
summary_columns[0].metric("Matching agents",len(filtered_agents))
summary_columns[1].metric("Classes",filtered_agents["agent_class"].nunique())
summary_columns[2].metric("Owner neurons",(filtered_agents["owner_uid"] .dropna() .nunique() if "owner_uid" in filtered_agents.columns else 0))
matching_threshold_count = 0
if not thresholds.empty:
    matching_uids = set(filtered_agents["uid"].astype(str).tolist())
    matching_threshold_count = len(thresholds[thresholds["uid"].astype(str).isin(matching_uids)])
summary_columns[3].metric("Threshold values",matching_threshold_count)
table = compact_agent_table_with_thresholds(service, filtered_agents, thresholds)
event = st.dataframe(
    table,
    width="stretch",
    hide_index=True,
    selection_mode="single-row",
    on_select="rerun",
    key="initialization_agent_table",
    column_config={
        "uid": st.column_config.TextColumn("UID",width="medium"),
        "display_label": st.column_config.TextColumn("Label",width="medium"),
        "agent_class": st.column_config.TextColumn("Class",width="medium"),
        "rank": st.column_config.NumberColumn("Rank",format="%d",width="small"),
        "position_x": st.column_config.NumberColumn("X",format="%d",width="small"),
        "position_y": st.column_config.NumberColumn("Y",format="%d",width="small"),
        "threshold_count": st.column_config.NumberColumn("Thresholds",format="%d",width="small"),
        "threshold_summary": st.column_config.TextColumn("Threshold summary",width="large")})
selected_rows = event.selection.rows
if selected_rows:
    selected_index = selected_rows[0]
else:
    selected_index = 0
selected_uid = str(table.iloc[selected_index]["uid"])
selected_agent = service.agent_by_uid(agents, selected_uid)
if selected_agent is None:
    st.warning("The selected agent could not be loaded.")
    st.stop()
st.divider()
render_agent_detail(service=service,agents=agents,manifest=manifest,thresholds=thresholds,agent=selected_agent)
render_threshold_distribution(thresholds=thresholds,selected_agent=selected_agent)
