from __future__ import annotations
import networkx as nx
import pandas as pd
import streamlit as st
from dashboard.components.graph_compute_panel import render_graph_compute_panel
from dashboard.components.graph_png_export_panel import render_graph_png_export_button
from dashboard.components.g0_viewer import render_g0_graph
from dashboard.services.g0_view_service import G0Entity, G0ViewError, G0ViewResult, G0ViewService

MAX_VIEW_NODES = 2_000

@st.cache_resource(show_spinner="Loading the generated G0 graph...")
def load_g0_graph(graph_path: str, modified_time: float) -> nx.DiGraph:
    del modified_time
    loader = G0ViewService(graph_path)
    return loader.load_graph()


@st.cache_data(show_spinner=False)
def extract_graph_metadata(graph_path: str, modified_time: float) -> tuple[list[int], list[G0Entity]]:
    graph = load_g0_graph(graph_path, modified_time)
    service = G0ViewService(graph_path)
    return (service.available_ticks(graph), service.available_entities(graph))


def render_view_summary(result: G0ViewResult) -> None:
    columns = st.columns(6)
    columns[0].metric("Entity", result.entity.label)
    columns[1].metric("Tick range",f"{result.start_tick}–{result.end_tick}")
    columns[2].metric("Ticks", result.tick_count)
    columns[3].metric("Radius", result.radius)
    columns[4].metric("Nodes", result.node_count)
    columns[5].metric("Edges", result.edge_count)
    st.caption(f"Direction: `{result.direction}` · Temporal continuity: `{'included' if result.include_continuity else 'excluded'}`")

def render_graph_tables(*, service: G0ViewService, graph: nx.DiGraph) -> None:
    tab_nodes, tab_edges, tab_metadata = st.tabs(["Nodes", "Edges", "View metadata"])
    with tab_nodes:
        node_rows = service.nodes_frame_rows(graph)
        if not node_rows:
            st.info("The view contains no nodes.")
        else:
            st.dataframe(pd.DataFrame(node_rows),width="stretch",hide_index=True)
    with tab_edges:
        edge_rows = service.edges_frame_rows(graph)
        if not edge_rows:
            st.info("The view contains no edges.")
        else:
            st.dataframe(pd.DataFrame(edge_rows), width="stretch", hide_index=True)
    with tab_metadata:
        metadata = [{"property": key,"value": value,}
            for key, value in graph.graph.items()]
        st.dataframe(pd.DataFrame(metadata), width="stretch",hide_index=True)


st.title("G0 explorer")
st.caption("Generate a bounded local view from the previously generated G0 graph.")
render_graph_compute_panel("G0", cache_clear_callbacks=[load_g0_graph,extract_graph_metadata,],session_keys_to_clear=["g0_view_result"])
service = G0ViewService()
if not service.has_graph():
    st.info("G0 has not been generated yet.")
    st.caption(f"Expected graph: `{service.graph_path}`.")
    st.stop()
try:
    graph_modified_time = service.graph_path.stat().st_mtime
    graph = load_g0_graph(str(service.graph_path), graph_modified_time)
    ticks, entities = extract_graph_metadata(str(service.graph_path), graph_modified_time)
except Exception as exc:
    st.error("Unable to load the generated G0 graph.")
    st.exception(exc)
    st.stop()

if not ticks:
    st.warning("The generated G0 graph does not contain valid tick attributes.")
    st.stop()

if not entities:
    st.warning("The generated G0 graph does not contain selectable entity keys.")
    st.stop()

graph_summary = st.columns(3)
graph_summary[0].metric("G0 nodes", graph.number_of_nodes())
graph_summary[1].metric("G0 edges", graph.number_of_edges())
graph_summary[2].metric("Selectable entities", len(entities))
st.divider()
st.subheader("View configuration")
categories = service.available_entity_categories(entities)
filter_columns = st.columns(2)
with filter_columns[0]:
    selected_category = st.selectbox("Entity category",options=["All", *categories])
with filter_columns[1]:
    entity_search = st.text_input("Search entity",placeholder=("Search by UID, agent type, field or entity key..."))
filtered_entities = service.filter_entities(entities, category=selected_category, search=entity_search)
if not filtered_entities:
    st.warning("No entities match the current filters.")
    st.stop()
selected_entity = st.selectbox("Central entity", options=filtered_entities, format_func=lambda entity: entity.label)
minimum_tick = min(ticks)
maximum_tick = max(ticks)
default_end_tick = min(minimum_tick + 10, maximum_tick)
selected_tick_range = st.slider("Tick range", min_value=minimum_tick, max_value=maximum_tick, value=(minimum_tick, default_end_tick), step=1)
option_columns = st.columns(3)
with option_columns[0]:
    radius = st.slider("Causal radius", min_value=0, max_value=4, value=1, step=1,
        help=("Number of incoming or outgoing graph hops from the selected temporal entity."))

with option_columns[1]:
    direction = st.selectbox("Causal direction",options=["both","incoming","outgoing"],
        format_func=lambda value: {"both": "Incoming and outgoing",
            "incoming": "Incoming only",
            "outgoing": "Outgoing only"}[value])

with option_columns[2]:
    include_continuity = st.checkbox("Include temporal continuity",value=True,
        help=("Include edges generated by temporal_identity between consecutive ticks."))

generate_clicked = st.button("Generate G0 view", type="primary",width="stretch")

if generate_clicked:
    try:
        with st.spinner("Generating the local G0 view...", show_time=True):
            result = service.build_view(graph, entity=selected_entity,
                start_tick=selected_tick_range[0], end_tick=selected_tick_range[1],
                radius=radius, direction=direction, include_continuity=include_continuity, max_nodes=MAX_VIEW_NODES)
    except G0ViewError as exc:
        st.warning(str(exc))
    except Exception as exc:
        st.error("Unable to generate the requested G0 view.")
        st.exception(exc)
    else:
        st.session_state["g0_view_result"] = result
result = st.session_state.get("g0_view_result")
if result is None:
    st.info("Select a central entity and a tick range, then press **Generate G0 view**.")
    st.stop()
st.divider()
st.subheader("Generated view")
render_view_summary(result)
if result.node_count > 500:
    st.warning("This view is relatively large. PyVis may take some time to stabilize. Reduce the tick range or causal radius for a clearer representation.")

if result.edge_count == 0:
    st.info("The selected view contains nodes but no edges. Try increasing the causal radius or changing direction.")
render_graph_png_export_button(result.graph, level="G0", name_hint=result.entity.key)
render_g0_graph(result.graph, central_entity_key=result.entity.key)
with st.expander("Inspect view data", expanded=False):
    render_graph_tables(service=service,graph=result.graph)
