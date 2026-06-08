from __future__ import annotations
import networkx as nx
import pandas as pd
import streamlit as st
from dashboard.components.graph_compute_panel import render_graph_compute_panel
from dashboard.components.g0_viewer import render_graph
from dashboard.components.graph_png_export_panel import render_graph_png_export_button
from dashboard.components.graph_tables import render_graph_summary, render_graph_tables
from dashboard.services.g0_view_service import G0Entity, G0ViewError
from dashboard.services.g3_annotation_service import G3AnnotationService, G3NodeAnnotation
from dashboard.services.graph_level_view_service import GraphLevelViewResult, GraphLevelViewService

MAX_ENTITY_VIEW_NODES = 2_000
LARGE_FULL_GRAPH_NODES = 1_500

@st.cache_resource(show_spinner="Loading graph...")
def load_level_graph(level: str, graph_path: str, modified_time: float) -> nx.DiGraph:
    del modified_time
    return GraphLevelViewService(level, graph_path).load_graph()

@st.cache_data(show_spinner=False)
def extract_level_entities(level: str, graph_path: str, modified_time: float) -> list[G0Entity]:
    graph = load_level_graph(level, graph_path, modified_time)
    return GraphLevelViewService(level, graph_path).available_entities(graph)

def render_g1_page() -> None:
    level = "G1"
    service = GraphLevelViewService(level)
    st.title("G1 explorer")
    render_graph_compute_panel(level,cache_clear_callbacks=[load_level_graph,extract_level_entities],session_keys_to_clear=["g1_view_result"])
    graph = load_page_graph(service)
    entities = extract_level_entities(level, str(service.graph_path), service.graph_path.stat().st_mtime)
    if not entities:
        st.warning("The generated G1 graph does not contain selectable entity keys.")
        st.stop()
    render_graph_summary(graph, level=level, extra_metric=("Selectable entities", len(entities)))
    st.divider()
    st.subheader("View configuration")
    selected_entity = render_entity_selector(service, entities, level_key="g1")
    radius, direction, include_continuity = render_entity_view_options(level_key="g1")
    if st.button("Generate G1 view", type="primary", width="stretch"):
        try:
            with st.spinner("Generating the local G1 view...", show_time=True):
                result = service.build_entity_view(graph,entity=selected_entity,radius=radius,direction=direction,include_continuity=include_continuity,max_nodes=MAX_ENTITY_VIEW_NODES)
        except G0ViewError as exc:
            st.warning(str(exc))
        except Exception as exc:
            st.error("Unable to generate the requested G1 view.")
            st.exception(exc)
        else:
            st.session_state["g1_view_result"] = result
    result = st.session_state.get("g1_view_result")
    if result is None:
        st.info("Select a central entity, then press **Generate G1 view**.")
        st.stop()

    st.divider()
    st.subheader("Generated view")
    render_entity_view_summary(result)

    if result.node_count > 500:
        st.warning("This view is relatively large. PyVis may take some time to stabilize.")
    if result.edge_count == 0:
        st.info("The selected view contains nodes but no edges. Try increasing the causal radius or changing direction.")

    render_graph_png_export_button(result.graph,level=level,name_hint=result.entity.key if result.entity else "view")
    render_graph(result.graph, central_entity_key=result.entity.key if result.entity else None)
    with st.expander("Inspect view data", expanded=False):
        render_graph_tables(service=service, graph=result.graph)


def render_full_level_page(level: str, *, title: str) -> None:
    service = GraphLevelViewService(level)
    st.title(title)

    render_graph_compute_panel(level,cache_clear_callbacks=[load_level_graph,extract_level_entities],session_keys_to_clear=[f"{level.lower()}_view_result"])
    graph = load_page_graph(service)
    if level.upper() == "G3":
        G3AnnotationService().apply_annotations(graph)
    result = service.build_full_view(graph)
    render_graph_summary(result.graph, level=level)
    if result.node_count > LARGE_FULL_GRAPH_NODES:
        st.warning("This graph is large. PyVis may take some time to stabilize.")
    if level.upper() == "G3":
        render_g3_annotation_editor(result.graph)
    render_graph_png_export_button(result.graph,level=level,name_hint="full")
    render_graph(result.graph, height=820)
    with st.expander("Inspect graph data", expanded=False):
        render_graph_tables(service=service, graph=result.graph, metadata_label="Graph metadata")

def load_page_graph(service: GraphLevelViewService) -> nx.DiGraph:
    if not service.has_graph():
        st.info(f"{service.level} has not been generated yet.")
        st.caption(f"Expected graph: `{service.graph_path}`.")
        st.stop()
    try:
        return load_level_graph(service.level, str(service.graph_path), service.graph_path.stat().st_mtime)
    except Exception as exc:
        st.error(f"Unable to load the generated {service.level} graph.")
        st.exception(exc)
        st.stop()


def render_entity_selector(service: GraphLevelViewService, entities: list[G0Entity], *, level_key: str) -> G0Entity:
    categories = service.available_entity_categories(entities)
    filter_columns = st.columns(2)
    with filter_columns[0]:
        selected_category = st.selectbox("Entity category", options=["All", *categories], key=f"{level_key}_category")
    with filter_columns[1]:
        entity_search = st.text_input("Search entity", placeholder="Search by UID, agent type, field or entity key...", key=f"{level_key}_search")
    filtered_entities = service.filter_entities(entities, category=selected_category, search=entity_search)
    if not filtered_entities:
        st.warning("No entities match the current filters.")
        st.stop()
    return st.selectbox("Central entity",options=filtered_entities,format_func=lambda entity: entity.label,key=f"{level_key}_entity")

def render_entity_view_options(*, level_key: str) -> tuple[int, str, bool]:
    option_columns = st.columns(3)
    with option_columns[0]:
        radius = st.slider("Causal radius", min_value=0, max_value=4, value=1, step=1, key=f"{level_key}_radius")
    with option_columns[1]:
        direction = st.selectbox(
            "Causal direction",options=["both", "incoming", "outgoing"],
            format_func=lambda value: {"both": "Incoming and outgoing","incoming": "Incoming only","outgoing": "Outgoing only"}[value],
            key=f"{level_key}_direction")
    with option_columns[2]:
        include_continuity = st.checkbox("Include continuity", value=True, key=f"{level_key}_continuity")
    return radius, direction, include_continuity

def render_entity_view_summary(result: GraphLevelViewResult) -> None:
    columns = st.columns(5)
    columns[0].metric("Entity", result.entity.label if result.entity else "")
    columns[1].metric("Radius", result.radius)
    columns[2].metric("Direction", result.direction)
    columns[3].metric("Nodes", result.node_count)
    columns[4].metric("Edges", result.edge_count)
    st.caption(f"Continuity: `{'included' if result.include_continuity else 'excluded'}`")

def render_g3_annotation_editor(graph: nx.DiGraph) -> None:
    annotation_service = G3AnnotationService()
    rows = annotation_service.supernode_rows(graph)
    st.divider()
    st.subheader("Manual loop semantics")
    if not rows:
        st.info("No G3 supernodes are available for annotation.")
        return
    with st.expander("Supernode contractions", expanded=True):
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    row_by_node = {str(row["node_id"]): row for row in rows}
    selected_node = st.selectbox(
        "G3 supernode",
        options=list(row_by_node),
        format_func=lambda node_id: supernode_option_label(row_by_node[node_id]),
        key="g3_annotation_node")
    selected_row = row_by_node[selected_node]
    selected_attributes = dict(graph.nodes[selected_node])
    detail_columns = st.columns([1, 1])
    with detail_columns[0]:
        biological_label = st.text_input(
            "Biological loop name",
            value=str(selected_row.get("biological_label") or ""),
            placeholder="Example: microglia inflammation feedback",
            key=f"g3_label_{selected_node}")
        semantic_note = st.text_area(
            "Semantic note",
            value=str(selected_row.get("semantic_note") or ""),
            placeholder="Describe why this contracted component should be read as a biological loop...",
            height=120,
            key=f"g3_note_{selected_node}")
        button_columns = st.columns(2)
        with button_columns[0]:
            save_clicked = st.button("Save semantic name", type="primary", width="stretch", key=f"g3_save_{selected_node}")
        with button_columns[1]:
            clear_clicked = st.button("Clear", width="stretch", key=f"g3_clear_{selected_node}")
    with detail_columns[1]:
        st.dataframe(
            pd.DataFrame(
                [{"property": key, "value": value}
                for key, value in selected_attributes.items()]),
            width="stretch",hide_index=True)
    if save_clicked:
        annotation_service.save_annotation(
            G3NodeAnnotation(node_id=selected_node,biological_label=biological_label.strip(),semantic_note=semantic_note.strip()))
        st.success("G3 supernode semantics saved.")
        st.rerun()
    if clear_clicked:
        annotation_service.save_annotation(G3NodeAnnotation(node_id=selected_node))
        st.success("G3 supernode semantics cleared.")
        st.rerun()

def supernode_option_label(row: dict[str, object]) -> str:
    biological_label = str(row.get("biological_label") or "").strip()
    if biological_label:
        return biological_label
    label = str(row.get("label") or "").strip()
    node_id = str(row.get("node_id") or "")
    pattern = str(row.get("pattern_kind") or "G3 node")
    size = row.get("component_size") or row.get("member_count") or ""
    if label and label != "mixed":
        return label
    return f"{pattern} {size} | {node_id}"
