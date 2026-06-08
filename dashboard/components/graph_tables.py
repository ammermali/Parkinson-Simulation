from __future__ import annotations
from typing import Any
import networkx as nx
import pandas as pd
import streamlit as st

def render_graph_summary(graph: nx.DiGraph, *, level: str, extra_metric: tuple[str, Any] | None = None) -> None:
    columns = st.columns(3)
    columns[0].metric(f"{level} nodes", graph.number_of_nodes())
    columns[1].metric(f"{level} edges", graph.number_of_edges())
    if extra_metric is None:
        columns[2].metric("Level", graph.graph.get("level", level))
    else:
        columns[2].metric(extra_metric[0], extra_metric[1])

def render_graph_tables(*, service: Any, graph: nx.DiGraph, metadata_label: str = "View metadata") -> None:
    tab_nodes, tab_edges, tab_metadata = st.tabs(["Nodes", "Edges", metadata_label])
    with tab_nodes:
        node_rows = service.nodes_frame_rows(graph)
        if not node_rows:
            st.info("The graph contains no nodes.")
        else:
            st.dataframe(pd.DataFrame(node_rows), width="stretch", hide_index=True)
    with tab_edges:
        edge_rows = service.edges_frame_rows(graph)
        if not edge_rows:
            st.info("The graph contains no edges.")
        else:
            st.dataframe(pd.DataFrame(edge_rows), width="stretch", hide_index=True)
    with tab_metadata:
        metadata = [{"property": key, "value": value} for key, value in graph.graph.items()]
        st.dataframe(pd.DataFrame(metadata), width="stretch", hide_index=True)
