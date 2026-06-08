from __future__ import annotations
from pathlib import Path
import networkx as nx
import streamlit as st
from dashboard.services.graph_png_export_service import GraphPngExportService

# TODO deprecated?

def render_graph_png_export_button(graph: nx.DiGraph, *, level: str, name_hint: str) -> Path | None:
    button_key = f"{level.lower()}_{name_hint}_save_png"
    if not st.button("Save PNG", use_container_width=True, key=button_key):
        return None
    try:
        with st.spinner("Saving graph view as PNG...", show_time=True):
            output_path = GraphPngExportService().save_png(graph, level=level, name_hint=name_hint)
    except Exception as exc:
        st.error("Unable to save the current graph view as PNG.")
        st.exception(exc)
        return None
    st.success(f"PNG saved: `{output_path}`")
    return output_path
