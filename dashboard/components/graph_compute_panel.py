from __future__ import annotations
import streamlit as st
from dashboard.services.graph_compute_service import GraphComputeResult, GraphComputeService

def render_graph_compute_panel(level: str,*,cache_clear_callbacks: list[object] | None = None,session_keys_to_clear: list[str] | None = None) -> GraphComputeResult | None:
    level = level.upper()
    service = GraphComputeService()
    result_key = f"{level.lower()}_compute_result"
    with st.container(border=True):
        columns = st.columns([1, 3], vertical_alignment="center")
        with columns[0]:
            compute_clicked = st.button("Compute", type="primary", use_container_width=True, key=f"{level.lower()}_compute")
        with columns[1]:
            st.caption(f"Run `{command_label(service.command_for(level))}` and refresh the generated {level} artifacts.")
    if compute_clicked:
        with st.spinner(f"Computing {level} graph...", show_time=True):
            result = service.compute(level)
        st.session_state[result_key] = result
        if result.success:
            clear_dashboard_caches(cache_clear_callbacks)
            clear_session_keys(session_keys_to_clear)
    result = st.session_state.get(result_key)
    if result is None:
        return None
    if result.success:
        st.success(f"{level} graph computed.")
    else:
        st.error(f"{level} graph computation failed.")

    with st.expander(f"{level} compute report", expanded=True):
        if result.report_text:
            st.markdown(result.report_text)
        elif result.stdout:
            st.code(result.stdout, language="text", wrap_lines=True)
        else:
            st.info("No report was produced.")
        if result.stderr:
            st.code(result.stderr, language="text", wrap_lines=True)
    return result


def clear_dashboard_caches(callbacks: list[object] | None) -> None:
    for callback in callbacks or []:
        clear = getattr(callback, "clear", None)
        if callable(clear):
            clear()

def clear_session_keys(keys: list[str] | None) -> None:
    for key in keys or []:
        st.session_state.pop(key, None)


def command_label(command: list[str]) -> str:
    return " ".join(str(part) for part in command[1:])
