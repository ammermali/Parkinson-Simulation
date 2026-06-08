from __future__ import annotations
from pathlib import Path
from typing import Any
import pandas as pd
import streamlit as st
import yaml
from dashboard.services.parameter_service import ParameterService

def display_yaml(data: Any) -> None:
    yaml_text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    st.code(yaml_text, language="yaml", line_numbers=False, wrap_lines=True)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMETER_GUIDE_PATH = PROJECT_ROOT / "specifics" / "parameters_guide.md"

def render_parameter_guide(path: Path) -> None:
    if not path.exists():
        st.info(f"Parameter guide not found: `{path}`")
        return
    try:
        markdown_text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        st.warning(f"Unable to read the parameter guide: {exc}")
        return
    with st.expander("Parameter guide", expanded=False):
        st.markdown(markdown_text)

st.title("Simulation parameters")
render_parameter_guide(PARAMETER_GUIDE_PATH)
service = ParameterService()
param_files = service.available_files()

if not param_files:
    st.warning(f"No YAML parameter files found in `{service.param_dir}`.")
    st.stop()

selected_path = st.selectbox("Parameter file",options=param_files,format_func=lambda path: path.name)
st.caption(f"Editing `{selected_path}`")
try:
    original_document = service.load_document(selected_path)
    rows = service.editable_rows(selected_path)
except Exception as exc:
    st.error("Unable to load the selected parameter file.")
    st.exception(exc)
    st.stop()
editor_frame = pd.DataFrame(rows, columns=["key", "type", "value"])
with st.form("parameter_editor"):
    edited_frame = st.data_editor(editor_frame,width="stretch",hide_index=True,disabled=["key", "type"],
        column_config={
            "key": st.column_config.TextColumn("Parameter",help="Dot-separated YAML path.",width="large"),
            "type": st.column_config.TextColumn("Type",width="small"),
            "value": st.column_config.TextColumn("Value",help="Values use YAML syntax: true, 1.5, [1, 2], {name: value}, null.",width="large")},
        num_rows="fixed",key=f"parameter_editor::{selected_path}")
    col_save, col_preview = st.columns(2)
    save_clicked = col_save.form_submit_button("Save parameters",type="primary",width="stretch")
    preview_clicked = col_preview.form_submit_button("Preview changes",width="stretch")
edited_rows = edited_frame.to_dict(orient="records")
if preview_clicked:
    updates, errors = service.parse_and_validate(selected_path, edited_rows)
    if not errors:
        changes = service.changed_values(selected_path, updates)
        if changes:
            st.json(changes)
        else:
            st.info("No parameters have changed.")
    if errors:
        st.error("Some values are invalid.")
        error_frame = pd.DataFrame(
            [{"parameter": error.key,"error": error.message}
            for error in errors])
        st.dataframe(error_frame, width="stretch", hide_index=True)
    else:
        changed_values = {}
        for key, updated_value in updates.items():
            current_value = service.load_document(selected_path)
            from src.configuration.param_editing import read_path
            original_value = read_path(current_value, key)
            if updated_value != original_value:
                changed_values[key] = {"before": original_value,"after": updated_value}
        if changed_values:
            st.subheader("Pending changes")
            st.json(changed_values)
        else:
            st.info("No parameters have changed.")
if save_clicked:
    try:
        updated_document = service.save_rows(selected_path, edited_rows)
    except ValueError as exc:
        st.error("The parameters were not saved.")
        st.code(str(exc), language="text")
    except Exception as exc:
        st.error("Unable to write the parameter file.")
        st.exception(exc)
    else:
        st.success(f"Saved `{selected_path.name}`.")
        with st.expander("Updated YAML document"):
            display_yaml(updated_document)
        st.rerun()

with st.expander("Raw YAML", expanded=False):
    display_yaml(original_document)
