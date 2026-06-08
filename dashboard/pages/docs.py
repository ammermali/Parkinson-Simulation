from __future__ import annotations
from pathlib import Path
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMETER_GUIDE_PATH = PROJECT_ROOT / "specifics" / "parameters_guide.md"

def render_markdown(name: str, path: Path) -> None:
    if not path.exists():
        st.info(f"Docs not found: {path}")
        return
    try:
        markdown_text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        st.warning(f"Unable to read the docs: {exc}")
        return
    with st.expander(name, expanded=False):
        st.markdown(markdown_text)

st.title("Simulation docs")
render_markdown("README", PROJECT_ROOT / "README.md")
render_markdown("Parameter guide", PROJECT_ROOT / "specifics" / "parameters_guide.md")
render_markdown("Multilevel", PROJECT_ROOT / "specifics" / "multilevel.md")
render_markdown("Data Pipeline", PROJECT_ROOT / "specifics" / "data_pipeline.md")
render_markdown("Agents Overview", PROJECT_ROOT / "specifics" / "agents_overview.md")