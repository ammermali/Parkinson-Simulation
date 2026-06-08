from __future__ import annotations
import streamlit as st


st.set_page_config(
    page_title="Parkinson Simulation",
    layout="wide"
)

pages = {
    "Run": [
        st.Page(
            "dashboard/pages/parameters.py",
            title="Parameters"
        ),
        st.Page(
            "dashboard/pages/simulation.py",
            title="Simulation"
        )
    ],
    "Post-run": [
        st.Page(
            "dashboard/pages/spatial_reconstruction.py",
            title = "Reconstruction"
        ),
        st.Page(
            "dashboard/pages/initialization.py",
            title = "Initialization Overview"
        ),
        st.Page(
            "dashboard/pages/metrics.py",
            title = "Post-run Metrics"
        )
    ],
    "Graphs": [
        st.Page(
            "dashboard/pages/g0.py",
            title="Temporal Causal Graph"
        ),
        st.Page(
            "dashboard/pages/g1.py",
            title="Time-contracted Graph"
        ),
        st.Page(
            "dashboard/pages/g2.py",
            title="Agent-state Graph"
        ),
        st.Page(
            "dashboard/pages/g3.py",
            title="Topological Pattern Graph"
        )
    ],
    "Docs": [
        st.Page("dashboard/pages/docs.py", title="Specifics and Docs")
    ]
}

page = st.navigation(pages)
page.run()
