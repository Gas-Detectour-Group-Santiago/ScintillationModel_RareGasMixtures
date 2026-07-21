from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import streamlit as st

from app.context import PROJECT_ROOT
from app.pages import figure_recipes, outputs, plot_style, run_pipeline
from scintillation.gui.catalog import discover_primary_datasets, discover_secondary_datasets
from scintillation.gui.style_config import active_style_name

st.set_page_config(
    page_title="ScintillationModel",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1500px;}
      [data-testid="stMetric"] {background: rgba(127,127,127,0.06); border: 1px solid rgba(127,127,127,0.16); padding: 0.75rem 0.9rem; border-radius: 0.7rem;}
      [data-testid="stSidebar"] {border-right: 1px solid rgba(127,127,127,0.16);}
      div[data-testid="stCode"] pre {font-size: 0.78rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

pages = {
    "Control": [
        st.Page(run_pipeline.page, title="Run pipeline", icon="▶️", url_path="run", default=True),
    ],
    "Configuration": [
        st.Page(figure_recipes.page, title="Figure recipes", icon="📊", url_path="recipes"),
        st.Page(outputs.page, title="Outputs", icon="📁", url_path="outputs"),
    ],
    "Appearance": [
        st.Page(plot_style.page, title="Plot style", icon="🎨", url_path="style"),
    ],
}

with st.sidebar:
    st.markdown("## ScintillationModel")
    st.caption("Control panel over the reproducible command-line workflows.")
    st.divider()
    st.caption(f"Project: `{PROJECT_ROOT.name}`")
    st.caption(f"Active style: `{active_style_name(PROJECT_ROOT)}`")
    st.caption(f"Primary datasets: **{len(discover_primary_datasets(PROJECT_ROOT))}**")
    st.caption(f"Secondary datasets: **{len(discover_secondary_datasets(PROJECT_ROOT))}**")
    st.divider()
    st.caption("The GUI writes recipes and calls `run_all.sh` or `run_products.sh`; it does not duplicate the physics.")

navigation = st.navigation(pages, position="sidebar", expanded=True)
navigation.run()
