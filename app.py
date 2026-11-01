from pathlib import Path

import streamlit as st

from components.state import init_session
from components.styles import apply_lab_chrome

st.set_page_config(
    page_title="CREDIT LAB",
    page_icon=":material/analytics:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

init_session()
apply_lab_chrome()

_PAGES = Path(__file__).resolve().parent / "app_pages"

page = st.navigation(
    [
        st.Page(_PAGES / "command_center.py", title="Command Center", icon=":material/hub:"),
        st.Page(_PAGES / "model_arena.py", title="Model Arena", icon=":material/swords:"),
        st.Page(_PAGES / "data_observatory.py", title="Data Observatory", icon=":material/query_stats:"),
        st.Page(_PAGES / "credit_simulator.py", title="Credit Simulator", icon=":material/tune:"),
    ],
    position="top",
)

page.run()
