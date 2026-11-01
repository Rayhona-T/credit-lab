import streamlit as st

from src.constants import MODEL_TREE, PRIMARY_METRIC
from src.profiles import default_client


def init_session() -> None:
    st.session_state.setdefault("selected_model", MODEL_TREE)
    st.session_state.setdefault("selected_metric", PRIMARY_METRIC)
    st.session_state.setdefault("client", default_client())
    st.session_state.setdefault("baseline_prediction", None)
    st.session_state.setdefault("baseline_client", None)
    st.session_state.setdefault("modified_prediction", None)
    st.session_state.setdefault("cm_cell", "true_positive")
    st.session_state.setdefault("simulator_model", MODEL_TREE)
