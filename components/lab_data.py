"""Cached dataset and model loading for the Streamlit process."""

import streamlit as st

from src.data_loader import DatasetError, load_credit_dataset, validate_dataset
from src.model_loader import load_lab


@st.cache_data(show_spinner="Loading the UCI credit dataset…")
def cached_dataset():
    return load_credit_dataset()


@st.cache_resource(show_spinner="Preparing comparison models…")
def cached_lab():
    df = cached_dataset()
    return load_lab(df)


def dataset_or_stop():
    try:
        df = cached_dataset()
        report = validate_dataset(df)
        return df, report
    except DatasetError as exc:
        st.error(f"The dataset could not be loaded. {exc}", icon=":material/error:")
        st.stop()
    except Exception:
        st.error(
            "The dataset could not be loaded. Check your network connection and the data/ folder.",
            icon=":material/error:",
        )
        st.stop()


def lab_or_stop():
    try:
        return cached_lab()
    except Exception:
        st.error(
            "Models could not be prepared. Confirm credit_model.joblib is present and scikit-learn is installed.",
            icon=":material/error:",
        )
        st.stop()
