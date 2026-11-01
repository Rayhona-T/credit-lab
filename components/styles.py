from pathlib import Path

import streamlit as st

from src.constants import PROJECT_ROOT

CSS_PATH = PROJECT_ROOT / "assets" / "styles.css"
LOGO_PATH = PROJECT_ROOT / "assets" / "logo.svg"


def apply_lab_chrome() -> None:
    if CSS_PATH.exists():
        st.html(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>")
    if LOGO_PATH.exists():
        st.logo(str(LOGO_PATH))


def lab_footer() -> None:
    st.html(
        """
        <div class="lab-footer">
          CREDIT LAB is an educational experiment on the UCI Default of Credit Card Clients dataset.
          Model outputs are estimates from historical patterns, not credit decisions, character judgments,
          or guarantees about any person.
        </div>
        """
    )
