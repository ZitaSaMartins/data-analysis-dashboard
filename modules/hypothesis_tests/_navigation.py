"""Legacy page-navigation helper, kept separate to avoid a core <-> page import cycle."""

import streamlit as st


def _open_hypothesis_category(category: str, selected_test: str = None):
    st.session_state.hypothesis_category = category
    st.session_state.parametric_test = (
        selected_test if category == "Parametric Tests" else None
    )
    st.session_state.nonparametric_test = (
        selected_test if category == "Non-parametric Tests" else None
    )
    st.rerun()

