"""Legacy hypothesis-testing workflow: page dispatch and shared layout helpers."""

import pandas as pd
import streamlit as st

from .distribution import run_test_distribution
from .nonparametric import run_nonparametric_tests
from .parametric import run_parametric_tests
from .suggested_path import run_suggested_path
from .variance import run_test_variance


def run_hypothesis_tests(
    df: pd.DataFrame,
    category: str = None,
    parametric_test: str = None,
    nonparametric_test: str = None,
):
    """Run the hypothesis testing interface for the selected sub-menu category."""
    if category == "Suggested Path":
        run_suggested_path(df)
    elif category == "Test Distribution":
        run_test_distribution(df)
    elif category == "Test Variance":
        run_test_variance(df)
    elif category == "Parametric Tests":
        run_parametric_tests(df, parametric_test)
    elif category == "Non-parametric Tests":
        run_nonparametric_tests(df, nonparametric_test)
    else:
        st.info("👈 Select a category from the sidebar to get started.")


def _render_test_list(tests: list, df: pd.DataFrame):
    for test in tests:
        with st.expander(f"**{test['name']}**"):
            st.markdown(f"**What it does:** {test['what']}")
            st.markdown(f"**When to use it:** {test['when']}")
            st.markdown("---")
            test['fn'](df)

