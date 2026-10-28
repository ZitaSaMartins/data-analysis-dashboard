"""Chart renderer registry and the EDA page entry point."""

import numpy as np
import pandas as pd
from scipy import stats
import streamlit as st

from .theme import _settings


_RENDERERS = {}


def _register(*names):
    def wrapper(fn):
        for name in names:
            _RENDERERS[name] = fn
        return fn
    return wrapper


def run_eda(df: pd.DataFrame, components: list, variables: list = None,
            dependent_vars: list = None, independent_vars: list = None,
            chart_settings: dict = None):
    """Run selected EDA components, restricted to the chosen dependent/independent variables."""
    settings = _settings(chart_settings)
    dependent_vars = dependent_vars or []
    independent_vars = independent_vars or []

    # Work only with the variables selected on the EDA page; fall back to all columns
    cols = variables if variables else df.columns.tolist()
    data = df[cols]

    if "Summary Statistics" in components:
        show_summary_statistics(data)

    for name in components:
        renderer = _RENDERERS.get(name)
        if renderer:
            renderer(data, settings, dependent_vars, independent_vars)


def show_summary_statistics(df: pd.DataFrame):
    """Display summary statistics for numeric columns."""
    st.markdown("### Summary Statistics")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        st.info("No numeric columns selected for summary statistics.")
        return
    
    summary = df[numeric_cols].describe().T
    summary['Missing'] = df[numeric_cols].isnull().sum()
    summary['Missing %'] = (summary['Missing'] / len(df) * 100).round(2)
    
    st.dataframe(summary, use_container_width=True)
    
    # Skewness and Kurtosis
    st.markdown("#### Distribution Shape (Skewness & Kurtosis)")
    
    skew_kurt = pd.DataFrame({
        'Column': numeric_cols,
        'Skewness': [stats.skew(df[col].dropna()) for col in numeric_cols],
        'Kurtosis': [stats.kurtosis(df[col].dropna()) for col in numeric_cols]
    })
    
    st.dataframe(skew_kurt, use_container_width=True)
    
    st.caption("Skewness: Negative (left-skewed) | 0 (symmetric) | Positive (right-skewed)")
    st.caption("Kurtosis: Negative (flat) | 0 (normal) | Positive (peaked)")

