"""Temporal charts: line, area, stacked area."""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from .registry import _register
from .theme import _category_colors, _palette, _plot


def _time_series_chart(df: pd.DataFrame, settings: dict, dependent_vars: list, independent_vars: list, kind: str):
    """Shared renderer for Line Chart, Area Chart and Stacked Area Chart."""
    labels = {"line": "Line Chart", "area": "Area Chart", "stacked_area": "Stacked Area Chart"}
    st.markdown(f"### {labels[kind]}")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if not numeric_cols:
        st.info("Need at least one numeric variable selected.")
        return
    if kind == "stacked_area" and not cat_cols:
        st.info("Need a categorical variable selected to stack by.")
        return

    dep_numeric = [c for c in dependent_vars if c in numeric_cols]
    all_cols = df.columns.tolist()

    c1, c2, c3 = st.columns(3)
    with c1:
        x_var = st.selectbox("X variable (sequence):", all_cols, key=f"{kind}_x")
    with c2:
        y_var = st.selectbox("Y variable (numeric):", numeric_cols,
                             index=numeric_cols.index(dep_numeric[0]) if dep_numeric else 0, key=f"{kind}_y")
    with c3:
        if kind == "stacked_area":
            color_by = st.selectbox("Stack by (categorical):", cat_cols, key=f"{kind}_color")
        elif cat_cols:
            group_choice = st.selectbox("Group by (optional):", ["(none)"] + cat_cols, key=f"{kind}_color")
            color_by = None if group_choice == "(none)" else group_choice
        else:
            color_by = None

    plot_df = df.sort_values(x_var)
    default_title = f"{y_var} over {x_var}"
    palette = _palette(settings)

    with st.expander(f"🎨 Customize: {default_title}"):
        chart_title = st.text_input("Title", default_title, key=f"{kind}_title")
        line_shape = st.selectbox("Line shape", ["linear", "spline", "hv"], key=f"{kind}_shape") if kind == "line" else "linear"

        if color_by:
            st.caption("Series colors:")
            categories = sorted(plot_df[color_by].dropna().unique().tolist())[:10]
            series_colors = _category_colors(categories, settings, key_prefix=kind)
            color_kwargs = {"color_discrete_map": series_colors}
        else:
            single_color = st.color_picker("Line color", palette[0], key=f"{kind}_single_color_{settings['palette']}")
            color_kwargs = {"color_discrete_sequence": [single_color]}

    if kind == "line":
        fig = px.line(plot_df, x=x_var, y=y_var, color=color_by, markers=True,
                     line_shape=line_shape, **color_kwargs)
    else:
        fig = px.area(plot_df, x=x_var, y=y_var, color=color_by, **color_kwargs)

    fig.update_xaxes(title=x_var)
    fig.update_yaxes(title=y_var)
    _plot(fig, settings, title=chart_title)


_register("Line Chart")(lambda df, s, d, i: _time_series_chart(df, s, d, i, "line"))
_register("Area Chart")(lambda df, s, d, i: _time_series_chart(df, s, d, i, "area"))
_register("Stacked Area Chart")(lambda df, s, d, i: _time_series_chart(df, s, d, i, "stacked_area"))

