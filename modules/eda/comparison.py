"""Comparison charts: bar, grouped bar, radar."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from .registry import _register
from .theme import _category_colors, _palette, _plot


def _bar_chart(df: pd.DataFrame, settings: dict, dependent_vars: list, independent_vars: list, grouped: bool):
    """Shared renderer for Bar Chart and Grouped Bar Chart."""
    st.markdown("### Grouped Bar Chart" if grouped else "### Bar Chart")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if not numeric_cols or not cat_cols:
        st.info("Need at least one numeric and one categorical variable selected.")
        return

    dep_numeric = [c for c in dependent_vars if c in numeric_cols]
    indep_cat = [c for c in independent_vars if c in cat_cols]
    suffix = "grouped" if grouped else "single"

    c1, c2, c3 = st.columns(3)
    with c1:
        cat_var = st.selectbox("Categorical variable (x-axis):", cat_cols,
                               index=cat_cols.index(indep_cat[0]) if indep_cat else 0, key=f"bar_cat_{suffix}")
    with c2:
        if grouped:
            default_num = dep_numeric[:2] if len(dep_numeric) >= 2 else numeric_cols[:2]
            num_vars = st.multiselect("Numeric variables (bars):", numeric_cols, default=default_num,
                                      key="bar_grouped_numeric")
        else:
            num_vars = [st.selectbox("Numeric variable (value):", numeric_cols,
                                     index=numeric_cols.index(dep_numeric[0]) if dep_numeric else 0, key="bar_numeric")]
    with c3:
        agg_label = st.selectbox("Aggregate with:", ["Mean", "Sum", "Median", "Count"], key=f"bar_agg_{suffix}")
        top_n = st.slider("Max categories shown", 2, 20, 10, key=f"bar_top_n_{suffix}")

    if not num_vars or not cat_var:
        return

    agg_map = {"Mean": "mean", "Sum": "sum", "Median": "median", "Count": "count"}
    top_cats = df[cat_var].value_counts().head(top_n).index
    sample = df[df[cat_var].isin(top_cats)]
    agg = sample.groupby(cat_var)[num_vars].agg(agg_map[agg_label]).reset_index()

    default_title = f"{', '.join(num_vars)} by {cat_var} ({agg_label})"
    palette = _palette(settings)

    if grouped:
        long = agg.melt(id_vars=cat_var, value_vars=num_vars, var_name="Variable", value_name="Value")
        with st.expander(f"🎨 Customize: {default_title}"):
            bar_title = st.text_input("Title", default_title, key="bar_grouped_title")
            st.caption("Series colors:")
            series_colors = _category_colors(num_vars, settings, key_prefix="bar_grouped")
        fig = px.bar(long, x=cat_var, y="Value", color="Variable", barmode="group",
                    color_discrete_map=series_colors)
    else:
        with st.expander(f"🎨 Customize: {default_title}"):
            bar_title = st.text_input("Title", default_title, key="bar_title")
            bar_color = st.color_picker("Bar color", palette[0], key=f"bar_color_{settings['palette']}")
        fig = px.bar(agg, x=cat_var, y=num_vars[0], color_discrete_sequence=[bar_color])

    fig.update_xaxes(title=cat_var)
    _plot(fig, settings, title=bar_title)


_register("Bar Chart")(lambda df, s, d, i: _bar_chart(df, s, d, i, grouped=False))
_register("Grouped Bar Chart")(lambda df, s, d, i: _bar_chart(df, s, d, i, grouped=True))


@_register("Radar Chart")
def _radar_chart(df: pd.DataFrame, settings: dict, dependent_vars: list, independent_vars: list):
    """Compare several numeric variables on a radial axis."""
    st.markdown("### Radar Chart")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if len(numeric_cols) < 3:
        st.info("Need at least 3 numeric variables selected for a radar chart.")
        return

    c1, c2 = st.columns(2)
    with c1:
        axes_vars = st.multiselect("Numeric variables (axes):", numeric_cols,
                                   default=numeric_cols[:min(5, len(numeric_cols))], key="radar_vars")
    with c2:
        group_choice = st.selectbox("Group by (optional):", ["(none)"] + cat_cols, key="radar_group")

    if len(axes_vars) < 3:
        st.info("Select at least 3 numeric variables.")
        return

    # Normalize to 0-1 so variables on different scales stay comparable
    scaled = df[axes_vars].apply(lambda s: (s - s.min()) / (s.max() - s.min()) if s.max() != s.min() else s * 0)

    if group_choice != "(none)":
        groups = df[group_choice].value_counts().head(6).index.tolist()
    else:
        groups = ["Overall"]

    default_title = "Radar Chart: " + ", ".join(axes_vars)
    with st.expander(f"🎨 Customize: {default_title}"):
        radar_title = st.text_input("Title", default_title, key="radar_title")
        st.caption("Series colors:")
        series_colors = _category_colors(groups, settings, key_prefix="radar")

    fig = go.Figure()
    theta = axes_vars + [axes_vars[0]]
    for g in groups:
        if group_choice != "(none)":
            r = scaled[df[group_choice] == g].mean().tolist()
        else:
            r = scaled.mean().tolist()
        r = r + [r[0]]
        fig.add_trace(go.Scatterpolar(r=r, theta=theta, fill='toself', name=str(g),
                                      line=dict(color=series_colors.get(g))))

    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
    _plot(fig, settings, title=radar_title)

