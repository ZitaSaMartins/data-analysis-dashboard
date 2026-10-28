"""Part-to-whole and hierarchical charts: pie, donut, treemap, sunburst."""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from .registry import _register
from .theme import _category_colors, _palette, _plot


def _pie_chart(df: pd.DataFrame, settings: dict, dependent_vars: list, independent_vars: list,
               hole: float, key_prefix: str):
    """Shared renderer for Pie Chart and Donut Chart."""
    st.markdown("### Donut Chart" if hole else "### Pie Chart")

    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not cat_cols:
        st.info("Need at least one categorical variable selected.")
        return

    indep_cat = [c for c in independent_vars if c in cat_cols]
    c1, c2, c3 = st.columns(3)
    with c1:
        cat_var = st.selectbox("Categorical variable:", cat_cols,
                               index=cat_cols.index(indep_cat[0]) if indep_cat else 0, key=f"{key_prefix}_cat")
    with c2:
        value_choice = st.selectbox("Value:", ["Count"] + numeric_cols, key=f"{key_prefix}_value")
    with c3:
        top_n = st.slider("Max slices shown", 2, 15, 8, key=f"{key_prefix}_top_n")

    if value_choice == "Count":
        agg = df[cat_var].value_counts().head(top_n).reset_index()
        agg.columns = [cat_var, "Value"]
    else:
        agg = df.groupby(cat_var)[value_choice].sum().sort_values(ascending=False).head(top_n).reset_index()
        agg.columns = [cat_var, "Value"]

    default_title = f"{cat_var} distribution" if value_choice == "Count" else f"{value_choice} by {cat_var}"

    with st.expander(f"🎨 Customize: {default_title}"):
        pie_title = st.text_input("Title", default_title, key=f"{key_prefix}_title")
        st.caption("Slice colors:")
        slice_colors = _category_colors(agg[cat_var].tolist(), settings, key_prefix=key_prefix)

    fig = px.pie(agg, names=cat_var, values="Value", hole=hole, color=cat_var,
                color_discrete_map=slice_colors)
    _plot(fig, settings, title=pie_title)


_register("Pie Chart")(lambda df, s, d, i: _pie_chart(df, s, d, i, hole=0.0, key_prefix="pie"))
_register("Donut Chart")(lambda df, s, d, i: _pie_chart(df, s, d, i, hole=0.45, key_prefix="donut"))


def _hierarchical_chart(df: pd.DataFrame, settings: dict, dependent_vars: list, independent_vars: list, kind: str):
    """Shared renderer for Treemap and Sunburst."""
    st.markdown("### Treemap" if kind == "treemap" else "### Sunburst")

    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not cat_cols:
        st.info("Need at least one categorical variable selected.")
        return

    c1, c2 = st.columns(2)
    with c1:
        path_vars = st.multiselect("Hierarchy levels (categorical):", cat_cols,
                                   default=cat_cols[:min(2, len(cat_cols))], key=f"{kind}_path")
    with c2:
        value_choice = st.selectbox("Value:", ["Count"] + numeric_cols, key=f"{kind}_value")

    if not path_vars:
        st.info("Select at least one categorical variable.")
        return

    values_col = None if value_choice == "Count" else value_choice
    default_title = " > ".join(path_vars)
    palette = _palette(settings)

    color_options = ["Palette"]
    if values_col:
        color_options.append(f"{values_col} (scale)")

    with st.expander(f"🎨 Customize: {default_title}"):
        chart_title = st.text_input("Title", default_title, key=f"{kind}_title")
        color_mode = st.radio("Color by", color_options, key=f"{kind}_color_mode", horizontal=True)

    kwargs = dict(path=[px.Constant("All")] + path_vars, values=values_col)
    if color_mode == "Palette":
        kwargs["color_discrete_sequence"] = palette
    else:
        kwargs["color"] = values_col
        kwargs["color_continuous_scale"] = settings["colorscale"]

    fig = px.treemap(df, **kwargs) if kind == "treemap" else px.sunburst(df, **kwargs)
    _plot(fig, settings, title=chart_title)


_register("Treemap")(lambda df, s, d, i: _hierarchical_chart(df, s, d, i, "treemap"))
_register("Sunburst")(lambda df, s, d, i: _hierarchical_chart(df, s, d, i, "sunburst"))

