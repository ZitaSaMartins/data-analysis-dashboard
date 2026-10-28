"""Distribution charts: histogram, categorical bar, box/violin/strip."""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from .registry import _register
from .theme import _category_colors, _palette, _plot


@_register("Histogram")
def _histogram_chart(df: pd.DataFrame, settings: dict, dependent_vars: list, independent_vars: list):
    """Numeric distributions as histograms."""
    st.markdown("### Histogram")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    palette = _palette(settings)

    if not numeric_cols:
        st.info("Need at least one numeric variable selected.")
        return

    col1, col2 = st.columns([3, 1])
    with col1:
        selected_cols = st.multiselect(
            "Select numeric columns to visualize:",
            numeric_cols,
            default=numeric_cols[:min(3, len(numeric_cols))],
            key="numeric_dist"
        )
    with col2:
        n_bins = st.slider("Bins", 5, 100, 30, key="dist_bins")

    for idx, col in enumerate(selected_cols):
        default_title = f'Distribution of {col}'
        with st.expander(f"🎨 Customize: {default_title}"):
            c1, c2 = st.columns([1, 2])
            with c1:
                bar_color = st.color_picker("Bar color", palette[idx % len(palette)],
                                           key=f"dist_num_color_{settings['palette']}_{col}")
            with c2:
                custom_title = st.text_input("Title", default_title, key=f"dist_num_title_{col}")

        fig = px.histogram(
            df, x=col, nbins=n_bins,
            color_discrete_sequence=[bar_color],
            opacity=0.85,
        )
        fig.update_xaxes(title=col)
        fig.update_yaxes(title='Frequency')
        _plot(fig, settings, title=custom_title)


@_register("Categorical Bar")
def _categorical_bar_chart(df: pd.DataFrame, settings: dict, dependent_vars: list, independent_vars: list):
    """Frequency of each category as bar charts."""
    st.markdown("### Categorical Bar")

    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    palette = _palette(settings)

    if not cat_cols:
        st.info("Need at least one categorical variable selected.")
        return

    selected_cat = st.multiselect(
        "Select categorical columns to visualize:",
        cat_cols,
        default=cat_cols[:min(2, len(cat_cols))],
        key="cat_dist"
    )

    for idx, col in enumerate(selected_cat):
        default_title = f'Distribution of {col}'
        with st.expander(f"🎨 Customize: {default_title}"):
            c1, c2 = st.columns([1, 2])
            with c1:
                bar_color = st.color_picker("Bar color", palette[idx % len(palette)],
                                           key=f"dist_cat_color_{settings['palette']}_{col}")
            with c2:
                custom_title = st.text_input("Title", default_title, key=f"dist_cat_title_{col}")

        value_counts = df[col].value_counts().head(10).reset_index()
        value_counts.columns = [col, 'Count']
        fig = px.bar(
            value_counts, x=col, y='Count',
            color_discrete_sequence=[bar_color],
        )
        _plot(fig, settings, title=custom_title)


def _numeric_by_category_chart(df: pd.DataFrame, settings: dict, dependent_vars: list,
                                independent_vars: list, kind: str):
    """Shared renderer for Box/Violin/Strip plots: a numeric value split by a category."""
    labels = {"box": "Box Plot", "violin": "Violin Plot", "strip": "Strip Plot"}
    st.markdown(f"### {labels[kind]}")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if not numeric_cols or not cat_cols:
        st.info("Need both a numeric and a categorical variable selected.")
        return

    dep_numeric = [c for c in dependent_vars if c in numeric_cols]
    indep_cat = [c for c in independent_vars if c in cat_cols]

    col1, col2, col3 = st.columns(3)
    with col1:
        selected_numeric = st.selectbox(
            "Numeric variable (value):", numeric_cols,
            index=numeric_cols.index(dep_numeric[0]) if dep_numeric else 0, key=f"{kind}_numeric"
        )
    with col2:
        selected_cat = st.selectbox(
            "Categorical variable (grouping):", cat_cols,
            index=cat_cols.index(indep_cat[0]) if indep_cat else 0, key=f"{kind}_cat"
        )
    with col3:
        top_n = st.slider("Max categories shown", 2, 20, 10, key=f"{kind}_top_n")

    if not (selected_numeric and selected_cat):
        return

    top_cats = df[selected_cat].value_counts().head(top_n).index
    df_filtered = df[df[selected_cat].isin(top_cats)]
    default_title = f'{selected_numeric} by {selected_cat}'

    with st.expander(f"🎨 Customize: {default_title}"):
        chart_title = st.text_input("Title", default_title, key=f"{kind}_title")
        st.caption("Category colors:")
        category_colors = _category_colors(list(top_cats), settings, key_prefix=kind)

    if kind == "box":
        fig = px.box(df_filtered, x=selected_cat, y=selected_numeric, color=selected_cat,
                     color_discrete_map=category_colors, points="outliers")
    elif kind == "violin":
        fig = px.violin(df_filtered, x=selected_cat, y=selected_numeric, color=selected_cat,
                        color_discrete_map=category_colors, box=True, points="outliers")
    else:
        fig = px.strip(df_filtered, x=selected_cat, y=selected_numeric, color=selected_cat,
                       color_discrete_map=category_colors)

    fig.update_xaxes(title=selected_cat)
    fig.update_yaxes(title=selected_numeric)
    fig.update_layout(showlegend=False)
    _plot(fig, settings, title=chart_title)


_register("Box Plot")(lambda df, s, d, i: _numeric_by_category_chart(df, s, d, i, "box"))
_register("Violin Plot")(lambda df, s, d, i: _numeric_by_category_chart(df, s, d, i, "violin"))
_register("Strip Plot")(lambda df, s, d, i: _numeric_by_category_chart(df, s, d, i, "strip"))

