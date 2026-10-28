"""Correlation charts: heatmap, scatter, bubble, contour."""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from .registry import _register
from .theme import COLORSCALE_OPTIONS, _category_colors, _palette, _plot


@_register("Correlation Heatmap")
def _correlation_heatmap_chart(df: pd.DataFrame, settings: dict, dependent_vars: list, independent_vars: list):
    """Correlation matrix heatmap."""
    st.markdown("### Correlation Heatmap")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        st.info("Need at least 2 numeric columns selected for correlation analysis.")
        return

    corr = df[numeric_cols].corr()

    with st.expander("🎨 Customize: Correlation Matrix"):
        c1, c2, c3 = st.columns(3)
        with c1:
            corr_colorscale = st.selectbox(
                "Color scale", COLORSCALE_OPTIONS,
                index=COLORSCALE_OPTIONS.index(settings["colorscale"]) if settings["colorscale"] in COLORSCALE_OPTIONS else 0,
                key=f"corr_colorscale_{settings['colorscale']}"
            )
        with c2:
            show_values = st.checkbox("Show values", value=True, key="corr_show_values")
        with c3:
            corr_title = st.text_input("Title", "Correlation Matrix", key="corr_title")

    fig = px.imshow(
        corr, text_auto='.2f' if show_values else False, color_continuous_scale=corr_colorscale,
        zmin=-1, zmax=1, aspect="auto",
    )
    _plot(fig, settings, title=corr_title)

    st.markdown("#### Strong Correlations (|r| > 0.7)")

    strong_corr = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            if abs(corr.iloc[i, j]) > 0.7:
                strong_corr.append({
                    'Variable 1': corr.columns[i],
                    'Variable 2': corr.columns[j],
                    'Correlation': corr.iloc[i, j]
                })

    if strong_corr:
        st.dataframe(pd.DataFrame(strong_corr), use_container_width=True)
    else:
        st.info("No strong correlations found.")


@_register("Scatter Plot")
def _scatter_chart(df: pd.DataFrame, settings: dict, dependent_vars: list, independent_vars: list):
    """Relationship between two numeric variables."""
    st.markdown("### Scatter Plot")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        st.info("Need at least 2 numeric columns selected for scatter plots.")
        return

    dep_numeric = [c for c in dependent_vars if c in numeric_cols]
    indep_numeric = [c for c in independent_vars if c in numeric_cols]

    col1, col2, col3 = st.columns(3)
    with col1:
        x_var = st.selectbox(
            "Select X variable:", numeric_cols,
            index=numeric_cols.index(indep_numeric[0]) if indep_numeric else 0, key="scatter_x"
        )
    with col2:
        y_options = [col for col in numeric_cols if col != x_var]
        default_y = next((c for c in dep_numeric if c != x_var), y_options[0])
        y_var = st.selectbox(
            "Select Y variable:", y_options, index=y_options.index(default_y), key="scatter_y"
        )
    with col3:
        marker_size = st.slider("Marker size", 4, 20, 9, key="scatter_size")

    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    color_by = None

    if cat_cols:
        use_color = st.checkbox("Color by categorical variable?", key="scatter_use_color")
        if use_color:
            color_by = st.selectbox("Select coloring variable:", cat_cols, key="scatter_color_by")

    if not (x_var and y_var):
        return

    default_title = f'{y_var} vs {x_var}'
    palette = _palette(settings)

    with st.expander(f"🎨 Customize: {default_title}"):
        c1, c2 = st.columns([2, 1])
        with c1:
            scatter_title = st.text_input("Title", default_title, key="scatter_title")
        with c2:
            opacity = st.slider("Opacity", 0.1, 1.0, 0.75, key="scatter_opacity")

        if color_by:
            st.caption("Category colors:")
            categories = sorted(df[color_by].dropna().unique().tolist())[:10]
            category_colors = _category_colors(categories, settings, key_prefix="scatter")
            color_kwargs = {"color_discrete_map": category_colors}
        else:
            marker_color = st.color_picker("Marker color", palette[0], key=f"scatter_color_{settings['palette']}")
            color_kwargs = {"color_discrete_sequence": [marker_color]}

    fig = px.scatter(df, x=x_var, y=y_var, color=color_by, opacity=opacity, **color_kwargs)
    fig.update_traces(marker=dict(size=marker_size, line=dict(width=0.5, color='white')))
    fig.update_xaxes(title=x_var)
    fig.update_yaxes(title=y_var)
    _plot(fig, settings, title=scatter_title)

    corr = df[[x_var, y_var]].corr().iloc[0, 1]
    st.caption(f"Pearson correlation: r = {corr:.3f}")


@_register("Bubble Chart")
def _bubble_chart(df: pd.DataFrame, settings: dict, dependent_vars: list, independent_vars: list):
    """Scatter plot with a third numeric variable encoded as bubble size."""
    st.markdown("### Bubble Chart")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 3:
        st.info("Need at least 3 numeric variables selected for a bubble chart.")
        return

    dep_numeric = [c for c in dependent_vars if c in numeric_cols]
    indep_numeric = [c for c in independent_vars if c in numeric_cols]

    col1, col2, col3 = st.columns(3)
    with col1:
        x_var = st.selectbox("X variable:", numeric_cols,
                             index=numeric_cols.index(indep_numeric[0]) if indep_numeric else 0, key="bubble_x")
    with col2:
        y_options = [c for c in numeric_cols if c != x_var]
        default_y = next((c for c in dep_numeric if c != x_var), y_options[0])
        y_var = st.selectbox("Y variable:", y_options, index=y_options.index(default_y), key="bubble_y")
    with col3:
        size_options = [c for c in numeric_cols if c not in (x_var, y_var)]
        size_var = st.selectbox("Size variable:", size_options, key="bubble_size")

    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    color_by = None
    if cat_cols:
        use_color = st.checkbox("Color by categorical variable?", key="bubble_use_color")
        if use_color:
            color_by = st.selectbox("Coloring variable:", cat_cols, key="bubble_color_by")

    default_title = f"{y_var} vs {x_var} (size: {size_var})"
    palette = _palette(settings)

    with st.expander(f"🎨 Customize: {default_title}"):
        bubble_title = st.text_input("Title", default_title, key="bubble_title")
        max_size = st.slider("Max bubble size", 10, 80, 40, key="bubble_max_size")
        if color_by:
            st.caption("Category colors:")
            categories = sorted(df[color_by].dropna().unique().tolist())[:10]
            category_colors = _category_colors(categories, settings, key_prefix="bubble")
            color_kwargs = {"color_discrete_map": category_colors}
        else:
            bubble_color = st.color_picker("Bubble color", palette[0], key=f"bubble_color_{settings['palette']}")
            color_kwargs = {"color_discrete_sequence": [bubble_color]}

    plot_df = df.dropna(subset=[size_var])
    if plot_df.empty:
        st.info("No rows have a value for the selected size variable.")
        return
    plot_df = plot_df.assign(_bubble_size=plot_df[size_var].abs())
    fig = px.scatter(plot_df, x=x_var, y=y_var, size="_bubble_size", color=color_by,
                     size_max=max_size, opacity=0.7, **color_kwargs)
    fig.update_xaxes(title=x_var)
    fig.update_yaxes(title=y_var)
    _plot(fig, settings, title=bubble_title)


@_register("Contour Plot")
def _contour_chart(df: pd.DataFrame, settings: dict, dependent_vars: list, independent_vars: list):
    """Density contour between two numeric variables."""
    st.markdown("### Contour Plot")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        st.info("Need at least 2 numeric variables selected for a contour plot.")
        return

    dep_numeric = [c for c in dependent_vars if c in numeric_cols]
    indep_numeric = [c for c in independent_vars if c in numeric_cols]

    col1, col2 = st.columns(2)
    with col1:
        x_var = st.selectbox("X variable:", numeric_cols,
                             index=numeric_cols.index(indep_numeric[0]) if indep_numeric else 0, key="contour_x")
    with col2:
        y_options = [c for c in numeric_cols if c != x_var]
        default_y = next((c for c in dep_numeric if c != x_var), y_options[0])
        y_var = st.selectbox("Y variable:", y_options, index=y_options.index(default_y), key="contour_y")

    default_title = f"Density of {y_var} vs {x_var}"

    with st.expander(f"🎨 Customize: {default_title}"):
        contour_title = st.text_input("Title", default_title, key="contour_title")
        filled = st.checkbox("Filled contours", value=True, key="contour_filled")
        contour_colorscale = st.selectbox(
            "Color scale", COLORSCALE_OPTIONS,
            index=COLORSCALE_OPTIONS.index(settings["colorscale"]) if settings["colorscale"] in COLORSCALE_OPTIONS else 0,
            key=f"contour_colorscale_{settings['colorscale']}"
        )

    fig = px.density_contour(df, x=x_var, y=y_var)
    fig.update_traces(contours_coloring="fill" if filled else "lines", colorscale=contour_colorscale)
    fig.update_xaxes(title=x_var)
    fig.update_yaxes(title=y_var)
    _plot(fig, settings, title=contour_title)

