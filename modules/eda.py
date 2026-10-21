import re
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

DEFAULT_SETTINGS = {
    "template": "plotly_white",
    "palette": "Plotly",
    "colorscale": "RdBu",
    "height": 450,
    "font_family": "Arial",
    "font_size": 14,
}

COLORSCALE_OPTIONS = ["RdBu", "Viridis", "Plasma", "Blues", "Greens", "Reds", "Turbo", "Cividis", "Sunset", "Tealrose"]

# Chart types grouped the way a chart-picker UI usually presents them. Only chart types
# that Plotly can genuinely render well are included (no word clouds, pictograms, Venn
# diagrams, waffle/marimekko charts, etc. - Plotly has no native support for those).
CHART_CATALOG = {
    "Comparison": {
        "Bar Chart": {"icon": "📶", "desc": "Compare a numeric value across categories.",
                      "requires": lambda n, c: len(n) >= 1 and len(c) >= 1},
        "Grouped Bar Chart": {"icon": "📊", "desc": "Compare several numeric values across categories, side by side.",
                              "requires": lambda n, c: len(n) >= 2 and len(c) >= 1},
        "Radar Chart": {"icon": "🕸️", "desc": "Compare several numeric variables on a radial axis.",
                        "requires": lambda n, c: len(n) >= 3},
    },
    "Correlation": {
        "Scatter Plot": {"icon": "⚡", "desc": "Relationship between two numeric variables.",
                         "requires": lambda n, c: len(n) >= 2},
        "Bubble Chart": {"icon": "🫧", "desc": "Scatter plot with a third variable encoded as bubble size.",
                         "requires": lambda n, c: len(n) >= 3},
        "Correlation Heatmap": {"icon": "🔗", "desc": "Which numeric variables move together.",
                                "requires": lambda n, c: len(n) >= 2},
        "Contour Plot": {"icon": "🌄", "desc": "Density of points between two numeric variables.",
                         "requires": lambda n, c: len(n) >= 2},
    },
    "Part-to-Whole & Hierarchical": {
        "Pie Chart": {"icon": "🥧", "desc": "Share of each category out of the whole.",
                      "requires": lambda n, c: len(c) >= 1},
        "Donut Chart": {"icon": "🍩", "desc": "Pie chart with a hollow center.",
                        "requires": lambda n, c: len(c) >= 1},
        "Treemap": {"icon": "🌳", "desc": "Nested rectangles sized by value, grouped by category.",
                    "requires": lambda n, c: len(c) >= 1},
        "Sunburst": {"icon": "☀️", "desc": "Nested rings showing hierarchy, like a radial treemap.",
                     "requires": lambda n, c: len(c) >= 1},
    },
    "Temporal": {
        "Line Chart": {"icon": "📈", "desc": "How a numeric value changes across a sequence.",
                       "requires": lambda n, c: len(n) >= 1},
        "Area Chart": {"icon": "🏔️", "desc": "Line chart with the area under the line filled in.",
                       "requires": lambda n, c: len(n) >= 1},
        "Stacked Area Chart": {"icon": "🗻", "desc": "Area chart split and stacked by category.",
                               "requires": lambda n, c: len(n) >= 1 and len(c) >= 1},
    },
    "Distribution": {
        "Histogram": {"icon": "📊", "desc": "How values are spread out, in bins.",
                      "requires": lambda n, c: len(n) >= 1},
        "Categorical Bar": {"icon": "🗂️", "desc": "Frequency of each category.",
                            "requires": lambda n, c: len(c) >= 1},
        "Box Plot": {"icon": "📦", "desc": "Median, quartiles & outliers of a numeric variable by category.",
                     "requires": lambda n, c: len(n) >= 1 and len(c) >= 1},
        "Violin Plot": {"icon": "🎻", "desc": "Box plot combined with a density curve.",
                        "requires": lambda n, c: len(n) >= 1 and len(c) >= 1},
        "Strip Plot": {"icon": "🔸", "desc": "Every individual point, jittered by category.",
                       "requires": lambda n, c: len(n) >= 1 and len(c) >= 1},
    },
}

CHART_INFO = {name: info for charts in CHART_CATALOG.values() for name, info in charts.items()}
CHART_INFO["Summary Statistics"] = {"icon": "📋", "desc": "Means, std dev, skewness & kurtosis as a table."}


def _settings(chart_settings):
    """Merge user-provided chart settings with defaults."""
    merged = DEFAULT_SETTINGS.copy()
    if chart_settings:
        merged.update({k: v for k, v in chart_settings.items() if v is not None})
    return merged


def _to_hex(color: str) -> str:
    """Convert a Plotly rgb()/rgba() color string to hex; st.color_picker only accepts hex."""
    if color.startswith('#'):
        return color
    match = re.match(r'rgba?\((\d+),\s*(\d+),\s*(\d+)', color)
    if match:
        r, g, b = (int(v) for v in match.groups())
        return f'#{r:02X}{g:02X}{b:02X}'
    return color


def _palette(settings):
    # Some Plotly qualitative palettes (e.g. Pastel, Set2, Alphabet) use rgb() strings
    # instead of hex, which crashes st.color_picker when used as a default value.
    colors = getattr(px.colors.qualitative, settings["palette"], px.colors.qualitative.Plotly)
    return [_to_hex(c) for c in colors]


def _plot(fig, settings, title=None):
    """Apply shared theme/font/size and render so every chart fits its container responsively."""
    fig.update_layout(
        template=settings["template"],
        font=dict(family=settings["font_family"], size=settings["font_size"]),
        height=settings["height"],
        autosize=True,
        margin=dict(l=40, r=20, t=60 if title else 30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=settings["font_size"] + 4)))
    st.plotly_chart(fig, use_container_width=True)


def _category_colors(categories: list, settings: dict, key_prefix: str) -> dict:
    """Let the user override each category's color after the chart is generated."""
    palette = _palette(settings)
    colors = {}
    cols = st.columns(min(len(categories), 4) or 1)
    for i, cat in enumerate(categories):
        default_color = palette[i % len(palette)]
        with cols[i % len(cols)]:
            colors[cat] = st.color_picker(str(cat), default_color,
                                          key=f"{key_prefix}_{settings['palette']}_color_{cat}")
    return colors


def recommend_components(df: pd.DataFrame, variables: list = None) -> dict:
    """Flag which chart types are a good fit for the currently selected variables."""
    cols = variables if variables else df.columns.tolist()
    data = df[cols]
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

    result = {"Summary Statistics": len(numeric_cols) >= 1}
    for charts in CHART_CATALOG.values():
        for name, info in charts.items():
            result[name] = info["requires"](numeric_cols, cat_cols)
    return result


def build_preview(kind: str, df: pd.DataFrame, chart_settings: dict = None):
    """Build a small, read-only preview of a chart type using the real selected data."""
    settings = _settings(chart_settings)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    palette = _palette(settings)
    fig = None

    try:
        if kind == "Bar Chart" and numeric_cols and cat_cols:
            agg = df.groupby(cat_cols[0])[numeric_cols[0]].mean().head(6).reset_index()
            fig = px.bar(agg, x=cat_cols[0], y=numeric_cols[0], color_discrete_sequence=[palette[0]])
        elif kind == "Grouped Bar Chart" and len(numeric_cols) >= 2 and cat_cols:
            agg = df.groupby(cat_cols[0])[numeric_cols[:2]].mean().head(6).reset_index()
            long = agg.melt(id_vars=cat_cols[0], var_name="Variable", value_name="Value")
            fig = px.bar(long, x=cat_cols[0], y="Value", color="Variable", barmode="group",
                         color_discrete_sequence=palette)
        elif kind == "Radar Chart" and len(numeric_cols) >= 3:
            axes = numeric_cols[:5]
            scaled = df[axes].apply(lambda s: (s - s.min()) / (s.max() - s.min()) if s.max() != s.min() else s * 0)
            r = scaled.mean().tolist()
            fig = go.Figure(go.Scatterpolar(r=r + [r[0]], theta=axes + [axes[0]], fill='toself',
                                             line_color=palette[0]))
        elif kind == "Scatter Plot" and len(numeric_cols) >= 2:
            fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1],
                              color_discrete_sequence=[palette[0]], opacity=0.7)
        elif kind == "Bubble Chart" and len(numeric_cols) >= 3:
            plot_df = df.assign(_size=df[numeric_cols[2]].abs())
            fig = px.scatter(plot_df, x=numeric_cols[0], y=numeric_cols[1], size="_size",
                              color_discrete_sequence=[palette[0]], opacity=0.7)
        elif kind == "Correlation Heatmap" and len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr()
            fig = px.imshow(corr, color_continuous_scale=settings["colorscale"], zmin=-1, zmax=1, aspect="auto")
        elif kind == "Contour Plot" and len(numeric_cols) >= 2:
            fig = px.density_contour(df, x=numeric_cols[0], y=numeric_cols[1])
            fig.update_traces(contours_coloring="fill", colorscale=settings["colorscale"])
        elif kind in ("Pie Chart", "Donut Chart") and cat_cols:
            vc = df[cat_cols[0]].value_counts().head(6).reset_index()
            vc.columns = [cat_cols[0], 'Count']
            fig = px.pie(vc, names=cat_cols[0], values='Count', hole=0.45 if kind == "Donut Chart" else 0,
                         color_discrete_sequence=palette)
        elif kind == "Treemap" and cat_cols:
            fig = px.treemap(df, path=cat_cols[:2], color_discrete_sequence=palette)
        elif kind == "Sunburst" and cat_cols:
            fig = px.sunburst(df, path=cat_cols[:2], color_discrete_sequence=palette)
        elif kind in ("Line Chart", "Area Chart") and numeric_cols:
            plot_df = df.reset_index()
            fn = px.line if kind == "Line Chart" else px.area
            fig = fn(plot_df, x="index", y=numeric_cols[0], color_discrete_sequence=[palette[0]])
        elif kind == "Stacked Area Chart" and numeric_cols and cat_cols:
            plot_df = df.reset_index()
            fig = px.area(plot_df, x="index", y=numeric_cols[0], color=cat_cols[0],
                          color_discrete_sequence=palette)
        elif kind == "Histogram" and numeric_cols:
            fig = px.histogram(df, x=numeric_cols[0], nbins=20, color_discrete_sequence=[palette[0]])
        elif kind == "Categorical Bar" and cat_cols:
            vc = df[cat_cols[0]].value_counts().head(6).reset_index()
            vc.columns = [cat_cols[0], 'Count']
            fig = px.bar(vc, x=cat_cols[0], y='Count', color_discrete_sequence=[palette[0]])
        elif kind in ("Box Plot", "Violin Plot", "Strip Plot") and numeric_cols and cat_cols:
            top_cats = df[cat_cols[0]].value_counts().head(5).index
            sample = df[df[cat_cols[0]].isin(top_cats)]
            if kind == "Box Plot":
                fig = px.box(sample, x=cat_cols[0], y=numeric_cols[0], color_discrete_sequence=palette)
            elif kind == "Violin Plot":
                fig = px.violin(sample, x=cat_cols[0], y=numeric_cols[0], box=True, color_discrete_sequence=palette)
            else:
                fig = px.strip(sample, x=cat_cols[0], y=numeric_cols[0], color_discrete_sequence=palette)
    except Exception:
        return None

    if fig is None:
        return None

    fig.update_layout(
        template=settings["template"],
        height=140,
        margin=dict(l=4, r=4, t=4, b=4),
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        coloraxis_showscale=False,
    )
    return fig


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

    plot_df = df.assign(_bubble_size=df[size_var].abs())
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
