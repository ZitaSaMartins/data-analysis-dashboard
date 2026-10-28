"""Small read-only chart thumbnails used by the chart picker UI."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .theme import _palette, _settings


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

