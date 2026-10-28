"""Shared chart theming: palettes, default settings and plot rendering."""

import re

import plotly.express as px
import streamlit as st


DEFAULT_SETTINGS = {
    "template": "plotly_white",
    "palette": "Plotly",
    "colorscale": "RdBu",
    "height": 450,
    "font_family": "Arial",
    "font_size": 14,
}


COLORSCALE_OPTIONS = ["RdBu", "Viridis", "Plasma", "Blues", "Greens", "Reds", "Turbo", "Cividis", "Sunset", "Tealrose"]


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

