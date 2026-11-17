"""Shared p-value display formatting and significance-level selection.

Centralizes the "<0.001" display convention used across every statistical
test page (hypothesis tests and sensory analyses) so very small p-values are
never shown as deceptively precise near-zero floats (e.g.
``0.000000000000000000000000000000001``). Only affects how a p-value is
*displayed*; callers should keep using the raw numeric p-value for any
comparison, filtering, or further computation and only format it at the final
display call.

Also centralizes the shared significance-level (α) selector: every test page
renders the same control, using the same widget keys, so a value chosen on
one page is remembered and shown pre-selected on every other page while still
being changeable independently there.
"""

from collections.abc import Iterable

import numpy as np
import pandas as pd
import streamlit as st

P_VALUE_DISPLAY_FLOOR = 0.001

DEFAULT_SIGNIFICANCE_LEVEL = 0.05
SIGNIFICANCE_LEVEL_PRESETS: dict[str, float] = {
    "5% (default)": 0.05,
    "1%": 0.01,
    "10%": 0.10,
}
_SIGNIFICANCE_LEVEL_KEY = "significance_level"
_SIGNIFICANCE_CHOICE_KEY = "significance_level_choice"
_SIGNIFICANCE_CUSTOM_KEY = "significance_level_custom_percent"


def format_p_value(p_value: float | None, decimals: int = 4) -> str:
    """Format a single p-value for display.

    Returns ``"<0.001"`` for any finite p-value below
    :data:`P_VALUE_DISPLAY_FLOOR`, a fixed-point string otherwise, and
    ``"—"`` for ``None``/``NaN``/non-finite values (for example a test that
    reports a critical value instead of a p-value).
    """
    try:
        value = float(p_value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(value):
        return "—"
    if value < P_VALUE_DISPLAY_FLOOR:
        return f"<{P_VALUE_DISPLAY_FLOOR}"
    return f"{value:.{decimals}f}"


def with_formatted_p_values(
    data: pd.DataFrame,
    columns: Iterable[str],
    decimals: int = 4,
) -> pd.DataFrame:
    """Return a copy of ``data`` with the given p-value columns display-formatted.

    Only the listed columns are touched, and only if present, so this is safe
    to call even when a p-value column is conditionally added upstream. Always
    returns a new DataFrame: callers keep their original numeric frame for any
    filtering, sorting, or multiple-comparison correction that must happen
    with raw floats, and should call this only immediately before the final
    ``st.dataframe(...)`` display call.
    """
    formatted = data.copy()
    for column in columns:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(
                lambda value, decimals=decimals: format_p_value(value, decimals=decimals)
            )
    return formatted


def format_alpha(alpha: float) -> str:
    """Format a significance level as a plain decimal, e.g. ``0.05`` or ``0.073``."""
    return f"{alpha:g}"


def format_alpha_percent(alpha: float) -> str:
    """Format a significance level as a percentage, e.g. ``5%`` or ``7.3%``."""
    return f"{alpha * 100:g}%"


def get_significance_level() -> float:
    """Return the currently selected significance level without rendering a widget.

    Falls back to :data:`DEFAULT_SIGNIFICANCE_LEVEL` if no test page has
    rendered :func:`select_significance_level` yet this session.
    """
    return float(st.session_state.get(_SIGNIFICANCE_LEVEL_KEY, DEFAULT_SIGNIFICANCE_LEVEL))


def select_significance_level() -> float:
    """Render the shared significance-level (α) selector and return the chosen value.

    Defaults to 5%, with 1% and 10% presets plus a free-typed custom value.
    Every test page should call this (using this single shared function, not
    a page-local copy) so the widgets share the same keys everywhere: picking
    a value on one page leaves it pre-selected the next time any other page
    renders the selector, while still allowing an independent change there.
    """
    choice = st.selectbox(
        "Significance level (α):",
        [*SIGNIFICANCE_LEVEL_PRESETS, "Custom"],
        key=_SIGNIFICANCE_CHOICE_KEY,
    )
    if choice == "Custom":
        custom_percent = st.number_input(
            "Custom significance level (%):",
            min_value=0.01,
            max_value=50.0,
            value=DEFAULT_SIGNIFICANCE_LEVEL * 100,
            step=0.5,
            key=_SIGNIFICANCE_CUSTOM_KEY,
        )
        alpha = float(custom_percent) / 100
    else:
        alpha = SIGNIFICANCE_LEVEL_PRESETS[choice]
    st.session_state[_SIGNIFICANCE_LEVEL_KEY] = alpha
    return alpha
