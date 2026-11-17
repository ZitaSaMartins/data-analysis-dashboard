"""Shared p-value display formatting.

Centralizes the "<0.001" display convention used across every statistical
test page (hypothesis tests and sensory analyses) so very small p-values are
never shown as deceptively precise near-zero floats (e.g.
``0.000000000000000000000000000000001``). Only affects how a p-value is
*displayed*; callers should keep using the raw numeric p-value for any
comparison, filtering, or further computation and only format it at the final
display call.
"""

from collections.abc import Iterable

import numpy as np
import pandas as pd

P_VALUE_DISPLAY_FLOOR = 0.001


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
