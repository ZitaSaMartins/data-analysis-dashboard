"""Smoke tests for modules.stat_format."""

import numpy as np
import pandas as pd

from modules.stat_format import format_p_value, with_formatted_p_values


def test_format_p_value_below_floor_shows_less_than_threshold():
    assert format_p_value(0.0) == "<0.001"
    assert format_p_value(1e-30) == "<0.001"
    assert format_p_value(0.0009999) == "<0.001"


def test_format_p_value_at_or_above_floor_shows_fixed_decimals():
    assert format_p_value(0.001) == "0.0010"
    assert format_p_value(0.0423) == "0.0423"
    assert format_p_value(0.05) == "0.0500"
    assert format_p_value(1.0) == "1.0000"


def test_format_p_value_handles_missing_and_non_finite_values():
    assert format_p_value(None) == "—"
    assert format_p_value(np.nan) == "—"
    assert format_p_value(float("inf")) == "—"


def test_format_p_value_respects_custom_decimals():
    assert format_p_value(0.123456, decimals=2) == "0.12"


def test_with_formatted_p_values_only_touches_listed_columns():
    df = pd.DataFrame({"p-value": [0.0003, 0.02], "Statistic": [1.23456, 7.89]})
    formatted = with_formatted_p_values(df, ["p-value"])
    assert list(formatted["p-value"]) == ["<0.001", "0.0200"]
    # Untouched numeric column is unaffected.
    assert list(formatted["Statistic"]) == [1.23456, 7.89]


def test_with_formatted_p_values_does_not_mutate_the_original_dataframe():
    df = pd.DataFrame({"p-value": [0.0003, 0.02]})
    with_formatted_p_values(df, ["p-value"])
    # The original frame must remain numeric so callers can still filter/sort on it.
    assert pd.api.types.is_numeric_dtype(df["p-value"])


def test_with_formatted_p_values_ignores_missing_columns():
    df = pd.DataFrame({"other": [1, 2]})
    formatted = with_formatted_p_values(df, ["p-value"])
    assert list(formatted.columns) == ["other"]


def test_with_formatted_p_values_handles_none_entries():
    df = pd.DataFrame({"p-value": [0.01, None]})
    formatted = with_formatted_p_values(df, ["p-value"])
    assert list(formatted["p-value"]) == ["0.0100", "—"]
