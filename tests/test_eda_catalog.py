"""Smoke tests for modules.eda.catalog."""

import pandas as pd

from modules.eda.catalog import CHART_CATALOG, CHART_INFO, recommend_components


def test_chart_info_includes_every_catalog_entry_and_summary_statistics():
    catalog_chart_names = {name for charts in CHART_CATALOG.values() for name in charts}
    assert catalog_chart_names <= CHART_INFO.keys()
    assert "Summary Statistics" in CHART_INFO


def test_recommend_components_flags_summary_statistics_only_with_numeric_data():
    numeric_df = pd.DataFrame({"value": [1, 2, 3]})
    categorical_df = pd.DataFrame({"label": ["a", "b", "c"]})
    assert recommend_components(numeric_df)["Summary Statistics"] is True
    assert recommend_components(categorical_df)["Summary Statistics"] is False


def test_recommend_components_scatter_requires_two_numeric_columns():
    one_numeric = pd.DataFrame({"x": [1, 2, 3]})
    two_numeric = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    assert recommend_components(one_numeric)["Scatter Plot"] is False
    assert recommend_components(two_numeric)["Scatter Plot"] is True


def test_recommend_components_pie_chart_requires_categorical_column():
    numeric_only = pd.DataFrame({"x": [1, 2, 3]})
    with_category = pd.DataFrame({"x": [1, 2, 3], "label": ["a", "b", "c"]})
    assert recommend_components(numeric_only)["Pie Chart"] is False
    assert recommend_components(with_category)["Pie Chart"] is True


def test_recommend_components_respects_variables_subset():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6], "label": ["a", "b", "c"]})
    # Restricting to a single numeric column should behave like a one-numeric-column dataset.
    result = recommend_components(df, variables=["x"])
    assert result["Scatter Plot"] is False
    assert result["Histogram"] is True
