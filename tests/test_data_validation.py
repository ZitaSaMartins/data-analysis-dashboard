"""Smoke tests for modules.data_validation."""

import pandas as pd

from modules.data_validation import (
    analyze_variable_types,
    generate_validation_report,
    get_statistical_summary,
    validate_dataset,
)


def test_validate_dataset_clean_data_has_no_issues():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": ["x", "y", "z", "w"]})
    result = validate_dataset(df)
    assert result["is_valid"] is True
    assert result["issues"] == []


def test_validate_dataset_flags_empty_dataframe():
    result = validate_dataset(pd.DataFrame())
    assert result["is_valid"] is False
    assert any("empty" in issue.lower() for issue in result["issues"])


def test_validate_dataset_flags_duplicate_columns():
    df = pd.DataFrame([[1, 2]], columns=["a", "a"])
    result = validate_dataset(df)
    assert result["is_valid"] is False
    assert any("duplicate" in issue.lower() for issue in result["issues"])


def test_validate_dataset_flags_single_value_column():
    df = pd.DataFrame({"a": [1, 2, 3], "constant": [5, 5, 5]})
    result = validate_dataset(df)
    assert any("only one unique value" in issue for issue in result["issues"])


def test_validate_dataset_flags_mixed_numeric_text_column():
    df = pd.DataFrame({"mixed": ["1", "2", "not_a_number", "4"]})
    result = validate_dataset(df)
    assert any("mixed numeric and text" in issue for issue in result["issues"])


def test_generate_validation_report_includes_row_and_column_counts():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    report = generate_validation_report(df)
    assert "Total Rows: 2" in report
    assert "Total Columns: 2" in report


def test_analyze_variable_types_classifies_binary_continuous_categorical():
    df = pd.DataFrame(
        {
            "flag": [0, 1] * 15,
            "score": range(30),
            "category": ["a", "b", "c"] * 10,
        }
    )
    result = analyze_variable_types(df)
    assert "flag" in result["binary"]
    assert "score" in result["continuous"]
    assert "category" in result["categorical"]


def test_get_statistical_summary_numeric_column():
    df = pd.DataFrame({"value": [1, 2, 3, 4, 5]})
    summary = get_statistical_summary(df, "value")
    assert summary["type"] == "numeric"
    assert summary["count"] == 5
    assert summary["mean"] == 3
    assert summary["min"] == 1
    assert summary["max"] == 5


def test_get_statistical_summary_categorical_column():
    df = pd.DataFrame({"label": ["a", "a", "b"]})
    summary = get_statistical_summary(df, "label")
    assert summary["type"] == "categorical"
    assert summary["unique"] == 2
    assert summary["most_common"] == "a"
    assert summary["most_common_freq"] == 2
