"""Correlation and simple linear regression tests."""

import pandas as pd
import streamlit as st
from scipy import stats

from ._shared import _require_numeric, _show_inference, _show_statistic


def test_pearson_correlation(df: pd.DataFrame) -> None:
    """Run Pearson's linear correlation test."""
    _test_correlation(
        df,
        method="pearson",
        title="🟦 Pearson Correlation",
        description="Measures a linear relationship between two numeric variables.",
    )


def test_linear_regression(df: pd.DataFrame) -> None:
    """Run simple linear regression for one predictor and one outcome."""
    st.markdown("## 📈 Linear Regression")
    st.write("Models a numeric outcome from one numeric predictor.")
    numeric_columns = _require_numeric(df, minimum=2)
    if numeric_columns is None:
        return
    col1, col2 = st.columns(2)
    predictor = col1.selectbox(
        "Predictor (X):",
        numeric_columns,
        key="regression_predictor",
    )
    outcome_options = [column for column in numeric_columns if column != predictor]
    outcome = col2.selectbox(
        "Outcome (Y):",
        outcome_options,
        key="regression_outcome",
    )
    if st.button("Run Linear Regression", type="primary", key="run_regression"):
        data = df[[predictor, outcome]].dropna().astype(float)
        if len(data) < 3 or data[predictor].nunique() < 2:
            st.error("At least three paired observations and two X values are required.")
            return
        result = stats.linregress(data[predictor], data[outcome])
        slope, intercept, r_squared = st.columns(3)
        slope.metric("Slope", f"{result.slope:.4f}")
        intercept.metric("Intercept", f"{result.intercept:.4f}")
        r_squared.metric("R²", f"{result.rvalue ** 2:.4f}")
        _show_statistic("t-statistic", result.slope / result.stderr, result.pvalue)
        predictions = result.intercept + result.slope * data[predictor]
        output = data.copy()
        output["Predicted"] = predictions
        output["Residual"] = data[outcome] - predictions
        st.dataframe(output.head(100), use_container_width=True)
        _show_inference(
            result.pvalue,
            "The predictor has a significant linear relationship with the outcome",
            "No significant linear relationship was detected",
        )


def test_spearman(df: pd.DataFrame) -> None:
    """Run Spearman's rank correlation."""
    _test_correlation(
        df,
        method="spearman",
        title="🟧 Spearman Correlation",
        description="Measures a monotonic rank-based relationship.",
    )


def test_kendall(df: pd.DataFrame) -> None:
    """Run Kendall's tau rank correlation."""
    _test_correlation(
        df,
        method="kendall",
        title="🟧 Kendall's Tau",
        description="Measures ordinal association and handles small samples well.",
    )


def _test_correlation(
    df: pd.DataFrame,
    method: str,
    title: str,
    description: str,
) -> None:
    st.markdown(f"## {title}")
    st.write(description)
    numeric_columns = _require_numeric(df, minimum=2)
    if numeric_columns is None:
        return
    col1, col2 = st.columns(2)
    first = col1.selectbox(
        "First variable:",
        numeric_columns,
        key=f"{method}_first",
    )
    second_options = [column for column in numeric_columns if column != first]
    second = col2.selectbox(
        "Second variable:",
        second_options,
        key=f"{method}_second",
    )
    if st.button(
        f"Run {title.removeprefix('🟦 ').removeprefix('🟧 ')}",
        type="primary",
        key=f"run_{method}",
    ):
        paired = df[[first, second]].dropna().astype(float)
        if len(paired) < 3:
            st.error("At least three complete pairs are required.")
            return
        if method == "pearson":
            statistic, p_value = stats.pearsonr(paired[first], paired[second])
            statistic_label = "Pearson r"
        elif method == "spearman":
            statistic, p_value = stats.spearmanr(paired[first], paired[second])
            statistic_label = "Spearman ρ"
        else:
            statistic, p_value = stats.kendalltau(paired[first], paired[second])
            statistic_label = "Kendall τ"
        _show_statistic(statistic_label, statistic, p_value)
        _show_inference(
            p_value,
            "A significant association was detected",
            "No significant association was detected",
        )

