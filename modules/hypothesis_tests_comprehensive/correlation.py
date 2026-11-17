"""Correlation and simple linear regression tests."""

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

from modules.stat_format import (
    format_alpha,
    select_significance_level,
    with_formatted_p_values,
)

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
    alpha = select_significance_level()
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
            alpha,
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
    selected_variables = st.multiselect(
        "Variables:",
        numeric_columns,
        default=numeric_columns[: min(3, len(numeric_columns))],
        key=f"{method}_variables",
    )
    alpha = select_significance_level()
    if st.button(
        f"Run {title.removeprefix('🟦 ').removeprefix('🟧 ')}",
        type="primary",
        key=f"run_{method}",
    ):
        if len(selected_variables) < 2:
            st.error("Select at least two variables.")
            return
        if method == "pearson":
            statistic_label = "Pearson r"
        elif method == "spearman":
            statistic_label = "Spearman ρ"
        else:
            statistic_label = "Kendall τ"

        coefficient_matrix = pd.DataFrame(
            np.nan, index=selected_variables, columns=selected_variables
        )
        for variable in selected_variables:
            coefficient_matrix.loc[variable, variable] = 1.0

        summary_rows: list[dict[str, object]] = []
        for first_index, first in enumerate(selected_variables):
            for second in selected_variables[first_index + 1 :]:
                paired = df[[first, second]].dropna().astype(float)
                if len(paired) < 3:
                    st.warning(
                        f"'{first}' vs '{second}': at least three complete pairs "
                        "are required; this pair was skipped."
                    )
                    continue
                if method == "pearson":
                    statistic, p_value = stats.pearsonr(paired[first], paired[second])
                elif method == "spearman":
                    statistic, p_value = stats.spearmanr(paired[first], paired[second])
                else:
                    statistic, p_value = stats.kendalltau(paired[first], paired[second])
                coefficient_matrix.loc[first, second] = statistic
                coefficient_matrix.loc[second, first] = statistic
                is_significant = np.isfinite(p_value) and p_value <= alpha
                summary_rows.append({
                    "Variable 1": first,
                    "Variable 2": second,
                    "n": len(paired),
                    statistic_label: statistic,
                    "p-value": p_value,
                    "Interpretation": (
                        "Significant association ✅"
                        if is_significant
                        else "No significant association ❌"
                    ),
                })

        if not summary_rows:
            st.error(
                "No valid pairs were produced. Select variables with at least "
                "three complete observations in common."
            )
            return

        st.markdown("#### Correlation matrix")
        st.dataframe(
            coefficient_matrix,
            width="stretch",
            column_config={
                column: st.column_config.NumberColumn(format="%.3f")
                for column in coefficient_matrix.columns
            },
        )
        st.markdown("#### Summary")
        st.dataframe(
            with_formatted_p_values(pd.DataFrame(summary_rows), ["p-value"]),
            hide_index=True,
            width="stretch",
            column_config={
                statistic_label: st.column_config.NumberColumn(format="%.4f"),
            },
        )
        st.caption(f"Interpretation uses the selected significance level α = {format_alpha(alpha)}.")

