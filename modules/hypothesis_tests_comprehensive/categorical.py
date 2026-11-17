"""Categorical association tests."""

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

from modules.stat_format import format_alpha, select_significance_level, with_formatted_p_values

from ._shared import _require_categorical, _show_variable_section_header


def test_chi_square(df: pd.DataFrame) -> None:
    """Run a chi-square test of independence between one variable and one or more others."""
    st.markdown("## 📋 Chi-Square Test")
    st.write(
        "Tests whether a categorical variable is associated with one or more "
        "other categorical variables."
    )
    categorical_columns = _require_categorical(df, minimum=2)
    if categorical_columns is None:
        return
    first = st.selectbox(
        "First categorical variable:",
        categorical_columns,
        key="chi_square_first",
    )
    second_options = [column for column in categorical_columns if column != first]
    second_variables = st.multiselect(
        "Second categorical variable(s):",
        second_options,
        default=second_options[:1],
        key="chi_square_second",
    )
    alpha = select_significance_level()
    if st.button("Run Chi-Square Test", type="primary", key="run_chi_square"):
        if not second_variables:
            st.error("Select at least one second categorical variable.")
            return
        summary_rows: list[dict[str, object]] = []
        detail_tables: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
        for second in second_variables:
            contingency = pd.crosstab(df[first], df[second])
            if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                st.warning(
                    f"'{second}': both variables must contain at least two observed "
                    "categories; skipped."
                )
                continue
            statistic, p_value, degrees_freedom, expected = stats.chi2_contingency(
                contingency
            )
            is_significant = np.isfinite(p_value) and p_value <= alpha
            summary_rows.append({
                "Variable": second,
                "χ² statistic": statistic,
                "df": degrees_freedom,
                "p-value": p_value,
                "Interpretation": (
                    f"Significantly associated with {first} ✅"
                    if is_significant
                    else f"No significant association with {first} ❌"
                ),
            })
            expected_table = pd.DataFrame(
                expected,
                index=contingency.index,
                columns=contingency.columns,
            ).round(2)
            detail_tables[second] = (contingency, expected_table)

        if not summary_rows:
            st.error("No valid chi-square results were produced.")
            return

        st.markdown("#### Summary")
        st.dataframe(
            with_formatted_p_values(pd.DataFrame(summary_rows), ["p-value"]),
            hide_index=True,
            width="stretch",
            column_config={
                "χ² statistic": st.column_config.NumberColumn(format="%.4f"),
            },
        )
        st.caption(f"Interpretation uses the selected significance level α = {format_alpha(alpha)}.")

        st.markdown(f"#### Observed / expected counts vs {first}")
        for index, (second, (observed, expected)) in enumerate(detail_tables.items()):
            _show_variable_section_header(second, index)
            st.markdown("**Observed counts**")
            st.dataframe(observed, use_container_width=True)
            st.markdown("**Expected counts**")
            st.dataframe(expected, use_container_width=True)


