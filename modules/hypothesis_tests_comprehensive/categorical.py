"""Categorical association tests."""

import pandas as pd
import streamlit as st
from scipy import stats

from ._shared import _require_categorical, _show_inference


def test_chi_square(df: pd.DataFrame) -> None:
    """Run a chi-square test of independence for two categorical variables."""
    st.markdown("## 📋 Chi-Square Test")
    st.write("Tests whether two categorical variables are associated.")
    categorical_columns = _require_categorical(df, minimum=2)
    if categorical_columns is None:
        return
    col1, col2 = st.columns(2)
    first = col1.selectbox(
        "First categorical variable:",
        categorical_columns,
        key="chi_square_first",
    )
    second_options = [column for column in categorical_columns if column != first]
    second = col2.selectbox(
        "Second categorical variable:",
        second_options,
        key="chi_square_second",
    )
    if st.button("Run Chi-Square Test", type="primary", key="run_chi_square"):
        contingency = pd.crosstab(df[first], df[second])
        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            st.error("Both variables must contain at least two observed categories.")
            return
        statistic, p_value, degrees_freedom, expected = stats.chi2_contingency(
            contingency
        )
        statistic_col, p_value_col, df_col = st.columns(3)
        statistic_col.metric("χ² statistic", f"{statistic:.4f}")
        p_value_col.metric("p-value", f"{p_value:.4f}")
        df_col.metric("Degrees of freedom", str(degrees_freedom))
        st.markdown("**Observed counts**")
        st.dataframe(contingency, use_container_width=True)
        st.markdown("**Expected counts**")
        st.dataframe(
            pd.DataFrame(
                expected,
                index=contingency.index,
                columns=contingency.columns,
            ).round(2),
            use_container_width=True,
        )
        _show_inference(
            p_value,
            "The categorical variables are significantly associated",
            "No significant categorical association was detected",
        )

