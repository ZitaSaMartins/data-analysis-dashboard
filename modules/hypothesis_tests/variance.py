"""Legacy "Test Variance" (homogeneity of variance) page."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats


def run_test_variance(df: pd.DataFrame):
    """Homogeneity of variance testing across groups."""
    st.markdown("### 📏 Test Variance (Homogeneity)")
    st.caption("Checks whether two or more groups have equal variance — an assumption behind "
               "the standard t-test and ANOVA.")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    if not numeric_cols or not cat_cols:
        st.error("Need both a numeric and a categorical column.")
        return

    selected_cols = st.multiselect(
        "Select numeric variable(s) to test:",
        numeric_cols,
        default=numeric_cols[:min(3, len(numeric_cols))],
        key="var_numeric_vars",
    )
    grouping_var = st.selectbox("Grouping variable:", cat_cols, key="var_cat")

    test_options = {
        "Levene's test — robust, recommended default": "levene",
        "Bartlett's test — more powerful, but assumes normal data": "bartlett",
        "Fligner-Killeen test — most robust to outliers/non-normality": "fligner",
    }
    test_label = st.selectbox("Variance test:", list(test_options.keys()), key="var_test_choice")
    test_choice = test_options[test_label]

    with st.expander("ℹ️ Which test should I use?"):
        st.markdown("""
        - **Levene's test**: the standard, robust choice — works well even if data isn't
          perfectly normal. Good default.
        - **Bartlett's test**: more statistical power, but only reliable when the data is
          (close to) normally distributed. Check *Test Distribution* first.
        - **Fligner-Killeen test**: non-parametric and very robust to outliers and non-normal
          data — use when you suspect normality is violated.
        """)

    if not selected_cols:
        return

    if st.button("Run Variance Test", type="primary", key="run_variance"):
        st.session_state.variance_test_result = None
        all_groups = [g for g in df[grouping_var].unique() if pd.notna(g)]
        if len(all_groups) < 2:
            st.error("Need at least 2 groups.")
            return

        alpha = 0.05
        results = []
        saved_results = []
        group_summaries = []

        for numeric_var in selected_cols:
            group_data = [
                df[df[grouping_var] == group][numeric_var].dropna().values
                for group in all_groups
            ]
            valid = [
                (group, data)
                for group, data in zip(all_groups, group_data)
                if len(data) >= 2
            ]
            if len(valid) < 2:
                st.warning(
                    f"'{numeric_var}' needs at least two groups with two observations each — skipped."
                )
                continue
            groups, valid_group_data = zip(*valid)

            if test_choice == "levene":
                stat, p_value = stats.levene(*valid_group_data)
                test_name = "Levene's test"
            elif test_choice == "bartlett":
                stat, p_value = stats.bartlett(*valid_group_data)
                test_name = "Bartlett's test"
            else:
                stat, p_value = stats.fligner(*valid_group_data)
                test_name = "Fligner-Killeen test"

            if not np.isfinite(stat) or not np.isfinite(p_value):
                st.warning(
                    f"'{numeric_var}' produced an invalid variance result. Check for "
                    "constant values or groups with too little variation."
                )
                continue

            has_equal_variance = p_value >= alpha
            results.append({
                "Variable": numeric_var,
                "Test": test_name,
                "Statistic": round(stat, 4),
                "p-value": round(p_value, 4),
                "Equal variance (α=0.05)": "Yes" if has_equal_variance else "No",
            })
            saved_results.append({
                "variable": numeric_var,
                "group_count": len(groups),
                "has_equal_variance": has_equal_variance,
                "test_name": test_name,
                "p_value": p_value,
                "reason": (
                    f"{test_name}: p = {p_value:.4f} "
                    f"({'≥' if has_equal_variance else '<'} {alpha})."
                ),
            })
            group_summaries.extend({
                "Variable": numeric_var,
                "Group": group,
                "n": len(data),
                "Variance": np.var(data, ddof=1),
                "Std Dev": np.std(data, ddof=1),
            } for group, data in zip(groups, valid_group_data))

        if not results:
            return

        st.session_state.variance_test_result = {
            "variables": saved_results,
            "grouping_variable": grouping_var,
            "alpha": alpha,
        }
        st.dataframe(pd.DataFrame(results), use_container_width=True)

        unequal_variables = [
            result["variable"]
            for result in saved_results
            if not result["has_equal_variance"]
        ]
        if unequal_variables:
            st.warning(
                "⚠️ **Significantly different variances:** "
                f"{', '.join(unequal_variables)}. Consider Welch-adjusted or "
                "non-parametric alternatives for these variables."
            )
        equal_variables = [
            result["variable"]
            for result in saved_results
            if result["has_equal_variance"]
        ]
        if equal_variables:
            st.success(
                "✅ **No significant variance difference detected:** "
                f"{', '.join(equal_variables)}."
            )

        st.markdown("#### Group variance details")
        st.dataframe(pd.DataFrame(group_summaries), use_container_width=True)

