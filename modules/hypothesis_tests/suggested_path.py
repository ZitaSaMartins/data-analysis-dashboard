"""Legacy "Suggested Path" page: recommends a testing family from assumption checks."""

import pandas as pd
import streamlit as st

from ._navigation import _open_hypothesis_category


def run_suggested_path(df: pd.DataFrame):
    """Recommend a group-comparison path from the latest assumption checks."""
    st.markdown("### 🧭 Suggested Path")
    st.caption(
        "Combines your latest distribution and variance results to suggest a testing "
        "family for comparing independent groups."
    )

    distribution_result = st.session_state.get("distribution_test_result")
    variance_result = st.session_state.get("variance_test_result")

    if distribution_result is None or variance_result is None:
        st.info(
            "Complete both assumption checks before requesting a recommendation. "
            "The results are saved automatically when each test finishes."
        )
        if distribution_result is None:
            st.warning("Distribution result missing — run **Test Distribution**.")
            if st.button("Go to Test Distribution", key="suggest_go_distribution"):
                _open_hypothesis_category("Test Distribution")
        if variance_result is None:
            st.warning("Variance result missing — run **Test Variance**.")
            if st.button("Go to Test Variance", key="suggest_go_variance"):
                _open_hypothesis_category("Test Variance")
        return

    distribution_by_variable = {
        result["variable"]: result
        for result in distribution_result["variables"]
    }
    variance_by_variable = {
        result["variable"]: result
        for result in variance_result["variables"]
    }
    matching_variables = [
        variable
        for variable in variance_by_variable
        if variable in distribution_by_variable
    ]
    unmatched_variables = [
        variable
        for variable in variance_by_variable
        if variable not in distribution_by_variable
    ]

    if unmatched_variables:
        st.warning(
            "The following variance-tested variables do not have a matching distribution "
            f"result: **{', '.join(unmatched_variables)}**."
        )
        if st.button("Test the missing distributions", key="suggest_match_distribution"):
            st.session_state.dist_vars = unmatched_variables
            _open_hypothesis_category("Test Distribution")

    if not matching_variables:
        return

    for numeric_var in matching_variables:
        matching_distribution = distribution_by_variable[numeric_var]
        matching_variance = variance_by_variable[numeric_var]
        is_normal = matching_distribution["is_normal"]
        has_equal_variance = matching_variance["has_equal_variance"]

        st.markdown(f"#### {numeric_var}")
        col1, col2 = st.columns(2)
        with col1:
            if is_normal:
                st.success("**Distribution:** consistent with normality")
            else:
                st.warning("**Distribution:** not normally distributed")
            st.caption(matching_distribution["reason"])
        with col2:
            if has_equal_variance:
                st.success("**Variance:** no significant difference detected")
            else:
                st.warning("**Variance:** significantly different between groups")
            st.caption(matching_variance["reason"])

        reasons = []
        if not is_normal:
            reasons.append(
                "the distribution check indicates that the normality assumption is not met"
            )
        if not has_equal_variance:
            reasons.append(
                f"the {matching_variance['test_name']} result indicates unequal variance "
                "between groups"
            )

        group_count = matching_variance["group_count"]
        if reasons:
            st.warning("**Recommended path: Non-parametric Tests**")
            st.markdown(
                "A non-parametric route is the safer choice because "
                + " and ".join(reasons)
                + "."
            )
            if group_count == 2:
                st.markdown(
                    "- For **two independent groups**, consider the **Mann-Whitney U Test**.\n"
                    "- For **paired measurements**, consider the **Wilcoxon Signed-Rank Test**."
                )
            else:
                st.markdown(
                    f"- For the **{group_count} independent groups** in the variance check, "
                    "consider the **Kruskal-Wallis Test**."
                )
        else:
            st.success("**Recommended path: Parametric Tests**")
            st.markdown(
                "The distribution check did not find significant evidence against normality, "
                "and the variance check did not find a significant difference between group "
                "variances. These results support the assumptions used by the standard "
                "parametric group-comparison tests in this dashboard."
            )
            if group_count == 2:
                st.markdown(
                    "- For **two independent groups**, consider the **Independent Samples t-Test**.\n"
                    "- For **paired measurements**, consider the **Paired Samples t-Test**."
                )
            else:
                st.markdown(
                    f"- For the **{group_count} independent groups** in the variance check, "
                    "consider **One-Way ANOVA**."
                )
        st.markdown("---")

    nonparametric_recommended = any(
        not distribution_by_variable[variable]["is_normal"]
        or not variance_by_variable[variable]["has_equal_variance"]
        for variable in matching_variables
    )
    if nonparametric_recommended:
        if st.button("Open Non-parametric Tests", type="primary", key="suggest_nonparametric"):
            group_count = variance_by_variable[matching_variables[0]]["group_count"]
            selected_test = (
                "Mann-Whitney U Test" if group_count == 2 else "Kruskal-Wallis Test"
            )
            _open_hypothesis_category("Non-parametric Tests", selected_test)
    else:
        if st.button("Open Parametric Tests", type="primary", key="suggest_parametric"):
            group_count = variance_by_variable[matching_variables[0]]["group_count"]
            selected_test = (
                "Independent Samples t-Test" if group_count == 2 else "One-Way ANOVA"
            )
            _open_hypothesis_category("Parametric Tests", selected_test)

    st.caption(
        "This guidance supports test selection but does not replace checking the study design, "
        "independence, outliers, sample size, or whether observations are paired."
    )

