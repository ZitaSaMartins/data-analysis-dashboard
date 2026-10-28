"""Combines distribution and variance results into inference-test recommendations."""

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

from ._shared import ALPHA


def _inference_recommendations(
    is_normal: bool,
    has_equal_variance: bool,
    group_count: int,
) -> dict[str, str]:
    if not is_normal:
        return {
            "family": "Non-parametric",
            "primary_test": (
                "Mann-Whitney U" if group_count == 2 else "Kruskal-Wallis"
            ),
            "primary_page": (
                "nonparametric_mannwhitney"
                if group_count == 2
                else "nonparametric_kruskal_wallis"
            ),
            "primary_reason": (
                "The distribution result indicates that the normality assumption "
                "is not supported."
            ),
            "related_test": (
                "Wilcoxon signed-rank" if group_count == 2 else "Friedman"
            ),
            "relationship_test": "Spearman correlation or Kendall's tau",
            "one_sample_test": "One-sample Wilcoxon signed-rank",
        }

    if not has_equal_variance and group_count == 2:
        return {
            "family": "Parametric with unequal-variance adjustment",
            "primary_test": "Welch independent t-test",
            "primary_page": "parametric_ttest_independent",
            "primary_reason": (
                "Normality is supported, but group variances differ. Open the "
                "independent t-test and turn off 'Assume equal variances'."
            ),
            "related_test": "Paired t-test, if observations are matched",
            "relationship_test": "Pearson correlation",
            "one_sample_test": "One-sample t-test",
        }

    if not has_equal_variance:
        return {
            "family": "Robust non-parametric",
            "primary_test": "Kruskal-Wallis",
            "primary_page": "nonparametric_kruskal_wallis",
            "primary_reason": (
                "Normality is supported, but variances differ across three or more "
                "groups. Kruskal-Wallis is the robust option available in this dashboard."
            ),
            "related_test": "Repeated-measures ANOVA, if observations are matched",
            "relationship_test": "Pearson correlation",
            "one_sample_test": "One-sample t-test",
        }

    return {
        "family": "Parametric",
        "primary_test": (
            "Independent t-test" if group_count == 2 else "One-way ANOVA"
        ),
        "primary_page": (
            "parametric_ttest_independent"
            if group_count == 2
            else "parametric_anova"
        ),
        "primary_reason": (
            "Both normality and equal-variance assumptions are supported by the "
            "latest checks."
        ),
        "related_test": (
            "Paired t-test" if group_count == 2 else "Repeated-measures ANOVA"
        ),
        "relationship_test": "Pearson correlation",
        "one_sample_test": "One-sample t-test",
    }


def _prepare_recommended_test(
    page: str,
    variable: str,
    grouping_variable: str,
    assume_equal_variance: bool = True,
) -> None:
    test_widget_keys = {
        "parametric_ttest_independent": (
            "independent_t_numeric",
            "independent_t_grouping",
        ),
        "parametric_anova": ("anova_numeric_variables", "anova_group"),
        "nonparametric_mannwhitney": (
            "mannwhitney_numeric",
            "mannwhitney_grouping",
        ),
        "nonparametric_kruskal_wallis": (
            "kruskal_numeric_variables",
            "kruskal_group",
        ),
    }
    numeric_key, grouping_key = test_widget_keys[page]
    if page in {"parametric_anova", "nonparametric_kruskal_wallis"}:
        st.session_state[numeric_key] = [variable]
    else:
        st.session_state[numeric_key] = variable
    st.session_state[grouping_key] = grouping_variable
    if page == "parametric_ttest_independent":
        st.session_state.independent_t_equal_variance = assume_equal_variance


def _within_group_normality(
    df: pd.DataFrame,
    variable: str,
    grouping_variable: str,
) -> tuple[bool | None, str]:
    """Check normality separately within each comparison group.

    Independent t-tests and ANOVA require normality within each group (or of
    the model residuals), not in the pooled outcome: normal groups with
    separated means can combine into a non-normal pooled mixture. Returns
    ``None`` when at least one group cannot be tested, together with an
    explanatory reason.
    """
    group_reports: list[str] = []
    all_normal = True
    for group_name, group in df.groupby(grouping_variable, observed=True):
        values = group[variable].dropna().astype(float)
        if len(values) < 3:
            return None, (
                f"group '{group_name}' has fewer than three observations and "
                "cannot be tested for within-group normality."
            )
        if values.std(ddof=1) == 0:
            return None, (
                f"group '{group_name}' is constant and cannot be tested for "
                "within-group normality."
            )
        if len(values) < 50:
            _, p_value = stats.shapiro(values)
            test_name = "Shapiro-Wilk"
        else:
            _, p_value = stats.normaltest(values)
            test_name = "D'Agostino-Pearson K²"
        supports_normality = p_value > ALPHA
        if not supports_normality:
            all_normal = False
        group_reports.append(
            f"'{group_name}': {test_name} p={p_value:.4f} "
            f"({'normal' if supports_normality else 'not normal'})"
        )
    return all_normal, "Within-group normality — " + "; ".join(group_reports) + "."


def show_suggested_analysis(df: pd.DataFrame) -> None:
    """Recommend inference tests from the latest assumption-check results."""
    st.markdown("## 🧭 Suggested Analysis")
    st.write(
        "Combines the latest distribution and variance results to suggest inference "
        "tests that can be carried out in this dashboard."
    )

    distribution_result = st.session_state.get("distribution_test_result") or {}
    variance_result = st.session_state.get("variance_test_result") or {}
    distribution_by_variable = {
        result["variable"]: result
        for result in distribution_result.get("variables", [])
    }
    variance_by_variable = {
        result["variable"]: result
        for result in variance_result.get("variables", [])
    }
    assumption_variables = [
        variable
        for variable in st.session_state.get("assumption_variables", [])
        if variable in df.columns
    ]
    if not assumption_variables:
        assumption_variables = list(dict.fromkeys([
            *distribution_by_variable,
            *variance_by_variable,
        ]))
        assumption_variables = [
            variable for variable in assumption_variables if variable in df.columns
        ]
    if not assumption_variables:
        st.info(
            "Select numeric variables in Test Distribution or Test Variance first. "
            "The same selection will be used by both pages and by Suggested Analysis."
        )
        return

    current_selection = st.session_state.get("_suggested_analysis_variables")
    if current_selection is None:
        st.session_state._suggested_analysis_variables = assumption_variables
    else:
        st.session_state._suggested_analysis_variables = [
            variable
            for variable in current_selection
            if variable in assumption_variables
        ]
    selected_variables = st.multiselect(
        "Variables to include:",
        assumption_variables,
        key="_suggested_analysis_variables",
    )
    if not selected_variables:
        st.info("Select at least one variable to review.")
        return

    missing_distribution = [
        variable
        for variable in selected_variables
        if variable not in distribution_by_variable
    ]
    missing_variance = [
        variable
        for variable in selected_variables
        if variable not in variance_by_variable
    ]
    if missing_distribution or missing_variance:
        st.warning(
            "A complete recommendation requires both assumption checks for every "
            "selected variable."
        )
        action_columns = st.columns(2)
        with action_columns[0]:
            if missing_distribution:
                st.error(
                    "**Distribution analysis missing:** "
                    + ", ".join(missing_distribution)
                )
                if st.button(
                    "Complete Test Distribution",
                    key="suggested_open_distribution",
                    width="stretch",
                ):
                    st.session_state.page = "test_distribution"
                    st.rerun()
            else:
                st.success("Distribution analysis is complete.")
        with action_columns[1]:
            if missing_variance:
                st.error(
                    "**Variance analysis missing:** "
                    + ", ".join(missing_variance)
                )
                if st.button(
                    "Complete Test Variance",
                    key="suggested_open_variance",
                    width="stretch",
                ):
                    st.session_state.page = "test_variance"
                    st.rerun()
            else:
                st.success("Variance analysis is complete.")

    grouping_variable = variance_result.get(
        "grouping_variable",
        "the selected grouping variable",
    )
    summary_rows = []
    recommendations_by_variable: dict[str, dict[str, str]] = {}
    normality_by_variable: dict[str, tuple[bool, str]] = {}
    for variable in selected_variables:
        distribution = distribution_by_variable.get(variable)
        variance = variance_by_variable.get(variable)
        if distribution is None or variance is None:
            summary_rows.append({
                "Variable": variable,
                "Distribution": (
                    "Missing" if distribution is None else "Complete"
                ),
                "Variance": "Missing" if variance is None else "Complete",
                "Groups": str(variance["group_count"]) if variance else "—",
                "Suggested test": "Complete missing checks",
            })
            continue

        is_normal, normality_reason = _within_group_normality(
            df, variable, grouping_variable
        )
        if is_normal is None:
            summary_rows.append({
                "Variable": variable,
                "Distribution": "Unavailable",
                "Variance": (
                    "Equal variance supported"
                    if variance["has_equal_variance"]
                    else "Unequal variance"
                ),
                "Groups": str(variance["group_count"]),
                "Suggested test": "Within-group normality unavailable",
            })
            continue
        normality_by_variable[variable] = (is_normal, normality_reason)

        recommendation = _inference_recommendations(
            is_normal,
            variance["has_equal_variance"],
            variance["group_count"],
        )
        recommendations_by_variable[variable] = recommendation
        summary_rows.append({
            "Variable": variable,
            "Distribution": (
                "Normality supported"
                if is_normal
                else "Normality not supported"
            ),
            "Variance": (
                "Equal variance supported"
                if variance["has_equal_variance"]
                else "Unequal variance"
            ),
            "Groups": str(variance["group_count"]),
            "Suggested test": recommendation["primary_test"],
        })

    st.markdown(f"#### Recommendation summary by `{grouping_variable}`")
    st.dataframe(
        pd.DataFrame(summary_rows),
        hide_index=True,
        width="stretch",
    )

    for variable in selected_variables:
        distribution = distribution_by_variable.get(variable)
        variance = variance_by_variable.get(variable)
        normality = normality_by_variable.get(variable)
        with st.container(border=True):
            st.markdown(f"### {variable}")
            distribution_column, variance_column = st.columns(2)
            with distribution_column:
                if distribution is None or variance is None:
                    st.error("Distribution analysis: missing")
                elif normality is None:
                    st.error("Within-group normality: could not be determined")
                elif normality[0]:
                    st.success("Distribution: normality supported")
                else:
                    st.warning("Distribution: normality not supported")
                if normality is not None:
                    st.caption(normality[1])
            with variance_column:
                if variance is None:
                    st.error("Variance analysis: missing")
                elif variance["has_equal_variance"]:
                    st.success("Variance: no significant difference detected")
                else:
                    st.warning("Variance: significant difference detected")
                if variance is not None:
                    st.caption(variance["reason"])

            if distribution is None or variance is None or normality is None:
                st.info(
                    "Complete the missing assumption analysis before selecting an "
                    "inference test for this variable."
                )
                continue

            recommendation = recommendations_by_variable[variable]
            st.markdown(
                f"**Primary recommendation: {recommendation['primary_test']}**  \n"
                f"Path: **{recommendation['family']}**  \n"
                f"{recommendation['primary_reason']}"
            )
            st.markdown(
                "**Other study designs:**\n"
                f"- Related or repeated observations: "
                f"**{recommendation['related_test']}**\n"
                f"- Association between numeric variables: "
                f"**{recommendation['relationship_test']}**\n"
                f"- One sample versus a reference value: "
                f"**{recommendation['one_sample_test']}**"
            )
            if st.button(
                f"Open {recommendation['primary_test']}",
                key=f"suggested_open_{variable}",
                type="primary",
                width="stretch",
            ):
                _prepare_recommended_test(
                    recommendation["primary_page"],
                    variable,
                    grouping_variable,
                    recommendation["primary_test"] != "Welch independent t-test",
                )
                st.session_state.page = recommendation["primary_page"]
                st.rerun()

    st.caption(
        "Recommendations assume independent groups for the primary test. Confirm "
        "your study design, observation independence, pairing, outliers, and sample "
        "size before drawing conclusions."
    )

