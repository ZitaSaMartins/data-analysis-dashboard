"""Interactive decision-tree guide for choosing a statistical test."""

import streamlit as st

from ._shared import _navigate


def _decision_option(
    column,
    title: str,
    description: str,
    page: str | None = None,
    key: str | None = None,
) -> None:
    with column:
        st.markdown(f"### {title}")
        st.write(description)
        if page and key and st.button("Open this test", key=key, width="stretch"):
            _navigate(page)


def show_test_decision_tree() -> None:
    """Display an interactive guide for choosing a statistical test."""
    st.markdown("## 🧭 How to Choose a Statistical Test")
    st.caption(
        "Choose your study structure first, then compare the parametric and "
        "non-parametric options."
    )
    st.info(
        "Recommended workflow: select the same variables in Test Distribution and "
        "Test Variance, run both checks, then review Suggested Analysis. The variable "
        "selection is shared automatically."
    )

    with st.expander("Before choosing a test", expanded=True):
        st.markdown(
            """
            1. Identify whether the outcome is **numeric** or **categorical**.
            2. Count the groups and decide whether observations are
               **independent** or **related/repeated**.
            3. For numeric group comparisons, review distribution, variance,
               outliers, and sample sizes.
            4. Use **Suggested Analysis** for assumption-based guidance, but confirm
               that its independent-group default matches your study design.
            """
        )

    with st.expander("How should I interpret the result?"):
        st.markdown(
            """
            - A **p-value ≤ 0.05** is commonly treated as evidence against the null
              hypothesis; it is not the probability that the hypothesis is true.
            - A **p-value > 0.05** means the analysis did not detect a statistically
              significant effect; it does not prove that no effect exists.
            - Review **effect sizes**, confidence intervals when available, plots,
              data quality, and practical importance alongside the p-value.
            - Statistical recommendations support, but do not replace, a valid
              sampling plan and study design.
            """
        )

    group_count = st.radio(
        "How many groups?",
        ["1 group", "2 groups", "3+ groups"],
        horizontal=True,
        key="decision_group_count",
    )

    if group_count == "1 group":
        parametric, nonparametric = st.columns(2)
        _decision_option(
            parametric,
            "🟦 One-sample t-test",
            "Use for a numeric sample compared with a reference value when the "
            "sample is approximately normal.",
            "parametric_ttest_one_sample",
            "decision_one_sample_t",
        )
        _decision_option(
            nonparametric,
            "🟧 Wilcoxon signed-rank",
            "Use when a numeric sample is not normal or contains influential outliers.",
            "nonparametric_wilcoxon",
            "decision_one_sample_wilcoxon",
        )
        return

    relationship = st.radio(
        "Are the groups independent or related?",
        ["Independent", "Related / repeated measures"],
        horizontal=True,
        key=f"decision_relationship_{group_count}",
    )

    parametric, nonparametric = st.columns(2)
    if group_count == "2 groups" and relationship == "Independent":
        _decision_option(
            parametric,
            "🟦 Independent t-test",
            "Compares means from two separate groups.",
            "parametric_ttest_independent",
            "decision_independent_t",
        )
        _decision_option(
            nonparametric,
            "🟧 Mann-Whitney U",
            "Compares two separate groups using ranks.",
            "nonparametric_mannwhitney",
            "decision_mannwhitney",
        )
    elif group_count == "2 groups":
        _decision_option(
            parametric,
            "🟦 Paired t-test",
            "Compares two related measurements when their differences are normal.",
            "parametric_ttest_paired",
            "decision_paired_t",
        )
        _decision_option(
            nonparametric,
            "🟧 Wilcoxon signed-rank",
            "Compares two related measurements using ranks.",
            "nonparametric_wilcoxon",
            "decision_wilcoxon",
        )
    elif relationship == "Independent":
        _decision_option(
            parametric,
            "🟦 One-Way ANOVA",
            "Compares means from three or more separate groups.",
            "parametric_anova",
            "decision_anova",
        )
        _decision_option(
            nonparametric,
            "🟧 Kruskal-Wallis",
            "Compares three or more separate groups using ranks.",
            "nonparametric_kruskal",
            "decision_kruskal",
        )
    else:
        _decision_option(
            parametric,
            "🟦 Repeated Measures ANOVA",
            "Compares three or more related measurements.",
            "parametric_repeated_anova",
            "decision_repeated_anova",
        )
        _decision_option(
            nonparametric,
            "🟧 Friedman",
            "Compares three or more related measurements using ranks.",
            "nonparametric_friedman",
            "decision_friedman",
        )

