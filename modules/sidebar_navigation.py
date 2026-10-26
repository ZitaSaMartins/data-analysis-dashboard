"""Hierarchical sidebar navigation and statistical-test guidance."""

from collections.abc import Iterable

import streamlit as st


PREPARE_PAGES = {"upload", "validate", "rename"}
EXPLORE_PAGES = {
    "eda_distributions",
    "eda_correlations",
    "eda_comparisons",
    "eda_hierarchical",
    "eda_temporal",
}
RELATIONSHIP_PAGES = {
    "test_distribution",
    "test_variance",
    "suggested_analysis",
    "parametric_ttest_one_sample",
    "parametric_ttest_independent",
    "parametric_ttest_paired",
    "parametric_anova",
    "parametric_twoway_anova",
    "parametric_repeated_anova",
    "nonparametric_mannwhitney",
    "nonparametric_wilcoxon",
    "nonparametric_kruskal_wallis",
    "nonparametric_friedman",
    "nonparametric_mood_median",
    "correlation_pearson",
    "correlation_spearman",
    "correlation_kendall",
    "regression",
    "chi_square",
}
SENSORY_PAGES = {
    "sensory_triangle",
    "sensory_paired",
    "sensory_duo_trio",
    "sensory_tetrad",
    "sensory_ranking",
    "sensory_qda",
    "sensory_temporal",
    "sensory_panel_discrimination",
    "sensory_panel_repeatability",
    "sensory_panel_agreement",
    "sensory_panel_scale",
    "sensory_product_characterization",
    "sensory_cata",
    "sensory_jar",
    "sensory_hedonic",
    "sensory_segmentation",
    "sensory_internal_preference",
    "sensory_external_preference",
    "sensory_pca",
    "sensory_gpa",
    "sensory_statis",
    "sensory_hac",
    "sensory_mfa",
    "sensory_pls",
}
ADVANCED_PAGES = {"machine_learning", "recommendations"}
HELP_PAGES = {
    "help_getting_started",
    "help_analysis_types",
    "help_choose_tests",
}


def _navigate(page: str) -> None:
    """Set the active page and immediately rerun the app."""
    st.session_state.page = page
    st.rerun()


def _nav_button(label: str, page: str, key: str) -> None:
    """Render a full-width navigation button with active-state styling."""
    is_active = st.session_state.get("page", "home") == page
    if st.button(
        label,
        key=key,
        width="stretch",
        type="primary" if is_active else "secondary",
    ):
        _navigate(page)


def _is_active(pages: Iterable[str]) -> bool:
    return st.session_state.get("page", "home") in pages


def build_main_navigation() -> None:
    """Render the hierarchical, collapsible application navigation."""
    with st.sidebar:
        st.markdown("## 📊 Navigation")
        _nav_button("🏠 Home", "home", "sidebar_home")

        with st.expander("📤 PREPARE DATA", expanded=_is_active(PREPARE_PAGES)):
            _nav_button("📤 Upload Data", "upload", "sidebar_upload")
            _nav_button("✅ Check Data Quality", "validate", "sidebar_validate")
            _nav_button("🏷️ Rename Variables", "rename", "sidebar_rename")

        with st.expander("🔍 EXPLORE DATA", expanded=_is_active(EXPLORE_PAGES)):
            _nav_button(
                "📊 Distributions",
                "eda_distributions",
                "sidebar_eda_distributions",
            )
            _nav_button(
                "🔗 Correlations",
                "eda_correlations",
                "sidebar_eda_correlations",
            )
            _nav_button(
                "🔀 Comparisons",
                "eda_comparisons",
                "sidebar_eda_comparisons",
            )
            _nav_button(
                "🌳 Part-to-Whole & Hierarchical",
                "eda_hierarchical",
                "sidebar_eda_hierarchical",
            )
            _nav_button(
                "⏱️ Temporal",
                "eda_temporal",
                "sidebar_eda_temporal",
            )

        with st.expander(
            "📊 ANALYZE RELATIONSHIPS",
            expanded=_is_active(RELATIONSHIP_PAGES),
        ):
            _nav_button(
                "Test Distribution",
                "test_distribution",
                "sidebar_test_distribution",
            )
            _nav_button(
                "Test Variance",
                "test_variance",
                "sidebar_test_variance",
            )
            _nav_button(
                "Suggested Analysis",
                "suggested_analysis",
                "sidebar_suggested_analysis",
            )

            with st.expander(
                "👥 Two Independent Groups",
                expanded=st.session_state.get("page") in {
                    "parametric_ttest_independent",
                    "nonparametric_mannwhitney",
                },
            ):
                st.markdown("**Parametric (assumes normal distribution)**")
                _nav_button(
                    "🟦 Independent t-test",
                    "parametric_ttest_independent",
                    "sidebar_independent_ttest",
                )
                st.markdown("**Non-Parametric (no normality assumption)**")
                _nav_button(
                    "🟧 Mann-Whitney U",
                    "nonparametric_mannwhitney",
                    "sidebar_mannwhitney",
                )

            with st.expander(
                "👥👥 3+ Groups (Single Factor)",
                expanded=st.session_state.get("page") in {
                    "parametric_anova",
                    "nonparametric_kruskal_wallis",
                    "nonparametric_mood_median",
                },
            ):
                st.markdown("**Parametric**")
                _nav_button(
                    "🟦 One-Way ANOVA",
                    "parametric_anova",
                    "sidebar_anova",
                )
                st.markdown("**Non-Parametric**")
                _nav_button(
                    "🟧 Kruskal-Wallis",
                    "nonparametric_kruskal_wallis",
                    "sidebar_kruskal",
                )
                _nav_button(
                    "🟧 Mood's Median Test",
                    "nonparametric_mood_median",
                    "sidebar_mood_median",
                )

            with st.expander(
                "📊 Multiple Factors",
                expanded=st.session_state.get("page") == "parametric_twoway_anova",
            ):
                st.markdown("**Parametric factorial designs**")
                _nav_button(
                    "🟦 Two-Way ANOVA",
                    "parametric_twoway_anova",
                    "sidebar_two_way_anova",
                )

            with st.expander(
                "🔁 Repeated Measures",
                expanded=st.session_state.get("page") in {
                    "parametric_ttest_paired",
                    "parametric_repeated_anova",
                    "nonparametric_wilcoxon",
                    "nonparametric_friedman",
                },
            ):
                st.markdown("**Two related measurements**")
                _nav_button(
                    "🟦 Paired t-test",
                    "parametric_ttest_paired",
                    "sidebar_paired_ttest",
                )
                _nav_button(
                    "🟧 Wilcoxon Signed-Rank",
                    "nonparametric_wilcoxon",
                    "sidebar_wilcoxon",
                )
                st.markdown("**Three or more related measurements**")
                _nav_button(
                    "🟦 Repeated Measures ANOVA",
                    "parametric_repeated_anova",
                    "sidebar_repeated_anova",
                )
                _nav_button(
                    "🟧 Friedman",
                    "nonparametric_friedman",
                    "sidebar_friedman",
                )

            with st.expander(
                "🔗 Correlations",
                expanded=st.session_state.get("page") in {
                    "correlation_pearson",
                    "correlation_spearman",
                    "correlation_kendall",
                },
            ):
                st.markdown("**Linear relationships**")
                _nav_button(
                    "🟦 Pearson",
                    "correlation_pearson",
                    "sidebar_pearson",
                )
                st.markdown("**Rank-based (Non-parametric)**")
                _nav_button(
                    "🟧 Spearman",
                    "correlation_spearman",
                    "sidebar_spearman",
                )
                _nav_button(
                    "🟧 Kendall's Tau",
                    "correlation_kendall",
                    "sidebar_kendall",
                )

            with st.expander(
                "📈 Regression",
                expanded=st.session_state.get("page") == "regression",
            ):
                _nav_button(
                    "Linear Regression",
                    "regression",
                    "sidebar_regression",
                )

            with st.expander(
                "📋 Categorical",
                expanded=st.session_state.get("page") == "chi_square",
            ):
                _nav_button(
                    "Chi-Square",
                    "chi_square",
                    "sidebar_chi_square",
                )

        with st.expander(
            "👃 SENSORY ANALYSIS",
            expanded=_is_active(SENSORY_PAGES),
        ):
            with st.expander(
                "🎯 Discrimination Tests (ISO)",
                expanded=st.session_state.get("page") in {
                    "sensory_triangle",
                    "sensory_paired",
                    "sensory_duo_trio",
                    "sensory_tetrad",
                    "sensory_ranking",
                },
            ):
                _nav_button("Triangle Test", "sensory_triangle", "sidebar_sensory_triangle")
                _nav_button(
                    "Paired Comparison",
                    "sensory_paired",
                    "sidebar_sensory_paired",
                )
                _nav_button("Duo-Trio Test", "sensory_duo_trio", "sidebar_sensory_duo")
                _nav_button("Tetrad Test", "sensory_tetrad", "sidebar_sensory_tetrad")
                _nav_button("Ranking Test", "sensory_ranking", "sidebar_sensory_ranking")

            with st.expander(
                "📊 Descriptive Analysis",
                expanded=st.session_state.get("page") in {
                    "sensory_qda",
                    "sensory_temporal",
                },
            ):
                _nav_button("QDA / Sensory Profile", "sensory_qda", "sidebar_sensory_qda")
                _nav_button(
                    "Temporal Methods",
                    "sensory_temporal",
                    "sidebar_sensory_temporal",
                )

            with st.expander(
                "🔍 Panel Analysis",
                expanded=st.session_state.get("page") in {
                    "sensory_panel_discrimination",
                    "sensory_panel_repeatability",
                    "sensory_panel_agreement",
                    "sensory_panel_scale",
                },
            ):
                _nav_button(
                    "Discrimination",
                    "sensory_panel_discrimination",
                    "sidebar_sensory_panel_discrimination",
                )
                _nav_button(
                    "Repeatability",
                    "sensory_panel_repeatability",
                    "sidebar_sensory_panel_repeatability",
                )
                _nav_button(
                    "Agreement & Consensus",
                    "sensory_panel_agreement",
                    "sidebar_sensory_panel_agreement",
                )
                _nav_button(
                    "Scale Use",
                    "sensory_panel_scale",
                    "sidebar_sensory_panel_scale",
                )

            with st.expander(
                "📈 Product Characterization",
                expanded=st.session_state.get("page")
                == "sensory_product_characterization",
            ):
                _nav_button(
                    "Product Profiles",
                    "sensory_product_characterization",
                    "sidebar_sensory_products",
                )

            with st.expander(
                "✅ CATA Analysis",
                expanded=st.session_state.get("page") == "sensory_cata",
            ):
                _nav_button(
                    "CATA Frequencies & Map",
                    "sensory_cata",
                    "sidebar_sensory_cata",
                )

            with st.expander(
                "📏 JAR Analysis",
                expanded=st.session_state.get("page") == "sensory_jar",
            ):
                _nav_button(
                    "JAR & Penalty Analysis",
                    "sensory_jar",
                    "sidebar_sensory_jar",
                )

            with st.expander(
                "❤️ Consumer Preference",
                expanded=st.session_state.get("page") in {
                    "sensory_hedonic",
                    "sensory_segmentation",
                    "sensory_internal_preference",
                    "sensory_external_preference",
                },
            ):
                _nav_button(
                    "Hedonic Analysis",
                    "sensory_hedonic",
                    "sidebar_sensory_hedonic",
                )
                _nav_button(
                    "Consumer Segmentation",
                    "sensory_segmentation",
                    "sidebar_sensory_segmentation",
                )
                _nav_button(
                    "Internal Preference Map",
                    "sensory_internal_preference",
                    "sidebar_sensory_internal_preference",
                )
                _nav_button(
                    "External Preference Map",
                    "sensory_external_preference",
                    "sidebar_sensory_external_preference",
                )

            with st.expander(
                "🎨 Advanced Multivariate",
                expanded=st.session_state.get("page") in {
                    "sensory_pca",
                    "sensory_gpa",
                    "sensory_statis",
                    "sensory_hac",
                    "sensory_mfa",
                    "sensory_pls",
                },
            ):
                _nav_button("PCA", "sensory_pca", "sidebar_sensory_pca")
                _nav_button("GPA", "sensory_gpa", "sidebar_sensory_gpa")
                _nav_button("STATIS", "sensory_statis", "sidebar_sensory_statis")
                _nav_button("Hierarchical Clustering", "sensory_hac", "sidebar_sensory_hac")
                _nav_button("MFA", "sensory_mfa", "sidebar_sensory_mfa")
                _nav_button("PLS Regression", "sensory_pls", "sidebar_sensory_pls")

        with st.expander(
            "🤖 ADVANCED ANALYSIS",
            expanded=_is_active(ADVANCED_PAGES),
        ):
            _nav_button(
                "🤖 Build Predictions",
                "machine_learning",
                "sidebar_machine_learning",
            )
            _nav_button(
                "💡 Get Recommendations",
                "recommendations",
                "sidebar_recommendations",
            )

        with st.expander("❓ HELP", expanded=_is_active(HELP_PAGES)):
            _nav_button(
                "Getting Started",
                "help_getting_started",
                "sidebar_help_getting_started",
            )
            _nav_button(
                "Analysis Types",
                "help_analysis_types",
                "sidebar_help_analysis_types",
            )
            _nav_button(
                "How to Choose Tests",
                "help_choose_tests",
                "sidebar_help_choose_tests",
            )


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


def show_sidebar_status() -> None:
    """Display the current dataset status at the bottom of the sidebar."""
    with st.sidebar:
        st.markdown("---")
        df = st.session_state.get("df")
        if df is not None:
            filename = st.session_state.get("uploaded_filename") or "Not available"
            st.success(
                f"✅ **File name:** {filename}\n\n"
                f"**Data loaded:** {df.shape[0]} rows × {df.shape[1]} columns"
            )
        else:
            st.info("👉 Upload data to get started")
