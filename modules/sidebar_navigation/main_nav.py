"""Builds the hierarchical, collapsible sidebar navigation."""

import streamlit as st

from ._shared import _is_active, _nav_button
from .page_sets import (
    ADVANCED_PAGES,
    EXPLORE_PAGES,
    HELP_PAGES,
    PREPARE_PAGES,
    RELATIONSHIP_PAGES,
    SENSORY_PAGES,
)


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

