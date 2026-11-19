import streamlit as st

from app_pages.state import set_page
from modules.hypothesis_tests_comprehensive import (
    run_distribution_tests,
    run_variance_tests,
    show_suggested_analysis,
    test_anova,
    test_chi_square,
    test_friedman,
    test_independent_ttest,
    test_kendall,
    test_kruskal_wallis,
    test_linear_regression,
    test_mannwhitney,
    test_mood_median,
    test_one_sample_ttest,
    test_paired_ttest,
    test_pearson_correlation,
    test_repeated_anova,
    test_spearman,
    test_two_way_anova,
    test_wilcoxon,
)

TEST_PAGE_HANDLERS = {
    "test_distribution": run_distribution_tests,
    "test_variance": run_variance_tests,
    "suggested_analysis": show_suggested_analysis,
    "parametric_ttest_one_sample": test_one_sample_ttest,
    "parametric_ttest_independent": test_independent_ttest,
    "parametric_ttest_paired": test_paired_ttest,
    "parametric_anova": test_anova,
    "parametric_twoway_anova": test_two_way_anova,
    "parametric_repeated_anova": test_repeated_anova,
    "correlation_pearson": test_pearson_correlation,
    "regression": test_linear_regression,
    "nonparametric_mannwhitney": test_mannwhitney,
    "nonparametric_wilcoxon": test_wilcoxon,
    "nonparametric_kruskal_wallis": test_kruskal_wallis,
    "nonparametric_friedman": test_friedman,
    "correlation_spearman": test_spearman,
    "correlation_kendall": test_kendall,
    "nonparametric_mood_median": test_mood_median,
    "chi_square": test_chi_square,
}


def render() -> None:
    if st.session_state.df is None:
        st.warning("⚠️ Please upload data first!")
        if st.button("Upload Data"):
            set_page("upload")
    else:
        TEST_PAGE_HANDLERS[st.session_state.page](st.session_state.df)
