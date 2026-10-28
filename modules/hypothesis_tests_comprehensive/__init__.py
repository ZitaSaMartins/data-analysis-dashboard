"""Comprehensive, student-friendly statistical tests for the dashboard.

This package mirrors the previous single-file module's public API so existing
imports such as ``from modules.hypothesis_tests_comprehensive import test_anova``
continue to work unchanged. Internals are split by statistical topic:

- ``_shared``: cross-cutting helpers (column pickers, result rendering).
- ``distribution`` / ``variance``: assumption-check pages.
- ``recommendations``: combines both assumption checks into test suggestions.
- ``posthoc``: shared pairwise/post-hoc comparison engine.
- ``parametric`` / ``nonparametric`` / ``correlation`` / ``categorical``: the
  individual test pages, grouped by family.
"""

from .categorical import test_chi_square
from .correlation import (
    test_kendall,
    test_linear_regression,
    test_pearson_correlation,
    test_spearman,
)
from .distribution import run_distribution_tests
from .nonparametric import (
    test_friedman,
    test_kruskal_wallis,
    test_mannwhitney,
    test_mood_median,
    test_wilcoxon,
)
from .parametric import (
    test_anova,
    test_independent_ttest,
    test_one_sample_ttest,
    test_paired_ttest,
    test_repeated_anova,
    test_two_way_anova,
)
from .recommendations import show_suggested_analysis
from .variance import run_variance_tests

__all__ = [
    "run_distribution_tests",
    "run_variance_tests",
    "show_suggested_analysis",
    "test_one_sample_ttest",
    "test_independent_ttest",
    "test_paired_ttest",
    "test_anova",
    "test_two_way_anova",
    "test_repeated_anova",
    "test_pearson_correlation",
    "test_linear_regression",
    "test_mannwhitney",
    "test_wilcoxon",
    "test_kruskal_wallis",
    "test_friedman",
    "test_spearman",
    "test_kendall",
    "test_mood_median",
    "test_chi_square",
]
