import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

HYPOTHESIS_CATALOG = {
    "Test Distribution": {
        "icon": "📐",
        "desc": "Check whether a variable follows a normal (bell-curve) distribution.",
    },
    "Test Variance": {
        "icon": "📏",
        "desc": "Check whether groups have equal variance (homogeneity of variance).",
    },
    "Suggested Path": {
        "icon": "🧭",
        "desc": "Use distribution and variance results to choose a statistical testing path.",
    },
    "Parametric Tests": {
        "icon": "📈",
        "desc": "Tests that assume normally distributed data and similar variance across groups.",
    },
    "Non-parametric Tests": {
        "icon": "🔀",
        "desc": "Distribution-free tests, used when normality/variance assumptions aren't met.",
    },
}

PARAMETRIC_TEST_GROUPS = {
    "Comparing 2 groups": [
        "Independent Samples t-Test",
        "Paired Samples t-Test",
    ],
    "Comparing more than 2 groups": [
        "One-Way ANOVA",
    ],
    "Relationships between variables": [
        "Pearson Correlation",
    ],
}

NONPARAMETRIC_TEST_GROUPS = {
    "Comparing 2 groups": [
        "Mann-Whitney U Test",
        "Wilcoxon Signed-Rank Test",
    ],
    "Comparing more than 2 groups": [
        "Kruskal-Wallis Test",
    ],
    "Relationships between variables": [
        "Spearman Correlation",
        "Chi-Square Test",
    ],
}


def run_hypothesis_tests(
    df: pd.DataFrame,
    category: str = None,
    parametric_test: str = None,
    nonparametric_test: str = None,
):
    """Run the hypothesis testing interface for the selected sub-menu category."""
    if category == "Suggested Path":
        run_suggested_path(df)
    elif category == "Test Distribution":
        run_test_distribution(df)
    elif category == "Test Variance":
        run_test_variance(df)
    elif category == "Parametric Tests":
        run_parametric_tests(df, parametric_test)
    elif category == "Non-parametric Tests":
        run_nonparametric_tests(df, nonparametric_test)
    else:
        st.info("👈 Select a category from the sidebar to get started.")


# ---------------------------------------------------------------------------
# Suggested Path
# ---------------------------------------------------------------------------

def _open_hypothesis_category(category: str, selected_test: str = None):
    st.session_state.hypothesis_category = category
    st.session_state.parametric_test = (
        selected_test if category == "Parametric Tests" else None
    )
    st.session_state.nonparametric_test = (
        selected_test if category == "Non-parametric Tests" else None
    )
    st.rerun()


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


# ---------------------------------------------------------------------------
# Test Distribution
# ---------------------------------------------------------------------------

def run_test_distribution(df: pd.DataFrame):
    """Normality testing: Shapiro-Wilk or an alternative, chosen by sample size."""
    st.markdown("### 📐 Test Distribution (Normality)")
    st.caption("Checks whether a numeric variable follows a normal distribution — many parametric "
               "tests rely on this assumption.")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.error("No numeric columns found.")
        return

    selected_cols = st.multiselect(
        "Select numeric variable(s) to test:",
        numeric_cols,
        default=numeric_cols[:min(3, len(numeric_cols))],
        key="dist_vars"
    )

    test_options = {
        "Auto (recommended based on sample size)": "auto",
        "Shapiro-Wilk — best for small/medium samples (n < 50)": "shapiro",
        "D'Agostino-Pearson K² — good for larger samples (n ≥ 20)": "dagostino",
        "Anderson-Darling — more sensitive to tail/outlier deviations": "anderson",
        "Kolmogorov-Smirnov — for large samples, less powerful": "ks",
    }
    test_label = st.selectbox("Normality test:", list(test_options.keys()), key="dist_test_choice")
    test_choice = test_options[test_label]

    with st.expander("ℹ️ Which test should I use?"):
        st.markdown("""
        - **Shapiro-Wilk**: the most widely recommended test for small-to-moderate samples
          (roughly n < 50, works up to ~5000). Most powerful in this range.
        - **D'Agostino-Pearson K²**: combines skewness and kurtosis; a solid general-purpose
          choice, especially for larger samples (n ≥ 20).
        - **Anderson-Darling**: gives more weight to the tails of the distribution; useful when
          outliers/tail behavior matter.
        - **Kolmogorov-Smirnov**: compares your data to a normal curve; less powerful than the
          above, mostly used for large samples.
        - **Auto**: picks Shapiro-Wilk for n < 50, otherwise D'Agostino-Pearson K².
        """)

    if not selected_cols:
        return

    if st.button("Run Distribution Test", type="primary", key="run_distribution"):
        st.session_state.distribution_test_result = None
        results = []
        saved_results = []
        for col in selected_cols:
            data = df[col].dropna()
            n = len(data)
            if n < 3:
                st.warning(f"'{col}' has fewer than 3 observations — skipped.")
                continue

            chosen = test_choice
            if chosen == "auto":
                chosen = "shapiro" if n < 50 else "dagostino"

            crit_5pct = None
            if chosen == "shapiro":
                stat, p_value = stats.shapiro(data)
                test_name = "Shapiro-Wilk"
            elif chosen == "dagostino":
                if n < 20:
                    st.warning(f"'{col}': D'Agostino-Pearson needs n ≥ 20; using Shapiro-Wilk instead.")
                    stat, p_value = stats.shapiro(data)
                    test_name = "Shapiro-Wilk (fallback)"
                else:
                    stat, p_value = stats.normaltest(data)
                    test_name = "D'Agostino-Pearson K²"
            elif chosen == "anderson":
                res = stats.anderson(data, dist='norm')
                stat = res.statistic
                crit_5pct = res.critical_values[2]  # 5% significance level
                p_value = None
                test_name = "Anderson-Darling"
            else:
                stat, p_value = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
                test_name = "Kolmogorov-Smirnov"

            if p_value is not None:
                if not np.isfinite(stat) or not np.isfinite(p_value):
                    st.warning(f"'{col}' produced an invalid test result — skipped.")
                    continue
                is_normal = "Yes" if p_value > 0.05 else "No"
                results.append({
                    'Variable': col, 'n': n, 'Test': test_name,
                    'Statistic': round(stat, 4), 'p-value': round(p_value, 4),
                    'Normal (α=0.05)': is_normal
                })
                saved_results.append({
                    "variable": col,
                    "is_normal": is_normal == "Yes",
                    "test_name": test_name,
                    "reason": (
                        f"{test_name}: p = {p_value:.4f} "
                        f"({'>' if p_value > 0.05 else '≤'} 0.05)."
                    ),
                })
            else:
                if not np.isfinite(stat) or not np.isfinite(crit_5pct):
                    st.warning(f"'{col}' produced an invalid test result — skipped.")
                    continue
                is_normal = "Yes" if stat < crit_5pct else "No"
                results.append({
                    'Variable': col, 'n': n, 'Test': test_name,
                    'Statistic': round(stat, 4), 'Critical value (5%)': round(crit_5pct, 4),
                    'Normal (5% level)': is_normal
                })
                saved_results.append({
                    "variable": col,
                    "is_normal": is_normal == "Yes",
                    "test_name": test_name,
                    "reason": (
                        f"{test_name}: statistic = {stat:.4f} "
                        f"({'<' if stat < crit_5pct else '≥'} "
                        f"critical value {crit_5pct:.4f})."
                    ),
                })

        if not results:
            return

        st.session_state.distribution_test_result = {
            "variables": saved_results,
            "alpha": 0.05,
        }
        st.dataframe(pd.DataFrame(results), use_container_width=True)

        st.markdown("#### Visual check")
        for col in selected_cols:
            data = df[col].dropna()
            if len(data) < 3:
                continue
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].hist(data, bins=20, color='steelblue', edgecolor='black', alpha=0.75)
            axes[0].set_title(f'Histogram: {col}')
            stats.probplot(data, dist="norm", plot=axes[1])
            axes[1].set_title(f'Q-Q Plot: {col}')
            plt.tight_layout()
            st.pyplot(fig)


# ---------------------------------------------------------------------------
# Test Variance
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Shared card layout for the Parametric / Non-parametric test lists
# ---------------------------------------------------------------------------

def _render_test_list(tests: list, df: pd.DataFrame):
    for test in tests:
        with st.expander(f"**{test['name']}**"):
            st.markdown(f"**What it does:** {test['what']}")
            st.markdown(f"**When to use it:** {test['when']}")
            st.markdown("---")
            test['fn'](df)


def run_parametric_tests(df: pd.DataFrame, selected_test: str = None):
    st.markdown("### 📈 Parametric Tests")
    st.info(
        "⚠️ **Before using parametric tests:** your data should be approximately **normally "
        "distributed** (check with *Test Distribution*) and groups should have **similar "
        "variance** (check with *Test Variance*). If these assumptions aren't met, use the "
        "**Non-parametric Tests** instead."
    )

    tests = [
        {
            "name": "Independent Samples t-Test",
            "what": "Compares the means of two independent groups.",
            "when": "Comparing a numeric variable between exactly 2 independent groups, with "
                    "approximately normal data and similar variances in each group.",
            "fn": test_ttest,
        },
        {
            "name": "Paired Samples t-Test",
            "what": "Compares the means of two related measurements (e.g., before/after on the "
                    "same subjects).",
            "when": "You have two numeric measurements per subject (paired/repeated data) and "
                    "the differences are approximately normally distributed.",
            "fn": test_paired_ttest,
        },
        {
            "name": "One-Way ANOVA",
            "what": "Compares the means of 3 or more independent groups at once.",
            "when": "Comparing a numeric variable across 3+ independent groups, assuming "
                    "normality and roughly equal variances within each group.",
            "fn": test_anova,
        },
        {
            "name": "Pearson Correlation",
            "what": "Measures the strength and direction of a LINEAR relationship between two "
                    "numeric variables.",
            "when": "Both variables are continuous, approximately normally distributed, and you "
                    "expect a straight-line relationship.",
            "fn": lambda d: test_correlation(d, default_method="Pearson", key_prefix="param_corr"),
        },
    ]
    if selected_test:
        selected = next((test for test in tests if test["name"] == selected_test), None)
        if selected is not None:
            with st.expander(f"**{selected['name']}**", expanded=True):
                st.markdown(f"**What it does:** {selected['what']}")
                st.markdown(f"**When to use it:** {selected['when']}")
                st.markdown("---")
                selected["fn"](df)
            return

    st.info("Select a parametric test from the sidebar.")


def run_nonparametric_tests(df: pd.DataFrame, selected_test: str = None):
    st.markdown("### 🔀 Non-parametric Tests")
    st.info(
        "ℹ️ **Non-parametric tests make no assumption about normal distribution.** Use them "
        "when *Test Distribution* shows your data isn't normal, when you have outliers or "
        "ordinal/ranked data, or when sample sizes are small. They're generally safer but "
        "slightly less powerful than parametric tests when normality truly holds."
    )

    tests = [
        {
            "name": "Mann-Whitney U Test",
            "what": "Non-parametric alternative to the independent t-test; compares the ranks "
                    "of two independent groups.",
            "when": "Comparing 2 independent groups when data is not normally distributed, has "
                    "outliers, or is ordinal.",
            "fn": test_mannwhitney,
        },
        {
            "name": "Wilcoxon Signed-Rank Test",
            "what": "Non-parametric alternative to the paired t-test; compares two related/paired "
                    "measurements.",
            "when": "Paired/repeated measurements when the differences are not normally distributed.",
            "fn": test_wilcoxon,
        },
        {
            "name": "Kruskal-Wallis Test",
            "what": "Non-parametric alternative to one-way ANOVA; compares 3+ independent groups.",
            "when": "Comparing 3+ independent groups when data is not normally distributed or "
                    "variances differ a lot.",
            "fn": test_kruskal,
        },
        {
            "name": "Spearman Correlation",
            "what": "Measures the strength and direction of a MONOTONIC (not necessarily linear) "
                    "relationship, using ranks.",
            "when": "Data is ordinal, non-normally distributed, or the relationship may be "
                    "non-linear but still monotonic.",
            "fn": lambda d: test_correlation(d, default_method="Spearman", key_prefix="nonparam_corr"),
        },
        {
            "name": "Chi-Square Test",
            "what": "Tests whether two categorical variables are associated (independent or not).",
            "when": "Both variables are categorical — no normality assumption is needed.",
            "fn": test_chi_square,
        },
    ]
    if selected_test:
        selected = next((test for test in tests if test["name"] == selected_test), None)
        if selected is not None:
            with st.expander(f"**{selected['name']}**", expanded=True):
                st.markdown(f"**What it does:** {selected['what']}")
                st.markdown(f"**When to use it:** {selected['when']}")
                st.markdown("---")
                selected["fn"](df)
            return

    st.info("Select a non-parametric test from the sidebar.")




def test_ttest(df: pd.DataFrame):
    """Independent samples t-test."""
    st.markdown("### Independent Samples t-Test")
    st.caption("Tests if the means of two groups are significantly different.")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if not numeric_cols or not cat_cols:
        st.error("Need both numeric and categorical columns.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        numeric_var = st.selectbox("Numeric variable (dependent):", numeric_cols)
    
    with col2:
        grouping_var = st.selectbox("Grouping variable (categorical):", cat_cols)
    
    # Get unique groups
    groups = df[grouping_var].unique()
    groups = [g for g in groups if pd.notna(g)]
    
    if len(groups) < 2:
        st.error("Grouping variable must have at least 2 groups.")
        return
    
    if len(groups) > 2:
        st.warning(f"Found {len(groups)} groups. Selecting first 2 for t-test.")
        groups = groups[:2]
    
    col1, col2 = st.columns(2)
    
    with col1:
        group1 = st.selectbox("Group 1:", groups, index=0)
    
    with col2:
        group2 = st.selectbox("Group 2:", groups, index=min(1, len(groups)-1))
    
    if st.button("Run t-test"):
        # Get data for each group
        data1 = df[df[grouping_var] == group1][numeric_var].dropna()
        data2 = df[df[grouping_var] == group2][numeric_var].dropna()
        
        if len(data1) < 2 or len(data2) < 2:
            st.error("Each group needs at least 2 observations.")
            return
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(data1, data2)
        
        # Effect size (Cohen's d)
        cohens_d = (data1.mean() - data2.mean()) / np.sqrt(((len(data1)-1)*data1.std()**2 + (len(data2)-1)*data2.std()**2) / (len(data1) + len(data2) - 2))
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Group 1 Mean", f"{data1.mean():.2f}")
        with col2:
            st.metric("Group 2 Mean", f"{data2.mean():.2f}")
        with col3:
            st.metric("t-statistic", f"{t_stat:.3f}")
        
        st.markdown(f"**p-value:** {p_value:.4f}")
        st.markdown(f"**Cohen's d (effect size):** {cohens_d:.3f}")
        
        # Interpretation
        alpha = 0.05
        if p_value < alpha:
            st.success(f"✅ **Significant difference** (p < {alpha}). The means are significantly different.")
        else:
            st.info(f"❌ **No significant difference** (p ≥ {alpha}). The means are not significantly different.")
        
        # Visualization
        fig, ax = plt.subplots(figsize=(10, 5))
        
        data_to_plot = pd.DataFrame({
            'Value': list(data1) + list(data2),
            'Group': [group1]*len(data1) + [group2]*len(data2)
        })
        
        sns.boxplot(data=data_to_plot, x='Group', y='Value', ax=ax, palette=['lightblue', 'lightcoral'])
        ax.set_title(f't-test: {numeric_var} by {grouping_var}', fontweight='bold')
        ax.set_ylabel(numeric_var)
        
        plt.tight_layout()
        st.pyplot(fig)


def test_anova(df: pd.DataFrame):
    """One-way ANOVA test."""
    st.markdown("### One-Way ANOVA")
    st.caption("Tests if means across 3+ groups are significantly different.")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if not numeric_cols or not cat_cols:
        st.error("Need both numeric and categorical columns.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        numeric_var = st.selectbox("Numeric variable:", numeric_cols)
    
    with col2:
        grouping_var = st.selectbox("Grouping variable:", cat_cols)
    
    if st.button("Run ANOVA"):
        # Get groups
        groups = df[grouping_var].unique()
        groups = [g for g in groups if pd.notna(g)]
        
        if len(groups) < 3:
            st.warning("ANOVA typically compares 3+ groups. Use t-test for 2 groups.")
        
        # Prepare data for ANOVA
        group_data = [df[df[grouping_var] == g][numeric_var].dropna().values for g in groups]
        
        # Perform ANOVA
        f_stat, p_value = stats.f_oneway(*group_data)
        
        st.markdown(f"**F-statistic:** {f_stat:.3f}")
        st.markdown(f"**p-value:** {p_value:.4f}")
        
        # Interpretation
        alpha = 0.05
        if p_value < alpha:
            st.success(f"✅ **Significant difference** (p < {alpha}). At least one group mean is different.")
        else:
            st.info(f"❌ **No significant difference** (p ≥ {alpha}). No significant differences between groups.")
        
        # Visualization
        fig, ax = plt.subplots(figsize=(10, 5))
        
        data_to_plot = pd.DataFrame({
            'Value': df[numeric_var].dropna(),
            'Group': df.loc[df[numeric_var].notna(), grouping_var]
        })
        
        sns.boxplot(data=data_to_plot, x='Group', y='Value', ax=ax)
        ax.set_title(f'ANOVA: {numeric_var} by {grouping_var}', fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)


def test_mannwhitney(df: pd.DataFrame):
    """Mann-Whitney U test (non-parametric alternative to t-test)."""
    st.markdown("### Mann-Whitney U Test")
    st.caption("Non-parametric alternative to t-test. Use when data is not normally distributed.")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if not numeric_cols or not cat_cols:
        st.error("Need both numeric and categorical columns.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        numeric_var = st.selectbox("Numeric variable:", numeric_cols, key="mw_numeric")
    
    with col2:
        grouping_var = st.selectbox("Grouping variable:", cat_cols, key="mw_cat")
    
    if st.button("Run Mann-Whitney U test"):
        groups = df[grouping_var].unique()
        groups = [g for g in groups if pd.notna(g)][:2]
        
        data1 = df[df[grouping_var] == groups[0]][numeric_var].dropna()
        data2 = df[df[grouping_var] == groups[1]][numeric_var].dropna()
        
        u_stat, p_value = stats.mannwhitneyu(data1, data2)
        
        st.markdown(f"**U-statistic:** {u_stat:.3f}")
        st.markdown(f"**p-value:** {p_value:.4f}")
        
        alpha = 0.05
        if p_value < alpha:
            st.success(f"✅ **Significant difference** (p < {alpha})")
        else:
            st.info(f"❌ **No significant difference** (p ≥ {alpha})")


def test_chi_square(df: pd.DataFrame):
    """Chi-square test for categorical variables."""
    st.markdown("### Chi-Square Test")
    st.caption("Tests association between two categorical variables.")
    
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if len(cat_cols) < 2:
        st.error("Need at least 2 categorical columns.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        var1 = st.selectbox("First categorical variable:", cat_cols)
    
    with col2:
        var2 = st.selectbox("Second categorical variable:", [v for v in cat_cols if v != var1])
    
    if st.button("Run Chi-Square test"):
        # Create contingency table
        contingency = pd.crosstab(df[var1], df[var2])
        
        # Perform chi-square test
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
        
        st.markdown(f"**Chi-square statistic:** {chi2:.3f}")
        st.markdown(f"**p-value:** {p_value:.4f}")
        st.markdown(f"**Degrees of freedom:** {dof}")
        
        alpha = 0.05
        if p_value < alpha:
            st.success(f"✅ **Significant association** (p < {alpha})")
        else:
            st.info(f"❌ **No significant association** (p ≥ {alpha})")
        
        # Display contingency table
        st.markdown("**Contingency Table:**")
        st.dataframe(contingency, use_container_width=True)


def test_correlation(df: pd.DataFrame, default_method: str = "Pearson", key_prefix: str = "corr"):
    """Correlation test (Pearson and Spearman)."""
    st.markdown("### Correlation Test")
    st.caption("Tests linear or monotonic association between numeric variables.")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.error("Need at least 2 numeric columns.")
        return
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        var1 = st.selectbox("First variable:", numeric_cols, key=f"{key_prefix}_var1")
    
    with col2:
        var2 = st.selectbox("Second variable:", [v for v in numeric_cols if v != var1], key=f"{key_prefix}_var2")
    
    with col3:
        methods = ["Pearson", "Spearman"]
        corr_type = st.selectbox("Correlation:", methods, index=methods.index(default_method), key=f"{key_prefix}_method")
    
    if st.button("Calculate correlation", key=f"{key_prefix}_run"):
        data1 = df[var1].dropna()
        data2 = df[var2].dropna()
        
        # Align data
        idx = data1.index.intersection(data2.index)
        data1 = data1[idx]
        data2 = data2[idx]
        
        if corr_type == "Pearson":
            corr_coef, p_value = stats.pearsonr(data1, data2)
        else:
            corr_coef, p_value = stats.spearmanr(data1, data2)
        
        st.markdown(f"**{corr_type} correlation coefficient:** {corr_coef:.3f}")
        st.markdown(f"**p-value:** {p_value:.4f}")
        
        alpha = 0.05
        if p_value < alpha:
            st.success(f"✅ **Significant correlation** (p < {alpha})")
        else:
            st.info(f"❌ **No significant correlation** (p ≥ {alpha})")


def test_paired_ttest(df: pd.DataFrame):
    """Paired samples t-test."""
    st.markdown("### Paired Samples t-Test")
    st.caption("Compares two related numeric measurements (e.g., before/after) from the same subjects.")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        st.error("Need at least 2 numeric columns.")
        return

    col1, col2 = st.columns(2)
    with col1:
        var1 = st.selectbox("Measurement 1:", numeric_cols, key="paired_var1")
    with col2:
        var2 = st.selectbox("Measurement 2:", [c for c in numeric_cols if c != var1], key="paired_var2")

    if st.button("Run Paired t-test", key="run_paired"):
        paired = df[[var1, var2]].dropna()
        if len(paired) < 2:
            st.error("Need at least 2 paired observations.")
            return

        t_stat, p_value = stats.ttest_rel(paired[var1], paired[var2])
        diff = paired[var1] - paired[var2]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(f"Mean {var1}", f"{paired[var1].mean():.2f}")
        with col2:
            st.metric(f"Mean {var2}", f"{paired[var2].mean():.2f}")
        with col3:
            st.metric("t-statistic", f"{t_stat:.3f}")

        st.markdown(f"**Mean difference:** {diff.mean():.3f}")
        st.markdown(f"**p-value:** {p_value:.4f}")

        alpha = 0.05
        if p_value < alpha:
            st.success(f"✅ **Significant difference** (p < {alpha}) between the paired measurements.")
        else:
            st.info(f"❌ **No significant difference** (p ≥ {alpha}) between the paired measurements.")


def test_wilcoxon(df: pd.DataFrame):
    """Wilcoxon signed-rank test (non-parametric alternative to paired t-test)."""
    st.markdown("### Wilcoxon Signed-Rank Test")
    st.caption("Non-parametric alternative to the paired t-test for two related numeric measurements.")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        st.error("Need at least 2 numeric columns.")
        return

    col1, col2 = st.columns(2)
    with col1:
        var1 = st.selectbox("Measurement 1:", numeric_cols, key="wilcoxon_var1")
    with col2:
        var2 = st.selectbox("Measurement 2:", [c for c in numeric_cols if c != var1], key="wilcoxon_var2")

    if st.button("Run Wilcoxon Test", key="run_wilcoxon"):
        paired = df[[var1, var2]].dropna()
        if len(paired) < 2:
            st.error("Need at least 2 paired observations.")
            return

        try:
            stat, p_value = stats.wilcoxon(paired[var1], paired[var2])
        except ValueError as e:
            st.error(f"Could not run test: {e}")
            return

        st.markdown(f"**W-statistic:** {stat:.3f}")
        st.markdown(f"**p-value:** {p_value:.4f}")

        alpha = 0.05
        if p_value < alpha:
            st.success(f"✅ **Significant difference** (p < {alpha}) between the paired measurements.")
        else:
            st.info(f"❌ **No significant difference** (p ≥ {alpha}) between the paired measurements.")


def test_kruskal(df: pd.DataFrame):
    """Kruskal-Wallis test (non-parametric alternative to one-way ANOVA)."""
    st.markdown("### Kruskal-Wallis Test")
    st.caption("Non-parametric alternative to one-way ANOVA for 3+ independent groups.")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    if not numeric_cols or not cat_cols:
        st.error("Need both numeric and categorical columns.")
        return

    col1, col2 = st.columns(2)
    with col1:
        numeric_var = st.selectbox("Numeric variable:", numeric_cols, key="kruskal_numeric")
    with col2:
        grouping_var = st.selectbox("Grouping variable:", cat_cols, key="kruskal_cat")

    if st.button("Run Kruskal-Wallis Test", key="run_kruskal"):
        groups = [g for g in df[grouping_var].unique() if pd.notna(g)]
        if len(groups) < 2:
            st.error("Need at least 2 groups.")
            return

        group_data = [df[df[grouping_var] == g][numeric_var].dropna().values for g in groups]
        group_data = [g for g in group_data if len(g) >= 1]

        h_stat, p_value = stats.kruskal(*group_data)

        st.markdown(f"**H-statistic:** {h_stat:.3f}")
        st.markdown(f"**p-value:** {p_value:.4f}")

        alpha = 0.05
        if p_value < alpha:
            st.success(f"✅ **Significant difference** (p < {alpha}). At least one group differs.")
        else:
            st.info(f"❌ **No significant difference** (p ≥ {alpha}).")

        fig, ax = plt.subplots(figsize=(10, 5))
        data_to_plot = pd.DataFrame({
            'Value': df[numeric_var].dropna(),
            'Group': df.loc[df[numeric_var].notna(), grouping_var]
        })
        sns.boxplot(data=data_to_plot, x='Group', y='Value', ax=ax)
        ax.set_title(f'Kruskal-Wallis: {numeric_var} by {grouping_var}', fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)