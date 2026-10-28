"""Legacy non-parametric tests: Mann-Whitney, Wilcoxon, Kruskal-Wallis, Chi-square."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy import stats

from .correlation import test_correlation


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

