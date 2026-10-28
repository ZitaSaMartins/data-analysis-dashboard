"""Legacy parametric tests: t-tests, ANOVA, and the parametric test-list page."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy import stats

from .correlation import test_correlation


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

