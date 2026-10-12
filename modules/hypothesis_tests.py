import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def run_hypothesis_tests(df: pd.DataFrame):
    """Run hypothesis testing interface."""
    
    st.markdown("Choose a hypothesis test based on your data and research question.")
    
    test_type = st.selectbox(
        "Select hypothesis test:",
        [
            "Comparing Two Groups (Independent t-test)",
            "Comparing More Than Two Groups (ANOVA)",
            "Non-parametric Alternative (Mann-Whitney U)",
            "Categorical Variables (Chi-Square)",
            "Correlation Test (Pearson/Spearman)",
            "Normality Test (Shapiro-Wilk)"
        ]
    )
    
    if test_type == "Comparing Two Groups (Independent t-test)":
        test_ttest(df)
    
    elif test_type == "Comparing More Than Two Groups (ANOVA)":
        test_anova(df)
    
    elif test_type == "Non-parametric Alternative (Mann-Whitney U)":
        test_mannwhitney(df)
    
    elif test_type == "Categorical Variables (Chi-Square)":
        test_chi_square(df)
    
    elif test_type == "Correlation Test (Pearson/Spearman)":
        test_correlation(df)
    
    elif test_type == "Normality Test (Shapiro-Wilk)":
        test_normality(df)


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


def test_correlation(df: pd.DataFrame):
    """Correlation test (Pearson and Spearman)."""
    st.markdown("### Correlation Test")
    st.caption("Tests linear or monotonic association between numeric variables.")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.error("Need at least 2 numeric columns.")
        return
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        var1 = st.selectbox("First variable:", numeric_cols)
    
    with col2:
        var2 = st.selectbox("Second variable:", [v for v in numeric_cols if v != var1])
    
    with col3:
        corr_type = st.selectbox("Correlation:", ["Pearson", "Spearman"])
    
    if st.button("Calculate correlation"):
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


def test_normality(df: pd.DataFrame):
    """Shapiro-Wilk normality test."""
    st.markdown("### Normality Test (Shapiro-Wilk)")
    st.caption("Tests if data is normally distributed.")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        st.error("No numeric columns found.")
        return
    
    selected_cols = st.multiselect(
        "Select columns to test for normality:",
        numeric_cols,
        default=numeric_cols[:min(3, len(numeric_cols))]
    )
    
    if st.button("Run Normality Tests"):
        results = []
        
        for col in selected_cols:
            data = df[col].dropna()
            
            if len(data) < 3:
                st.warning(f"Column {col} has < 3 observations, skipping.")
                continue
            
            w_stat, p_value = stats.shapiro(data)
            
            is_normal = "Yes" if p_value > 0.05 else "No"
            results.append({
                'Column': col,
                'W-statistic': w_stat,
                'p-value': p_value,
                'Normally Distributed': is_normal
            })
        
        results_df = pd.DataFrame(results)
        st.dataframe(results_df, use_container_width=True)