"""Legacy correlation test, shared by the parametric (Pearson) and non-parametric (Spearman) pages."""

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats


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

