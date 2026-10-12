import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_style("whitegrid")

def run_eda(df: pd.DataFrame, components: list):
    """Run selected EDA components."""
    
    if "Summary Statistics" in components:
        show_summary_statistics(df)
    
    if "Distributions" in components:
        show_distributions(df)
    
    if "Correlations" in components:
        show_correlations(df)
    
    if "Box Plots" in components:
        show_box_plots(df)
    
    if "Scatter Plots" in components:
        show_scatter_plots(df)


def show_summary_statistics(df: pd.DataFrame):
    """Display summary statistics for numeric columns."""
    st.markdown("### Summary Statistics")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        st.info("No numeric columns found for summary statistics.")
        return
    
    summary = df[numeric_cols].describe().T
    summary['Missing'] = df[numeric_cols].isnull().sum()
    summary['Missing %'] = (summary['Missing'] / len(df) * 100).round(2)
    
    st.dataframe(summary, use_container_width=True)
    
    # Skewness and Kurtosis
    st.markdown("#### Distribution Shape (Skewness & Kurtosis)")
    
    skew_kurt = pd.DataFrame({
        'Column': numeric_cols,
        'Skewness': [stats.skew(df[col].dropna()) for col in numeric_cols],
        'Kurtosis': [stats.kurtosis(df[col].dropna()) for col in numeric_cols]
    })
    
    st.dataframe(skew_kurt, use_container_width=True)
    
    st.caption("Skewness: Negative (left-skewed) | 0 (symmetric) | Positive (right-skewed)")
    st.caption("Kurtosis: Negative (flat) | 0 (normal) | Positive (peaked)")


def show_distributions(df: pd.DataFrame):
    """Display distributions of columns."""
    st.markdown("### Data Distributions")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if numeric_cols:
        st.markdown("#### Numeric Distributions (Histograms)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_cols = st.multiselect(
                "Select numeric columns to visualize:",
                numeric_cols,
                default=numeric_cols[:min(3, len(numeric_cols))],
                key="numeric_dist"
            )
        
        if selected_cols:
            # Create subplots
            n_cols = len(selected_cols)
            n_rows = (n_cols + 1) // 2
            
            fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4*n_rows))
            axes = axes.flatten() if n_cols > 1 else [axes]
            
            for idx, col in enumerate(selected_cols):
                axes[idx].hist(df[col].dropna(), bins=30, color='steelblue', edgecolor='black', alpha=0.7)
                axes[idx].set_title(f'Distribution of {col}', fontweight='bold')
                axes[idx].set_xlabel(col)
                axes[idx].set_ylabel('Frequency')
                axes[idx].grid(alpha=0.3)
            
            # Hide extra subplots
            for idx in range(len(selected_cols), len(axes)):
                axes[idx].axis('off')
            
            plt.tight_layout()
            st.pyplot(fig)
    
    if cat_cols:
        st.markdown("#### Categorical Distributions")
        
        selected_cat = st.multiselect(
            "Select categorical columns to visualize:",
            cat_cols,
            default=cat_cols[:min(2, len(cat_cols))],
            key="cat_dist"
        )
        
        if selected_cat:
            for col in selected_cat:
                fig, ax = plt.subplots(figsize=(10, 4))
                
                value_counts = df[col].value_counts().head(10)
                value_counts.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
                ax.set_title(f'Distribution of {col}', fontweight='bold', fontsize=12)
                ax.set_xlabel(col)
                ax.set_ylabel('Count')
                ax.grid(alpha=0.3, axis='y')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                st.pyplot(fig)


def show_correlations(df: pd.DataFrame):
    """Display correlation matrix."""
    st.markdown("### Correlation Analysis")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.info("Need at least 2 numeric columns for correlation analysis.")
        return
    
    # Correlation matrix
    corr = df[numeric_cols].corr()
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
                square=True, ax=ax, cbar_kws={'label': 'Correlation'})
    ax.set_title('Correlation Matrix', fontweight='bold', fontsize=14)
    plt.tight_layout()
    
    st.pyplot(fig)
    
    # Find strong correlations
    st.markdown("#### Strong Correlations (|r| > 0.7)")
    
    strong_corr = []
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            if abs(corr.iloc[i, j]) > 0.7:
                strong_corr.append({
                    'Variable 1': corr.columns[i],
                    'Variable 2': corr.columns[j],
                    'Correlation': corr.iloc[i, j]
                })
    
    if strong_corr:
        st.dataframe(pd.DataFrame(strong_corr), use_container_width=True)
    else:
        st.info("No strong correlations found.")


def show_box_plots(df: pd.DataFrame):
    """Display box plots for numeric columns grouped by categorical."""
    st.markdown("### Box Plots")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if not numeric_cols or not cat_cols:
        st.info("Need both numeric and categorical columns for box plots.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_numeric = st.selectbox(
            "Select numeric variable:",
            numeric_cols,
            key="box_numeric"
        )
    
    with col2:
        selected_cat = st.selectbox(
            "Select categorical variable (grouping):",
            cat_cols,
            key="box_cat"
        )
    
    if selected_numeric and selected_cat:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Limit categories to top 10 to avoid overcrowding
        top_cats = df[selected_cat].value_counts().head(10).index
        df_filtered = df[df[selected_cat].isin(top_cats)]
        
        sns.boxplot(data=df_filtered, x=selected_cat, y=selected_numeric, ax=ax)
        ax.set_title(f'{selected_numeric} by {selected_cat}', fontweight='bold', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        st.pyplot(fig)


def show_scatter_plots(df: pd.DataFrame):
    """Display scatter plots for numeric variables."""
    st.markdown("### Scatter Plots")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.info("Need at least 2 numeric columns for scatter plots.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        x_var = st.selectbox(
            "Select X variable:",
            numeric_cols,
            key="scatter_x"
        )
    
    with col2:
        y_var = st.selectbox(
            "Select Y variable:",
            [col for col in numeric_cols if col != x_var],
            key="scatter_y"
        )
    
    if x_var and y_var:
        # Color by categorical if available
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        color_by = None
        
        if cat_cols:
            use_color = st.checkbox("Color by categorical variable?")
            if use_color:
                color_by = st.selectbox("Select coloring variable:", cat_cols)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if color_by:
            for category in df[color_by].unique():
                mask = df[color_by] == category
                ax.scatter(df[mask][x_var], df[mask][y_var], label=category, alpha=0.6, s=50)
            ax.legend()
        else:
            ax.scatter(df[x_var], df[y_var], alpha=0.6, s=50, color='steelblue')
        
        ax.set_xlabel(x_var, fontweight='bold')
        ax.set_ylabel(y_var, fontweight='bold')
        ax.set_title(f'{y_var} vs {x_var}', fontweight='bold', fontsize=12)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Add correlation coefficient
        corr = df[[x_var, y_var]].corr().iloc[0, 1]
        st.caption(f"Pearson correlation: r = {corr:.3f}")