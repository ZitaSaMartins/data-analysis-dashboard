import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def run_sensory_analysis(df: pd.DataFrame):
    """Run sensory analysis tools."""
    
    st.markdown("Sensory analysis for food/beverage evaluation and consumer studies.")
    
    analysis_type = st.selectbox(
        "Select sensory analysis type:",
        [
            "Descriptive Analysis (Sensory Profile)",
            "Hedonic Analysis (Liking Scores)",
            "Attribute Ratings Summary",
            "Panelist Performance",
            "Product Comparison"
        ]
    )
    
    if analysis_type == "Descriptive Analysis (Sensory Profile)":
        descriptive_analysis(df)
    
    elif analysis_type == "Hedonic Analysis (Liking Scores)":
        hedonic_analysis(df)
    
    elif analysis_type == "Attribute Ratings Summary":
        attribute_summary(df)
    
    elif analysis_type == "Panelist Performance":
        panelist_performance(df)
    
    elif analysis_type == "Product Comparison":
        product_comparison(df)


def descriptive_analysis(df: pd.DataFrame):
    """Sensory profile/descriptive analysis."""
    st.markdown("### Descriptive Sensory Analysis (Sensory Profile)")
    st.caption("Average ratings across sensory attributes by product/sample.")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if not numeric_cols or not cat_cols:
        st.error("Need both numeric and categorical columns.")
        return
    
    product_var = st.selectbox("Product/Sample variable:", cat_cols)
    attribute_vars = st.multiselect("Sensory attributes (numeric ratings):", numeric_cols)
    
    if st.button("Generate Sensory Profile"):
        # Calculate mean ratings by product
        profile = df.groupby(product_var)[attribute_vars].mean()
        
        st.markdown("**Mean Sensory Ratings by Product:**")
        st.dataframe(profile, use_container_width=True)
        
        # Radar/Spider plot
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2*np.pi, len(attribute_vars), endpoint=False).tolist()
        angles += angles[:1]  # Complete circle
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(profile)))
        
        for idx, (product, row) in enumerate(profile.iterrows()):
            values = row.tolist()
            values += values[:1]  # Complete circle
            ax.plot(angles, values, 'o-', linewidth=2, label=product, color=colors[idx])
            ax.fill(angles, values, alpha=0.15, color=colors[idx])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(attribute_vars, size=9)
        ax.set_ylim(0, 10)
        ax.set_title('Sensory Profile', fontweight='bold', fontsize=14, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
        
        st.pyplot(fig)


def hedonic_analysis(df: pd.DataFrame):
    """Hedonic analysis for consumer liking scores."""
    st.markdown("### Hedonic Analysis (Consumer Liking)")
    st.caption("Analyze consumer preference/liking scores for products.")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if not numeric_cols or not cat_cols:
        st.error("Need both numeric and categorical columns.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        product_var = st.selectbox("Product variable:", cat_cols)
    
    with col2:
        liking_var = st.selectbox("Liking score variable:", numeric_cols)
    
    if st.button("Analyze Liking"):
        # Summary statistics by product
        summary = df.groupby(product_var)[liking_var].agg([
            'mean', 'std', 'count', 'min', 'max'
        ]).round(2)
        
        st.markdown("**Liking Score Summary:**")
        st.dataframe(summary, use_container_width=True)
        
        # Distribution plot
        fig, ax = plt.subplots(figsize=(12, 5))
        
        products = df[product_var].unique()
        for product in products:
            data = df[df[product_var] == product][liking_var].dropna()
            ax.hist(data, bins=20, alpha=0.6, label=product, edgecolor='black')
        
        ax.set_xlabel('Liking Score', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title('Distribution of Liking Scores', fontweight='bold', fontsize=12)
        ax.legend()
        ax.grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Box plot
        fig, ax = plt.subplots(figsize=(10, 6))
        df.boxplot(column=liking_var, by=product_var, ax=ax)
        ax.set_xlabel(product_var, fontweight='bold')
        ax.set_ylabel('Liking Score', fontweight='bold')
        ax.set_title('Liking Scores by Product', fontweight='bold')
        plt.suptitle('')  # Remove default title
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        st.pyplot(fig)


def attribute_summary(df: pd.DataFrame):
    """Summary of sensory attribute ratings."""
    st.markdown("### Sensory Attribute Summary")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        st.error("No numeric columns found.")
        return
    
    selected_attrs = st.multiselect(
        "Select sensory attributes to summarize:",
        numeric_cols,
        default=numeric_cols[:min(5, len(numeric_cols))]
    )
    
    if st.button("Generate Summary"):
        # Comprehensive statistics
        summary = df[selected_attrs].describe().T
        
        st.markdown("**Descriptive Statistics:**")
        st.dataframe(summary, use_container_width=True)
        
        # Heatmap of mean by attribute (if there's a product variable)
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if cat_cols:
            product_var = st.selectbox("Group by (optional):", [None] + cat_cols)
            
            if product_var:
                grouped = df.groupby(product_var)[selected_attrs].mean()
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(grouped, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax, 
                           cbar_kws={'label': 'Mean Rating'})
                ax.set_title('Mean Sensory Attributes by Product', fontweight='bold')
                
                plt.tight_layout()
                st.pyplot(fig)


def panelist_performance(df: pd.DataFrame):
    """Evaluate panelist performance and consistency."""
    st.markdown("### Panelist Performance Evaluation")
    st.caption("Assess panelist discrimination ability and consistency.")
    
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not cat_cols or not numeric_cols:
        st.error("Need categorical (panelist ID) and numeric (ratings) columns.")
        return
    
    panelist_var = st.selectbox("Panelist ID variable:", cat_cols)
    attribute_vars = st.multiselect("Sensory attributes:", numeric_cols, default=numeric_cols[:3])
    
    if st.button("Evaluate Panelists"):
        # Calculate panelist statistics
        panelist_stats = []
        
        for panelist in df[panelist_var].unique():
            panelist_data = df[df[panelist_var] == panelist][attribute_vars]
            
            stats_dict = {
                'Panelist': panelist,
                'Samples': len(panelist_data),
                'Mean Rating': panelist_data.values.mean(),
                'Std Dev': panelist_data.values.std(),
                'CV (%)': (panelist_data.values.std() / panelist_data.values.mean() * 100) if panelist_data.values.mean() != 0 else 0
            }
            panelist_stats.append(stats_dict)
        
        panelist_df = pd.DataFrame(panelist_stats).sort_values('CV (%)')
        
        st.markdown("**Panelist Performance Metrics:**")
        st.dataframe(panelist_df, use_container_width=True)
        
        st.caption("CV (Coefficient of Variation): Lower values indicate more consistent ratings")


def product_comparison(df: pd.DataFrame):
    """Compare multiple products on sensory attributes."""
    st.markdown("### Product Comparison")
    
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not cat_cols or not numeric_cols:
        st.error("Need categorical and numeric columns.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        product_var = st.selectbox("Product variable:", cat_cols)
    
    with col2:
        attribute_var = st.selectbox("Attribute to compare:", numeric_cols)
    
    products = df[product_var].unique()
    
    if len(products) < 2:
        st.error("Need at least 2 products to compare.")
        return
    
    if st.button("Compare Products"):
        # Statistical comparison (Kruskal-Wallis test for non-parametric)
        groups = [df[df[product_var] == p][attribute_var].dropna().values for p in products]
        h_stat, p_value = stats.kruskal(*groups)
        
        st.markdown(f"**Kruskal-Wallis H-test:**")
        st.markdown(f"H-statistic: {h_stat:.3f}")
        st.markdown(f"p-value: {p_value:.4f}")
        
        if p_value < 0.05:
            st.success("✅ Significant differences between products")
        else:
            st.info("❌ No significant differences between products")
        
        # Visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Box plot
        df.boxplot(column=attribute_var, by=product_var, ax=ax1)
        ax1.set_xlabel(product_var, fontweight='bold')
        ax1.set_ylabel(attribute_var, fontweight='bold')
        ax1.set_title('Distribution by Product', fontweight='bold')
        plt.suptitle('')
        
        # Bar plot with error bars
        means = df.groupby(product_var)[attribute_var].mean()
        stds = df.groupby(product_var)[attribute_var].std()
        
        ax2.bar(range(len(means)), means.values, yerr=stds.values, capsize=5, 
               color='steelblue', edgecolor='black', alpha=0.7)
        ax2.set_xticks(range(len(means)))
        ax2.set_xticklabels(means.index, rotation=45, ha='right')
        ax2.set_ylabel(attribute_var, fontweight='bold')
        ax2.set_title('Mean Ratings by Product (±SD)', fontweight='bold')
        ax2.grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        st.pyplot(fig)