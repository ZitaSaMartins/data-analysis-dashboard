import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats

DEFAULT_SETTINGS = {
    "template": "plotly_white",
    "palette": "Plotly",
    "colorscale": "RdBu",
    "height": 450,
    "font_family": "Arial",
    "font_size": 14,
}


def _settings(chart_settings):
    """Merge user-provided chart settings with defaults."""
    merged = DEFAULT_SETTINGS.copy()
    if chart_settings:
        merged.update({k: v for k, v in chart_settings.items() if v is not None})
    return merged


def _palette(settings):
    return getattr(px.colors.qualitative, settings["palette"], px.colors.qualitative.Plotly)


def _plot(fig, settings, title=None):
    """Apply shared theme/font/size and render so every chart fits its container responsively."""
    fig.update_layout(
        template=settings["template"],
        font=dict(family=settings["font_family"], size=settings["font_size"]),
        height=settings["height"],
        autosize=True,
        margin=dict(l=40, r=20, t=60 if title else 30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=settings["font_size"] + 4)))
    st.plotly_chart(fig, use_container_width=True)


def run_eda(df: pd.DataFrame, components: list, variables: list = None,
            dependent_vars: list = None, independent_vars: list = None,
            chart_settings: dict = None):
    """Run selected EDA components, restricted to the chosen dependent/independent variables."""
    settings = _settings(chart_settings)
    dependent_vars = dependent_vars or []
    independent_vars = independent_vars or []

    # Work only with the variables selected on the EDA page; fall back to all columns
    cols = variables if variables else df.columns.tolist()
    data = df[cols]

    if "Summary Statistics" in components:
        show_summary_statistics(data)

    if "Distributions" in components:
        show_distributions(data, settings)

    if "Correlations" in components:
        show_correlations(data, settings)

    if "Box Plots" in components:
        show_box_plots(data, settings, dependent_vars, independent_vars)

    if "Scatter Plots" in components:
        show_scatter_plots(data, settings, dependent_vars, independent_vars)


def show_summary_statistics(df: pd.DataFrame):
    """Display summary statistics for numeric columns."""
    st.markdown("### Summary Statistics")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        st.info("No numeric columns selected for summary statistics.")
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


def show_distributions(df: pd.DataFrame, settings: dict):
    """Display distributions of columns."""
    st.markdown("### Data Distributions")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    palette = _palette(settings)
    
    if numeric_cols:
        st.markdown("#### Numeric Distributions (Histograms)")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            selected_cols = st.multiselect(
                "Select numeric columns to visualize:",
                numeric_cols,
                default=numeric_cols[:min(3, len(numeric_cols))],
                key="numeric_dist"
            )
        with col2:
            n_bins = st.slider("Bins", 5, 100, 30, key="dist_bins")
        
        for idx, col in enumerate(selected_cols):
            fig = px.histogram(
                df, x=col, nbins=n_bins,
                color_discrete_sequence=[palette[idx % len(palette)]],
                opacity=0.85,
            )
            fig.update_xaxes(title=col)
            fig.update_yaxes(title='Frequency')
            _plot(fig, settings, title=f'Distribution of {col}')
    
    if cat_cols:
        st.markdown("#### Categorical Distributions")
        
        selected_cat = st.multiselect(
            "Select categorical columns to visualize:",
            cat_cols,
            default=cat_cols[:min(2, len(cat_cols))],
            key="cat_dist"
        )
        
        for idx, col in enumerate(selected_cat):
            value_counts = df[col].value_counts().head(10).reset_index()
            value_counts.columns = [col, 'Count']
            fig = px.bar(
                value_counts, x=col, y='Count',
                color_discrete_sequence=[palette[idx % len(palette)]],
            )
            _plot(fig, settings, title=f'Distribution of {col}')


def show_correlations(df: pd.DataFrame, settings: dict):
    """Display correlation matrix."""
    st.markdown("### Correlation Analysis")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.info("Need at least 2 numeric columns selected for correlation analysis.")
        return
    
    # Correlation matrix
    corr = df[numeric_cols].corr()
    
    fig = px.imshow(
        corr, text_auto='.2f', color_continuous_scale=settings["colorscale"],
        zmin=-1, zmax=1, aspect="auto",
    )
    _plot(fig, settings, title='Correlation Matrix')
    
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


def show_box_plots(df: pd.DataFrame, settings: dict, dependent_vars: list, independent_vars: list):
    """Display box plots for numeric columns grouped by categorical."""
    st.markdown("### Box Plots")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if not numeric_cols or not cat_cols:
        st.info("Need both a numeric and a categorical variable selected for box plots.")
        return
    
    dep_numeric = [c for c in dependent_vars if c in numeric_cols]
    indep_cat = [c for c in independent_vars if c in cat_cols]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_numeric = st.selectbox(
            "Numeric variable (value):",
            numeric_cols,
            index=numeric_cols.index(dep_numeric[0]) if dep_numeric else 0,
            key="box_numeric"
        )
    
    with col2:
        selected_cat = st.selectbox(
            "Categorical variable (grouping):",
            cat_cols,
            index=cat_cols.index(indep_cat[0]) if indep_cat else 0,
            key="box_cat"
        )
    
    with col3:
        top_n = st.slider("Max categories shown", 2, 20, 10, key="box_top_n")
    
    if selected_numeric and selected_cat:
        # Limit categories to avoid overcrowding
        top_cats = df[selected_cat].value_counts().head(top_n).index
        df_filtered = df[df[selected_cat].isin(top_cats)]
        
        fig = px.box(
            df_filtered, x=selected_cat, y=selected_numeric, color=selected_cat,
            color_discrete_sequence=_palette(settings), points="outliers",
        )
        fig.update_xaxes(title=selected_cat)
        fig.update_yaxes(title=selected_numeric)
        fig.update_layout(showlegend=False)
        _plot(fig, settings, title=f'{selected_numeric} by {selected_cat}')


def show_scatter_plots(df: pd.DataFrame, settings: dict, dependent_vars: list, independent_vars: list):
    """Display scatter plots for numeric variables."""
    st.markdown("### Scatter Plots")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.info("Need at least 2 numeric columns selected for scatter plots.")
        return
    
    dep_numeric = [c for c in dependent_vars if c in numeric_cols]
    indep_numeric = [c for c in independent_vars if c in numeric_cols]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        x_var = st.selectbox(
            "Select X variable:",
            numeric_cols,
            index=numeric_cols.index(indep_numeric[0]) if indep_numeric else 0,
            key="scatter_x"
        )
    
    with col2:
        y_options = [col for col in numeric_cols if col != x_var]
        default_y = next((c for c in dep_numeric if c != x_var), y_options[0])
        y_var = st.selectbox(
            "Select Y variable:",
            y_options,
            index=y_options.index(default_y),
            key="scatter_y"
        )
    
    with col3:
        marker_size = st.slider("Marker size", 4, 20, 9, key="scatter_size")
    
    # Color by categorical if available
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    color_by = None
    
    if cat_cols:
        use_color = st.checkbox("Color by categorical variable?", key="scatter_use_color")
        if use_color:
            color_by = st.selectbox("Select coloring variable:", cat_cols, key="scatter_color_by")
    
    if x_var and y_var:
        fig = px.scatter(
            df, x=x_var, y=y_var, color=color_by,
            color_discrete_sequence=_palette(settings), opacity=0.75,
        )
        fig.update_traces(marker=dict(size=marker_size, line=dict(width=0.5, color='white')))
        fig.update_xaxes(title=x_var)
        fig.update_yaxes(title=y_var)
        _plot(fig, settings, title=f'{y_var} vs {x_var}')
        
        # Add correlation coefficient
        corr = df[[x_var, y_var]].corr().iloc[0, 1]
        st.caption(f"Pearson correlation: r = {corr:.3f}")