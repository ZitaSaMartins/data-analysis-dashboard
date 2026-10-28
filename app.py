import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import hashlib
import sys

# Import custom modules
from modules.data_validation import validate_dataset, generate_validation_report
from modules.eda import run_eda, recommend_components, build_preview, CHART_INFO, CHART_CATALOG
from modules.sidebar_navigation import (
    build_main_navigation,
    show_sidebar_status,
    show_test_decision_tree,
)
from modules.hypothesis_tests_comprehensive import (
    run_distribution_tests,
    run_variance_tests,
    show_suggested_analysis,
    test_one_sample_ttest,
    test_independent_ttest,
    test_paired_ttest,
    test_anova,
    test_two_way_anova,
    test_repeated_anova,
    test_pearson_correlation,
    test_linear_regression,
    test_mannwhitney,
    test_wilcoxon,
    test_kruskal_wallis,
    test_friedman,
    test_spearman,
    test_kendall,
    test_mood_median,
    test_chi_square,
)
from modules.sensory_analysis import run_sensory_page
from modules.ml_models import run_ml_pipeline
from modules.model_recommender import recommend_models

# Page config
st.set_page_config(
    page_title="Data Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling - Clean, student-friendly interface
st.markdown("""
    <style>
    /* Hide the default sidebar menu */
    [data-testid="stSidebarNav"] {display: none;}
    
    /* Main header */
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Card styling for menu buttons */
    .menu-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        cursor: pointer;
        text-align: center;
        transition: transform 0.2s;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        margin: 0.5rem;
        font-size: 1.1rem;
        font-weight: bold;
    }
    
    .menu-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    }
    
    .menu-card.upload { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    .menu-card.validate { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }
    .menu-card.eda { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); }
    .menu-card.hypothesis { background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); }
    .menu-card.sensory { background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); }
    .menu-card.ml { background: linear-gradient(135deg, #30cfd0 0%, #330867 100%); }
    .menu-card.recommend { background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); }
    
    .menu-card.recommend { color: #333; }
    
    /* Section headers */
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #667eea;
        padding-bottom: 0.5rem;
    }
    
    .step-number {
        background: #667eea;
        color: white;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 1.2rem;
        margin-right: 1rem;
    }
    
    .info-box {
        background: #e8f4f8;
        border-left: 4px solid #0288d1;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    
    .success-box {
        background: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    
    .back-button {
        background: #f0f0f0;
        color: #333;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        cursor: pointer;
        text-decoration: none;
        margin-bottom: 1rem;
        display: inline-block;
    }
    
    /* Sidebar menu - flat list style with chevrons, like a help-center nav */
    section[data-testid="stSidebar"] p.nav-section-label {
        color: #8DB2D9;
        opacity: 0.75;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 1.25rem 0 0.25rem 0.25rem;
    }

    section[data-testid="stSidebar"] .stButton > button {
        position: relative;
        width: 100%;
        text-align: left;
        justify-content: flex-start;
        background-color: transparent;
        color: #C7DCE5;
        border: none;
        border-left: 3px solid transparent;
        border-radius: 0;
        padding: 0.55rem 1.6rem 0.55rem 1.15rem;
        font-weight: 500;
        font-size: 0.92rem;
    }

    section[data-testid="stSidebar"] .stButton > button p {
        width: 100%;
        margin: 0;
        text-align: left;
    }

    section[data-testid="stSidebar"] .stButton > button [data-testid="stMarkdownContainer"] {
        width: 100%;
    }

    section[data-testid="stSidebar"] .stButton > button > div,
    section[data-testid="stSidebar"] .stButton > button > div > span {
        width: 100%;
        justify-content: flex-start;
    }
    
    section[data-testid="stSidebar"] .stButton > button::after {
        content: "›";
        position: absolute;
        right: 0.85rem;
        top: 50%;
        transform: translateY(-50%);
        font-size: 1.1rem;
        opacity: 0.55;
    }
    
    section[data-testid="stSidebar"] .stButton > button:hover {
        background-color: rgba(255,255,255,0.08);
        color: #ffffff;
    }
    
    section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background-color: rgba(255,255,255,0.1);
        color: #ffffff;
        border-left: 3px solid #FFBF3E;
        font-weight: 700;
    }
    
    section[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
        background-color: rgba(255,255,255,0.14);
    }
    </style>
""", unsafe_allow_html=True)

EDA_PAGE_CATEGORIES = {
    "eda_distributions": "Distribution",
    "eda_correlations": "Correlation",
    "eda_comparisons": "Comparison",
    "eda_hierarchical": "Part-to-Whole & Hierarchical",
    "eda_temporal": "Temporal",
}

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

SENSORY_PAGES = {
    "sensory_triangle",
    "sensory_paired",
    "sensory_duo_trio",
    "sensory_tetrad",
    "sensory_ranking",
    "sensory_qda",
    "sensory_temporal",
    "sensory_panel_discrimination",
    "sensory_panel_repeatability",
    "sensory_panel_agreement",
    "sensory_panel_scale",
    "sensory_product_characterization",
    "sensory_cata",
    "sensory_jar",
    "sensory_hedonic",
    "sensory_segmentation",
    "sensory_internal_preference",
    "sensory_external_preference",
    "sensory_pca",
    "sensory_gpa",
    "sensory_statis",
    "sensory_hac",
    "sensory_mfa",
    "sensory_pls",
}

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'uploaded_filename' not in st.session_state:
    st.session_state.uploaded_filename = None
if 'validation_report' not in st.session_state:
    st.session_state.validation_report = None
if 'page' not in st.session_state:
    st.session_state.page = "home"
if 'eda_category' not in st.session_state:
    st.session_state.eda_category = list(CHART_CATALOG.keys())[0]
if 'dataset_signature' not in st.session_state:
    st.session_state.dataset_signature = None

legacy_eda_pages = {
    "Distribution": "eda_distributions",
    "Correlation": "eda_correlations",
    "Comparison": "eda_comparisons",
    "Part-to-Whole & Hierarchical": "eda_hierarchical",
    "Temporal": "eda_temporal",
}
legacy_page_map = {
    "eda": legacy_eda_pages.get(st.session_state.eda_category, "eda_comparisons"),
    "hypothesis": "test_distribution",
    "sensory": "sensory_qda",
    "sensory_descriptive": "sensory_qda",
    "sensory_comparison": "sensory_product_characterization",
    "ml": "machine_learning",
    "recommend": "recommendations",
    "help_data": "help_getting_started",
    "help_analysis": "help_analysis_types",
}
if st.session_state.page in legacy_page_map:
    st.session_state.page = legacy_page_map[st.session_state.page]


def set_page(page_name):
    st.session_state.page = page_name
    st.rerun()


def store_eda_independent_selection():
    st.session_state.eda_independent_selection = list(
        st.session_state.get("_eda_independent_vars", [])
    )


def store_eda_dependent_selection():
    st.session_state.eda_dependent_selection = list(
        st.session_state.get("_eda_dependent_vars", [])
    )


build_main_navigation()
show_sidebar_status()


# HOME PAGE - Welcome & status
if st.session_state.page == "home":
    st.markdown('<p class="main-header">📊 Data Analysis Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Your easy guide to understanding data without coding</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    if st.session_state.df is not None:
        st.success(f"✅ Data loaded: {st.session_state.df.shape[0]} rows × {st.session_state.df.shape[1]} columns")
    else:
        st.info("👉 Use the sidebar menu to get started — begin with **Upload Data**")
    
    st.markdown("### 🗺️ How it works")
    st.markdown("""
    Use the **menu in the sidebar** to navigate between steps:
    1. **Upload & Prepare** — Upload Data, Check Data Quality, then Rename Variables
    2. **Explore** — Choose a chart category to explore patterns and relationships
    3. **Statistical Inference** — Check assumptions, review Suggested Analysis, then choose a test
    4. **Sensory** — Open the independent Sensory Analysis module when needed
    5. **Advanced** — Build Predictions, Get Recommendations
    6. **Help** — Getting Started, Analysis Types, and How to Choose Tests guides
    """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #999; font-size: 0.9rem;'>
        <p>📚 <b>Tip:</b> Follow the steps in order: Upload → Check → Explore → Test → Advanced</p>
        <p>No coding knowledge needed • All analysis in your browser • Your data stays private</p>
    </div>
    """, unsafe_allow_html=True)


# PAGE 1: Upload Data
if st.session_state.page == "upload":
    st.markdown("## 📤 Upload Your Data")
    st.markdown("Choose an Excel or CSV file with your data")
    
    if st.session_state.uploaded_filename:
        st.caption(f"📄 Currently loaded file: **{st.session_state.uploaded_filename}**")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader(
            "📎 Select file:",
            type=["xlsx", "xls", "csv"],
            help="Supported formats: Excel (.xlsx, .xls) or CSV"
        )
    
    if uploaded_file is not None:
        try:
            dataset_signature = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
            dataset_changed = dataset_signature != st.session_state.dataset_signature
            if dataset_changed:
                st.session_state.distribution_test_result = None
                st.session_state.variance_test_result = None
                st.session_state.dataset_signature = dataset_signature

            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.session_state.df = df
            st.session_state.uploaded_filename = uploaded_file.name
            if dataset_changed:
                st.rerun()
            st.success(f"✅ Successfully loaded: {uploaded_file.name}")
            
            # Display basic info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Total Rows", f"{df.shape[0]:,}")
            with col2:
                st.metric("📋 Total Columns", df.shape[1])
            with col3:
                st.metric("⚠️ Missing Values", df.isnull().sum().sum())
            
            # Preview
            st.markdown("### Preview of Your Data")
            st.dataframe(df.head(10), use_container_width=True)
            
            st.info("✨ Great! Your data is uploaded. Next step: Rename variables (optional) →")
            
            if st.button("✏️ Rename Variables", type="primary"):
                set_page("rename")
            
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            st.info("Try saving your file as CSV or Excel format")
    else:
        st.info("👆 Click above to select your file")
        
        with st.expander("ℹ️ What file should I use?"):
            st.markdown("""
            **Accepted formats:**
            - Excel: .xlsx, .xls
            - Spreadsheet: .csv
            
            **Requirements:**
            - First row: Column names (headers)
            - Data below: Your actual data
            - No blank rows in the middle
            
            **Example:**
            | Product | Score | Temperature |
            |---------|-------|-------------|
            | A | 7.5 | 23 |
            | B | 8.2 | 24 |
            """)


# PAGE 1b: Rename Variables
elif st.session_state.page == "rename":
    if st.session_state.df is None:
        st.warning("⚠️ Please upload data first!")
        if st.button("Upload Data"):
            set_page("upload")
    else:
        st.markdown("## ✏️ Rename Variables")
        st.markdown("Give your columns clearer names — accents, special characters, and any language script are supported")
        
        df = st.session_state.df
        
        st.caption("Type any characters, accents, or language script — e.g. 'Température', '甜み', 'Préférence (n=50)'.")
        rename_table = pd.DataFrame({"Current Name": df.columns, "New Name": df.columns})
        edited_names = st.data_editor(
            rename_table,
            column_config={
                "Current Name": st.column_config.TextColumn(disabled=True),
                "New Name": st.column_config.TextColumn(),
            },
            hide_index=True,
            width="stretch",
            key="rename_editor"
        )
        
        if st.button("✅ Apply Renaming", type="primary"):
            new_names = [str(n).strip() for n in edited_names["New Name"]]
            if any(not n for n in new_names):
                st.error("❌ Column names cannot be empty.")
            elif len(set(new_names)) != len(new_names):
                st.error("❌ Column names must be unique.")
            else:
                rename_map = {old: new for old, new in zip(df.columns, new_names) if old != new}
                if rename_map:
                    df = df.rename(columns=rename_map)
                    st.session_state.df = df
                    st.session_state.distribution_test_result = None
                    st.session_state.variance_test_result = None
                    st.session_state.dataset_signature = None
                    for state_key in (
                        "dependent_vars",
                        "independent_vars",
                        "eda_dependent_vars",
                        "eda_independent_vars",
                        "eda_dependent_selection",
                        "eda_independent_selection",
                        "assumption_variables",
                    ):
                        if state_key in st.session_state:
                            st.session_state[state_key] = [rename_map.get(v, v) for v in st.session_state[state_key]]
                    st.success(f"✅ Renamed {len(rename_map)} column(s)! The new names are used everywhere in the dashboard.")
                else:
                    st.info("No changes to apply.")
        
        if st.button("➡️ Check Data Quality"):
            set_page("validate")


# PAGE 2: Data Validation
elif st.session_state.page == "validate":
    if st.session_state.df is None:
        st.warning("⚠️ Please upload data first!")
        if st.button("Upload Data"):
            set_page("upload")
    else:
        st.markdown("## ✅ Check Data Quality")
        st.markdown("Making sure your data is ready for analysis")
        
        df = st.session_state.df
        
        # Run validation
        validation_result = validate_dataset(df)
        st.session_state.validation_report = validation_result
        
        # Display validation results
        if validation_result['is_valid']:
            st.success("✅ Data validation passed!")
        else:
            st.warning("⚠️ Data has issues that should be reviewed:")
        
        # Validation Report Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Column Info", "Missing Data", "Issues"])
        
        with tab1:
            st.markdown("**Dataset Summary:**")
            report = generate_validation_report(df)
            st.write(report)
        
        with tab2:
            st.markdown("**Column Information:**")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': [str(df[col].dtype) for col in df.columns],
                'Non-Null Count': [df[col].notna().sum() for col in df.columns],
                'Unique Values': [df[col].nunique() for col in df.columns]
            })
            st.dataframe(col_info, use_container_width=True)
        
        with tab3:
            st.markdown("**Missing Values Analysis:**")
            missing = df.isnull().sum()
            missing_pct = (missing / len(df) * 100).round(2)
            missing_df = pd.DataFrame({
                'Column': missing[missing > 0].index,
                'Missing Count': missing[missing > 0].values,
                'Percentage': missing_pct[missing > 0].values
            })
            if len(missing_df) > 0:
                st.dataframe(missing_df, use_container_width=True)
            else:
                st.success("✅ No missing values!")
        
        with tab4:
            if validation_result['issues']:
                for issue in validation_result['issues']:
                    if issue.startswith('❌'):
                        st.error(issue)
                    else:
                        st.warning(issue)
            else:
                st.success("✅ No data format issues detected!")
        
        # Dependent/Independent variable specification
        st.markdown('<p class="section-text">Specify Variable Roles</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            independent_vars = st.multiselect(
                "Select Independent Variables (Features/Predictors):",
                options=df.columns,
                key="independent_vars",
                help="Variables you'll use to predict or explain"
            )
        
        with col2:
            dependent_vars = st.multiselect(
                "Select Dependent Variables (Target/Response):",
                options=[col for col in df.columns if col not in independent_vars],
                key="dependent_vars",
                help="Variables you want to predict or analyze"
            )
        
        if dependent_vars and independent_vars:
            st.success("✅ Variable roles configured!")
            if st.button("🤖 Go to Build Predictions"):
                set_page("machine_learning")


# PAGE 3: EDA
elif st.session_state.page in EDA_PAGE_CATEGORIES:
    st.session_state.eda_category = EDA_PAGE_CATEGORIES[st.session_state.page]
    if st.session_state.df is None:
        st.warning("⚠️ Please upload data first!")
        if st.button("Upload Data"):
            set_page("upload")
    else:
        st.markdown("## 🔍 Explore Your Data")
        st.markdown("Visualize patterns and relationships in your data")
        
        df = st.session_state.df
        
        st.markdown("### Select Variables to Analyze")
        if "eda_independent_selection" not in st.session_state:
            st.session_state.eda_independent_selection = list(
                st.session_state.get(
                    "eda_independent_vars",
                    st.session_state.get("independent_vars", []),
                )
            )
        if "eda_dependent_selection" not in st.session_state:
            st.session_state.eda_dependent_selection = list(
                st.session_state.get(
                    "eda_dependent_vars",
                    st.session_state.get("dependent_vars", []),
                )
            )

        selected_independent = [
            variable
            for variable in st.session_state.eda_independent_selection
            if variable in df.columns
        ]
        selected_dependent = [
            variable
            for variable in st.session_state.eda_dependent_selection
            if variable in df.columns and variable not in selected_independent
        ]
        st.session_state.eda_independent_selection = selected_independent
        st.session_state.eda_dependent_selection = selected_dependent

        # Widget keys can be removed during a navigation rerun; restore them
        # from persistent, non-widget state before rendering each EDA page.
        st.session_state._eda_independent_vars = selected_independent
        st.session_state._eda_dependent_vars = selected_dependent

        col1, col2 = st.columns(2)
        with col1:
            eda_independent = st.multiselect(
                "Independent Variables (Features/Predictors):",
                options=df.columns,
                key="_eda_independent_vars",
                on_change=store_eda_independent_selection,
                help="Variables you'll use to predict or explain"
            )
        with col2:
            eda_dependent = st.multiselect(
                "Dependent Variables (Target/Response):",
                options=[c for c in df.columns if c not in eda_independent],
                key="_eda_dependent_vars",
                on_change=store_eda_dependent_selection,
                help="Variables you want to predict or analyze"
            )
        
        eda_vars = eda_dependent + [v for v in eda_independent if v not in eda_dependent]
        
        if not eda_vars:
            st.info("👆 Select at least one dependent or independent variable to analyze")
        else:
            with st.expander("🎨 Chart Settings", expanded=False):
                s1, s2, s3 = st.columns(3)
                with s1:
                    chart_template = st.selectbox(
                        "Theme",
                        ["plotly_white", "plotly", "plotly_dark", "ggplot2", "seaborn", "simple_white"],
                        key="chart_template"
                    )
                    chart_palette = st.selectbox(
                        "Color palette (categories)",
                        ["Plotly", "D3", "T10", "G10", "Set1", "Set2", "Set3", "Pastel", "Bold", "Vivid", "Dark24"],
                        key="chart_palette"
                    )
                with s2:
                    chart_colorscale = st.selectbox(
                        "Color scale (heatmap/numeric)",
                        ["RdBu", "Viridis", "Plasma", "Blues", "Greens", "Reds", "Turbo", "Cividis", "Sunset", "Tealrose"],
                        key="chart_colorscale"
                    )
                    chart_height = st.slider("Plot height (px)", 300, 900, 450, step=50, key="chart_height")
                with s3:
                    chart_font_family = st.selectbox(
                        "Font",
                        ["Arial", "Helvetica", "Times New Roman", "Courier New", "Georgia", "Verdana", "Open Sans, sans-serif"],
                        key="chart_font_family"
                    )
                    chart_font_size = st.slider("Font size", 10, 24, 14, key="chart_font_size")
            
            chart_settings = {
                "template": chart_template,
                "palette": chart_palette,
                "colorscale": chart_colorscale,
                "height": chart_height,
                "font_family": chart_font_family,
                "font_size": chart_font_size,
            }
            
            st.markdown(f"### {st.session_state.eda_category}")
            st.caption("⭐ marks the chart types recommended for your current variable selection. Each card shows a live preview. Switch categories directly under \"Explore\" in the sidebar.")
            
            show_summary = st.checkbox("📋 Also include Summary Statistics table", value=False, key="include_summary_stats")
            
            recommended = recommend_components(df, eda_vars)
            preview_df = df[eda_vars]
            
            chart_names = list(CHART_CATALOG[st.session_state.eda_category].keys())
            eda_options = ["Summary Statistics"] if show_summary else []
            gallery_cols = st.columns(3)
            for idx, name in enumerate(chart_names):
                info = CHART_INFO[name]
                is_recommended = recommended.get(name, False)
                with gallery_cols[idx % 3]:
                    with st.container(border=True):
                        badge = " ⭐" if is_recommended else ""
                        st.markdown(f"**{info['icon']} {name}{badge}**")
                        st.caption(info['desc'])
                        preview_fig = build_preview(name, preview_df, chart_settings)
                        if preview_fig is not None:
                            st.plotly_chart(
                                preview_fig, use_container_width=True,
                                config={"staticPlot": True, "displayModeBar": False},
                                key=f"preview_{name}"
                            )
                        else:
                            st.caption("Not enough matching variables selected")
                        include = st.checkbox("Include", value=is_recommended, key=f"include_{name}")
                        if include:
                            eda_options.append(name)
            
            if st.button("🚀 Generate Analysis", type="primary"):
                st.session_state.eda_generated = True
            
            # Keep rendering on every rerun (e.g. when a customize color picker changes)
            # instead of only on the exact run the button was clicked.
            if st.session_state.get("eda_generated"):
                if eda_options:
                    run_eda(df, eda_options, eda_vars, eda_dependent, eda_independent, chart_settings)
                else:
                    st.warning("Select at least one option")


# PAGE 4: Statistical Tests
elif st.session_state.page in TEST_PAGE_HANDLERS:
    if st.session_state.df is None:
        st.warning("⚠️ Please upload data first!")
        if st.button("Upload Data"):
            set_page("upload")
    else:
        TEST_PAGE_HANDLERS[st.session_state.page](st.session_state.df)


# PAGE 5: Sensory Analysis
elif st.session_state.page in SENSORY_PAGES:
    if st.session_state.df is None:
        st.warning("⚠️ Please upload data first!")
        if st.button("Upload Data"):
            set_page("upload")
    else:
        st.markdown("## 👃 Sensory Analysis")
        st.markdown("Consumer preferences and sensory evaluation")
        
        run_sensory_page(st.session_state.df, st.session_state.page)


# PAGE 6: Machine Learning
elif st.session_state.page == "machine_learning":
    if st.session_state.df is None:
        st.warning("⚠️ Please upload data first!")
        if st.button("Upload Data"):
            set_page("upload")
    else:
        st.markdown("## 🤖 Build Predictions")
        st.markdown("Train machine learning models to predict outcomes")
        
        if not hasattr(st.session_state, 'dependent_vars') or not hasattr(st.session_state, 'independent_vars'):
            st.warning("⚠️ Please go to 'Check Data Quality' and specify your target and predictor variables first!")
            if st.button("Go to Check Data Quality"):
                set_page("validate")
        else:
            df = st.session_state.df
            run_ml_pipeline(
                df,
                st.session_state.dependent_vars,
                st.session_state.independent_vars
            )


# PAGE 7: Model Recommender
elif st.session_state.page == "recommendations":
    st.markdown("## 💡 Which Model Should I Use?")
    st.markdown("Get expert recommendations based on your data and question")
    
    if st.session_state.df is None:
        st.warning("⚠️ Please upload data first!")
        if st.button("Upload Data"):
            set_page("upload")
    else:
        df = st.session_state.df
        
        col1, col2 = st.columns(2)
        
        with col1:
            analysis_type = st.selectbox(
                "🎯 What are you trying to do?",
                ["EDA (Explore data)", "Statistical Inference (Test differences)",
                 "Classification (Predict categories)", "Regression (Predict numbers)", 
                 "Sensory Analysis (Consumer preferences)"],
                format_func=lambda x: x.split("(")[0].strip() if "(" in x else x
            )
        
        with col2:
            st.markdown("**Your data looks like:**")
            has_small = st.checkbox("Small dataset (<100 rows)", value=False)
            has_imbalance = st.checkbox("Imbalanced classes", value=False)
            has_nonlinear = st.checkbox("Complex patterns", value=False)
        
        data_characteristics = []
        if has_small: data_characteristics.append("Small sample size (<100)")
        if has_imbalance: data_characteristics.append("Imbalanced classes")
        if has_nonlinear: data_characteristics.append("Non-linear relationships")
        
        if st.button("📌 Get Recommendations", type="primary"):
            # Extract analysis type from the formatted string
            analysis_type_mapping = {
                "EDA (Explore data)": "EDA",
                "Statistical Inference (Test differences)": "Hypothesis Testing",
                "Classification (Predict categories)": "Classification",
                "Regression (Predict numbers)": "Regression",
                "Sensory Analysis (Consumer preferences)": "Sensory Analysis"
            }
            analysis_type_key = analysis_type_mapping.get(analysis_type, analysis_type)
            
            recommendations = recommend_models(
                analysis_type_key,
                df,
                data_characteristics
            )
            
            st.markdown("---")
            st.markdown("### 📊 Top Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                color = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
                with st.expander(f"{color} **{rec['model']}** (Suitability: {rec['suitability']}/10)", expanded=(i==1)):
                    st.markdown(f"**💡 Why use this?**\n{rec['reason']}")
                    st.markdown(f"**✅ Advantages:**\n{rec['pros']}")
                    st.markdown(f"**⚠️ Disadvantages:**\n{rec['cons']}")
                    st.markdown(f"**🎯 Best for:**\n{rec['when_to_use']}")

# HELP PAGES
elif st.session_state.page == "help_getting_started":
    st.markdown("## ❓ Help: Getting Started")
    st.write(
        "Follow this workflow from preparing your file through selecting and "
        "interpreting an analysis."
    )
    
    with st.expander("How do I prepare my data?", expanded=True):
        st.markdown("""
        Your data should be in a spreadsheet with:
        - **First row:** Column names/headers
        - **Each row:** One observation, participant, sample, or measurement
        - **Each column:** One variable
        
        Example:
        | Product | Taste | Texture | Liking |
        |---------|-------|---------|--------|
        | A | 7 | 8 | 7.5 |
        | B | 6 | 9 | 8 |

        Use short, meaningful column names. You can change them later with
        **Prepare Data → Rename Variables**.
        """)
    
    with st.expander("How do I upload and confirm my dataset?"):
        st.markdown("""
        1. Open **Prepare Data → Upload Data**.
        2. Select an Excel (`.xlsx`, `.xls`) or CSV (`.csv`) file.
        3. Review the preview, row count, column count, and missing-value count.
        4. Check the sidebar status card. It shows the **file name** and
           **dataset dimensions** throughout the dashboard.

        Google Sheets should be exported as CSV first. PDF files are not supported.
        """)
    
    with st.expander("What should I check before analysis?"):
        st.markdown("""
        Open **Prepare Data → Check Data Quality** to:
        - Review variable types and missing values
        - Identify blocking errors and non-blocking warnings
        - Assign independent variables (features/predictors)
        - Assign dependent variables (targets/responses)

        Missing values are allowed. Analyses remove missing observations where
        necessary, and the dashboard warns when too little valid data remains.
        """)
    
    with st.expander("What is the recommended workflow?"):
        st.markdown("""
        1. **Upload Data** and confirm the file status.
        2. **Check Data Quality** and optionally rename variables.
        3. **Explore Data** using Distributions, Correlations, Comparisons,
           Part-to-Whole & Hierarchical, or Temporal charts.
        4. **Test Distribution** and **Test Variance** for inference questions.
           Numeric-variable selections are shared automatically between these pages.
        5. Open **Suggested Analysis** to combine matching assumption results,
           identify missing checks, and open a recommended inference test.
        6. Use **Sensory Analysis** or **Advanced Analysis** when relevant.
        """)

    with st.expander("Why do my variable selections appear on other pages?"):
        st.markdown("""
        The dashboard intentionally reuses selections:
        - Explore Data variables remain selected when switching chart categories.
        - Test Distribution and Test Variance share one numeric-variable selection.
        - Suggested Analysis uses that same selection and reports missing results
          for each variable.
        - A recommended test opens with its numeric and grouping variables preselected.

        Uploading a different dataset or renaming columns safely refreshes related
        state so results are not mixed between datasets.
        """)

elif st.session_state.page == "help_analysis_types":
    st.markdown("## ❓ Help: Analysis Types")
    st.write("Choose a section based on the question you want to answer.")
    
    with st.expander("🔍 Explore Data", expanded=True):
        st.markdown("""
        Use Explore Data to describe patterns before formal inference:
        - **Distributions:** Histograms, categorical bars, box plots, and violin plots
        - **Correlations:** Scatter, contour, bubble, and correlation heatmap charts
        - **Comparisons:** Bar, grouped bar, and radar charts
        - **Part-to-Whole & Hierarchical:** Pie, donut, treemap, and sunburst charts
        - **Temporal:** Line, area, and stacked-area charts

        Recommended chart cards are marked with ⭐. Variable selections are preserved
        across categories, and chart options refresh automatically.
        """)
    
    with st.expander("📐 Assumption Checks and Suggested Analysis"):
        st.markdown("""
        **Test Distribution** offers Automatic, Shapiro-Wilk,
        D'Agostino-Pearson K², Anderson-Darling, and Kolmogorov-Smirnov options.
        Review the histogram, fitted normal curve, Q-Q plot, and box plot.

        **Test Variance** offers Automatic, Levene, Bartlett, and
        Fligner-Killeen options. Review grouped box plots, violin plots, variance
        bars, and group-level summaries.

        **Suggested Analysis** matches both results by variable. It explains the
        recommended parametric, adjusted-parametric, or non-parametric path,
        identifies missing checks, and opens the recommended test directly.
        """)

    with st.expander("👥 Comparing Groups"):
        st.markdown("""
        - **Two independent groups:** Independent t-test, Welch t-test, or
          Mann-Whitney U
        - **Two related measurements:** Paired t-test or Wilcoxon signed-rank
        - **Three or more independent groups:** One-way ANOVA, Kruskal-Wallis,
          or Mood's median test
        - **Pairwise follow-up:** Tukey HSD, Bonferroni, Scheffé, Fisher's LSD,
          or Holm-adjusted Welch comparisons for parametric analyses; Dunn or
          Conover-Iman with Holm correction for rank-based analyses
        - **Two independent factors:** Two-way ANOVA, including both main effects
          and their interaction
        - **Three or more related measurements:** Repeated-measures ANOVA or Friedman

        Parametric tests generally require approximately normal data. Standard
        independent t-tests and ANOVA also rely on similar group variances.
        Pairwise grouping letters indicate adjusted results: groups sharing a
        letter are not significantly different.
        """)

    with st.expander("🔗 Relationships, Regression, and Categorical Data"):
        st.markdown("""
        - **Pearson correlation:** Linear relationship between approximately
          normal numeric variables
        - **Spearman correlation:** Monotonic rank relationship without a
          normality requirement
        - **Kendall's tau:** Rank association, especially useful for smaller
          samples or tied ranks
        - **Linear regression:** Predict a numeric outcome from a numeric predictor
        - **Chi-square:** Association between two categorical variables
        """)
    
    with st.expander("👃 Sensory Analysis"):
        st.markdown("""
        - **Discrimination Tests:** Triangle, paired comparison, duo-trio,
          tetrad, and ranking workflows
        - **Descriptive Analysis:** QDA/sensory profiles and temporal methods
        - **Panel Analysis:** Discrimination, repeatability, agreement, and scale use
        - **Product Characterization:** Inferential comparisons and product maps
        - **CATA and JAR:** Binary attribute associations and penalty analysis
        - **Consumer Preference:** Hedonic testing, segmentation, and internal or
          external preference mapping
        - **Advanced Multivariate:** PCA, GPA, STATIS, hierarchical clustering,
          MFA, and PLS regression

        Each category expands into its individual analyses in the sidebar.
        Pages explain the required data structure and validate the selected fields
        before running an analysis.
        """)
    
    with st.expander("🤖 Advanced Analysis"):
        st.markdown("""
        - **Build Predictions:** Train classification or regression models using
          the variable roles selected during data-quality checking.
        - **Get Recommendations:** Review model suggestions based on your goal,
          dataset size, class balance, and pattern complexity.
        """)

elif st.session_state.page == "help_choose_tests":
    show_test_decision_tree()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #999; font-size: 0.85rem;'>
    <p>Data Analysis Dashboard • No coding required • Your data stays private</p>
    <p style='font-size: 0.75rem;'>Built with Streamlit • Open source • Free to use</p>
</div>
""", unsafe_allow_html=True)