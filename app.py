import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import sys

# Import custom modules
from modules.data_validation import validate_dataset, generate_validation_report
from modules.eda import run_eda, recommend_components, build_preview, CHART_INFO, CHART_CATALOG
from modules.hypothesis_tests import run_hypothesis_tests
from modules.sensory_analysis import run_sensory_analysis
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
        background-color: transparent;
        color: #C7DCE5;
        border: none;
        border-left: 3px solid transparent;
        border-radius: 0;
        padding: 0.55rem 1.6rem 0.55rem 0.9rem;
        font-weight: 500;
        font-size: 0.92rem;
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

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'validation_report' not in st.session_state:
    st.session_state.validation_report = None
if 'page' not in st.session_state:
    st.session_state.page = "home"
if 'eda_category' not in st.session_state:
    st.session_state.eda_category = list(CHART_CATALOG.keys())[0]

def set_page(page_name):
    st.session_state.page = page_name

# Sidebar navigation - flat list with section labels and chevrons, always visible
with st.sidebar:
    st.markdown("## 📊 Navigation")

    if st.session_state.df is not None:
        st.success(f"✅ {st.session_state.df.shape[0]} rows × {st.session_state.df.shape[1]} cols")
    else:
        st.caption("No data loaded yet")

    def nav_button(page_name, label):
        is_active = st.session_state.page == page_name
        if st.button(label, key=f"nav_{page_name}", width="stretch",
                    type="primary" if is_active else "secondary"):
            set_page(page_name)
            st.rerun()

    def nav_section(title):
        st.markdown(f'<p class="nav-section-label">{title}</p>', unsafe_allow_html=True)

    nav_button("home", "Home")

    nav_section("Prepare")
    nav_button("upload", "Upload data")
    nav_button("validate", "Check data quality")

    nav_section("Analyze")
    is_eda_active = st.session_state.page == "eda"
    with st.expander("Explore data", expanded=is_eda_active):
        for category in CHART_CATALOG.keys():
            is_active_category = is_eda_active and st.session_state.eda_category == category
            if st.button(category, key=f"nav_eda_{category}", width="stretch",
                        type="primary" if is_active_category else "secondary"):
                st.session_state.eda_category = category
                set_page("eda")
                st.rerun()
    nav_button("hypothesis", "Test hypotheses")
    nav_button("sensory", "Sensory analysis")

    nav_section("Advanced analysis")
    nav_button("ml", "Build predictions")
    nav_button("recommend", "Get recommendations")

    nav_section("Help")
    nav_button("help_data", "Getting started")
    nav_button("help_analysis", "Analysis types")


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
    1. **Upload & Prepare** — Upload Data, then Check Data Quality
    2. **Analyze** — Explore Data, Test Hypotheses, Sensory Analysis
    3. **Advanced** — Build Predictions, Get Recommendations
    """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #999; font-size: 0.9rem;'>
        <p>📚 <b>Tip:</b> Follow the steps in order: Upload → Check → Analyze → Advanced</p>
        <p>No coding knowledge needed • All analysis in your browser • Your data stays private</p>
    </div>
    """, unsafe_allow_html=True)


# PAGE 1: Upload Data
if st.session_state.page == "upload":
    st.markdown("## 📤 Upload Your Data")
    st.markdown("Choose an Excel or CSV file with your data")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader(
            "📎 Select file:",
            type=["xlsx", "xls", "csv"],
            help="Supported formats: Excel (.xlsx, .xls) or CSV"
        )
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.session_state.df = df
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
            
            st.info("✨ Great! Your data is uploaded. Next step: Check data quality →")
            
            if st.button("➡️ Check Data Quality", type="primary"):
                set_page("validate")
                st.rerun()
            
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


# PAGE 2: Data Validation
elif st.session_state.page == "validate":
    if st.session_state.df is None:
        st.warning("⚠️ Please upload data first!")
        if st.button("Upload Data"):
            set_page("upload")
            st.rerun()
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
                set_page("ml")
                st.rerun()


# PAGE 3: EDA
elif st.session_state.page == "eda":
    if st.session_state.df is None:
        st.warning("⚠️ Please upload data first!")
        if st.button("Upload Data"):
            set_page("upload")
            st.rerun()
    else:
        st.markdown("## 🔍 Explore Your Data")
        st.markdown("Visualize patterns and relationships in your data")
        
        df = st.session_state.df
        
        st.markdown("### Select Variables to Analyze")
        col1, col2 = st.columns(2)
        with col1:
            eda_independent = st.multiselect(
                "Independent Variables (Features/Predictors):",
                options=df.columns,
                default=[v for v in st.session_state.get('independent_vars', []) if v in df.columns],
                key="eda_independent_vars",
                help="Variables you'll use to predict or explain"
            )
        with col2:
            eda_dependent = st.multiselect(
                "Dependent Variables (Target/Response):",
                options=[c for c in df.columns if c not in eda_independent],
                default=[v for v in st.session_state.get('dependent_vars', []) if v in df.columns and v not in eda_independent],
                key="eda_dependent_vars",
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
            st.caption("⭐ marks the chart types recommended for your current variable selection. Each card shows a live preview. Switch categories from the \"Explore data\" menu in the sidebar.")
            
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
                if eda_options:
                    run_eda(df, eda_options, eda_vars, eda_dependent, eda_independent, chart_settings)
                else:
                    st.warning("Select at least one option")


# PAGE 4: Hypothesis Testing
elif st.session_state.page == "hypothesis":
    if st.session_state.df is None:
        st.warning("⚠️ Please upload data first!")
        if st.button("Upload Data"):
            set_page("upload")
            st.rerun()
    else:
        st.markdown("## 📊 Test Your Hypotheses")
        st.markdown("Answer statistical questions about your data")
        
        df = st.session_state.df
        run_hypothesis_tests(df)


# PAGE 5: Sensory Analysis
elif st.session_state.page == "sensory":
    if st.session_state.df is None:
        st.warning("⚠️ Please upload data first!")
        if st.button("Upload Data"):
            set_page("upload")
            st.rerun()
    else:
        st.markdown("## 👃 Sensory Analysis")
        st.markdown("Consumer preferences and sensory evaluation")
        
        df = st.session_state.df
        run_sensory_analysis(df)


# PAGE 6: Machine Learning
elif st.session_state.page == "ml":
    if st.session_state.df is None:
        st.warning("⚠️ Please upload data first!")
        if st.button("Upload Data"):
            set_page("upload")
            st.rerun()
    else:
        st.markdown("## 🤖 Build Predictions")
        st.markdown("Train machine learning models to predict outcomes")
        
        if not hasattr(st.session_state, 'dependent_vars') or not hasattr(st.session_state, 'independent_vars'):
            st.warning("⚠️ Please go to 'Check Data Quality' and specify your target and predictor variables first!")
            if st.button("Go to Check Data Quality"):
                set_page("validate")
                st.rerun()
        else:
            df = st.session_state.df
            run_ml_pipeline(
                df,
                st.session_state.dependent_vars,
                st.session_state.independent_vars
            )


# PAGE 7: Model Recommender
elif st.session_state.page == "recommend":
    st.markdown("## 💡 Which Model Should I Use?")
    st.markdown("Get expert recommendations based on your data and question")
    
    if st.session_state.df is None:
        st.warning("⚠️ Please upload data first!")
        if st.button("Upload Data"):
            set_page("upload")
            st.rerun()
    else:
        df = st.session_state.df
        
        col1, col2 = st.columns(2)
        
        with col1:
            analysis_type = st.selectbox(
                "🎯 What are you trying to do?",
                ["EDA (Explore data)", "Hypothesis Testing (Test differences)", 
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
                "Hypothesis Testing (Test differences)": "Hypothesis Testing",
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
elif st.session_state.page == "help_data":
    st.markdown("## ❓ Help: Getting Started")
    
    with st.expander("How do I prepare my data?", expanded=True):
        st.markdown("""
        Your data should be in a spreadsheet with:
        - **First row:** Column names/headers
        - **Other rows:** Your actual data
        
        Example:
        | Product | Taste | Texture | Liking |
        |---------|-------|---------|--------|
        | A | 7 | 8 | 7.5 |
        | B | 6 | 9 | 8 |
        """)
    
    with st.expander("What file formats work?"):
        st.markdown("""
        ✅ **Excel:** .xlsx, .xls
        ✅ **Spreadsheet:** .csv
        ❌ **Google Sheets:** Export as CSV first
        ❌ **PDF:** Not supported
        """)
    
    with st.expander("What if my data has missing values?"):
        st.markdown("""
        Missing values are OK! The dashboard will:
        - Show you where they are
        - Skip them in analysis
        - Warn you if there are too many
        """)
    
    with st.expander("What should I do after uploading?"):
        st.markdown("""
        1. **Check Data Quality** - Make sure everything looks right
        2. **Explore Data** - See patterns and relationships
        3. **Test Hypotheses** - Answer your research questions
        4. **Build Predictions** - Train models (optional)
        """)

elif st.session_state.page == "help_analysis":
    st.markdown("## ❓ Help: Analysis Types")
    
    with st.expander("🔍 Explore Data", expanded=True):
        st.markdown("""
        Shows you patterns in your data through:
        - **Statistics:** Mean, median, standard deviation
        - **Distributions:** Histograms showing data shape
        - **Correlations:** Which variables go together
        - **Visualizations:** Charts and plots
        """)
    
    with st.expander("📊 Test Hypotheses"):
        st.markdown("""
        Answers YES/NO questions like:
        - "Are products different in quality?"
        - "Does temperature affect taste?"
        - "Is there a relationship between price and liking?"
        
        Uses p-values: p < 0.05 usually means YES
        """)
    
    with st.expander("👃 Sensory Analysis"):
        st.markdown("""
        Special analysis for food/consumer testing:
        - **Descriptive:** What does the product taste like?
        - **Hedonic:** How much do consumers like it?
        - **Comparison:** Which product is preferred?
        """)
    
    with st.expander("🤖 Build Predictions"):
        st.markdown("""
        Teaches the computer to predict outcomes:
        - **Classification:** "Will this product be liked?" (Yes/No)
        - **Regression:** "What will the rating be?" (Number)
        """)
    
    with st.expander("💡 Get Recommendations"):
        st.markdown("""
        Explains which analysis method to use for your question.
        All recommendations are expert suggestions, not code!
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #999; font-size: 0.85rem;'>
    <p>Data Analysis Dashboard • No coding required • Your data stays private</p>
    <p style='font-size: 0.75rem;'>Built with Streamlit • Open source • Free to use</p>
</div>
""", unsafe_allow_html=True)