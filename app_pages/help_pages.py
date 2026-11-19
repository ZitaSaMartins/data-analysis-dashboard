import streamlit as st

from modules.sidebar_navigation import show_test_decision_tree


def render() -> None:
    page = st.session_state.page
    if page == "help_getting_started":
        _getting_started()
    elif page == "help_analysis_types":
        _analysis_types()
    elif page == "help_choose_tests":
        show_test_decision_tree()

def _getting_started() -> None:
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

def _analysis_types() -> None:
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
        - **CATA:** Data validation, Cochran's Q with pairwise letters, attribute
            independence, Correspondence Analysis, correlation/PCoA, and an
            ideal-product penalty analysis
        - **JAR:** Just-About-Right penalty analysis
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
