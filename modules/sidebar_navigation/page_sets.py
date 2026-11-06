"""Page-id sets used to decide which sidebar section should start expanded."""


HOME_PAGES = {"home"}


PREPARE_PAGES = {"upload", "validate", "rename"}


EXPLORE_PAGES = {
    "eda_distributions",
    "eda_correlations",
    "eda_comparisons",
    "eda_hierarchical",
    "eda_temporal",
}


RELATIONSHIP_PAGES = {
    "test_distribution",
    "test_variance",
    "suggested_analysis",
    "parametric_ttest_one_sample",
    "parametric_ttest_independent",
    "parametric_ttest_paired",
    "parametric_anova",
    "parametric_twoway_anova",
    "parametric_repeated_anova",
    "nonparametric_mannwhitney",
    "nonparametric_wilcoxon",
    "nonparametric_kruskal_wallis",
    "nonparametric_friedman",
    "nonparametric_mood_median",
    "correlation_pearson",
    "correlation_spearman",
    "correlation_kendall",
    "regression",
    "chi_square",
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


ADVANCED_PAGES = {"machine_learning", "recommendations"}


HELP_PAGES = {
    "help_getting_started",
    "help_analysis_types",
    "help_choose_tests",
}


# Top-level app sections shown as a horizontal bar. Each section's sidebar
# only renders the submenu for whichever section is currently active.
SECTIONS = [
    {
        "id": "home",
        "label": "Home",
        "icon": "🏠",
        "default_page": "home",
        "pages": HOME_PAGES,
    },
    {
        "id": "prepare",
        "label": "Prepare Data",
        "icon": "📤",
        "default_page": "upload",
        "pages": PREPARE_PAGES,
    },
    {
        "id": "explore",
        "label": "Explore Data",
        "icon": "🔍",
        "default_page": "eda_distributions",
        "pages": EXPLORE_PAGES,
    },
    {
        "id": "relationships",
        "label": "Analyze Relationships",
        "icon": "📊",
        "default_page": "test_distribution",
        "pages": RELATIONSHIP_PAGES,
    },
    {
        "id": "sensory",
        "label": "Sensory Analysis",
        "icon": "👃",
        "default_page": "sensory_triangle",
        "pages": SENSORY_PAGES,
    },
    {
        "id": "advanced",
        "label": "Advanced Analysis",
        "icon": "🤖",
        "default_page": "machine_learning",
        "pages": ADVANCED_PAGES,
    },
    {
        "id": "help",
        "label": "Help",
        "icon": "❓",
        "default_page": "help_getting_started",
        "pages": HELP_PAGES,
    },
]

