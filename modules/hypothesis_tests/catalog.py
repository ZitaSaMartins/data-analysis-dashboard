"""Legacy hypothesis-testing menu catalog (category and test-group metadata)."""


HYPOTHESIS_CATALOG = {
    "Test Distribution": {
        "icon": "📐",
        "desc": "Check whether a variable follows a normal (bell-curve) distribution.",
    },
    "Test Variance": {
        "icon": "📏",
        "desc": "Check whether groups have equal variance (homogeneity of variance).",
    },
    "Suggested Path": {
        "icon": "🧭",
        "desc": "Use distribution and variance results to choose a statistical testing path.",
    },
    "Parametric Tests": {
        "icon": "📈",
        "desc": "Tests that assume normally distributed data and similar variance across groups.",
    },
    "Non-parametric Tests": {
        "icon": "🔀",
        "desc": "Distribution-free tests, used when normality/variance assumptions aren't met.",
    },
}


PARAMETRIC_TEST_GROUPS = {
    "Comparing 2 groups": [
        "Independent Samples t-Test",
        "Paired Samples t-Test",
    ],
    "Comparing more than 2 groups": [
        "One-Way ANOVA",
    ],
    "Relationships between variables": [
        "Pearson Correlation",
    ],
}


NONPARAMETRIC_TEST_GROUPS = {
    "Comparing 2 groups": [
        "Mann-Whitney U Test",
        "Wilcoxon Signed-Rank Test",
    ],
    "Comparing more than 2 groups": [
        "Kruskal-Wallis Test",
    ],
    "Relationships between variables": [
        "Spearman Correlation",
        "Chi-Square Test",
    ],
}

