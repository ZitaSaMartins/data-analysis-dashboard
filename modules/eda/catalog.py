"""Chart catalog metadata: what chart types exist and when they apply."""

import numpy as np
import pandas as pd


CHART_CATALOG = {
    "Comparison": {
        "Bar Chart": {"icon": "📶", "desc": "Compare a numeric value across categories.",
                      "requires": lambda n, c: len(n) >= 1 and len(c) >= 1},
        "Grouped Bar Chart": {"icon": "📊", "desc": "Compare several numeric values across categories, side by side.",
                              "requires": lambda n, c: len(n) >= 2 and len(c) >= 1},
        "Radar Chart": {"icon": "🕸️", "desc": "Compare several numeric variables on a radial axis.",
                        "requires": lambda n, c: len(n) >= 3},
    },
    "Correlation": {
        "Scatter Plot": {"icon": "⚡", "desc": "Relationship between two numeric variables.",
                         "requires": lambda n, c: len(n) >= 2},
        "Bubble Chart": {"icon": "🫧", "desc": "Scatter plot with a third variable encoded as bubble size.",
                         "requires": lambda n, c: len(n) >= 3},
        "Correlation Heatmap": {"icon": "🔗", "desc": "Which numeric variables move together.",
                                "requires": lambda n, c: len(n) >= 2},
        "Contour Plot": {"icon": "🌄", "desc": "Density of points between two numeric variables.",
                         "requires": lambda n, c: len(n) >= 2},
    },
    "Part-to-Whole & Hierarchical": {
        "Pie Chart": {"icon": "🥧", "desc": "Share of each category out of the whole.",
                      "requires": lambda n, c: len(c) >= 1},
        "Donut Chart": {"icon": "🍩", "desc": "Pie chart with a hollow center.",
                        "requires": lambda n, c: len(c) >= 1},
        "Treemap": {"icon": "🌳", "desc": "Nested rectangles sized by value, grouped by category.",
                    "requires": lambda n, c: len(c) >= 1},
        "Sunburst": {"icon": "☀️", "desc": "Nested rings showing hierarchy, like a radial treemap.",
                     "requires": lambda n, c: len(c) >= 1},
    },
    "Temporal": {
        "Line Chart": {"icon": "📈", "desc": "How a numeric value changes across a sequence.",
                       "requires": lambda n, c: len(n) >= 1},
        "Area Chart": {"icon": "🏔️", "desc": "Line chart with the area under the line filled in.",
                       "requires": lambda n, c: len(n) >= 1},
        "Stacked Area Chart": {"icon": "🗻", "desc": "Area chart split and stacked by category.",
                               "requires": lambda n, c: len(n) >= 1 and len(c) >= 1},
    },
    "Distribution": {
        "Histogram": {"icon": "📊", "desc": "How values are spread out, in bins.",
                      "requires": lambda n, c: len(n) >= 1},
        "Categorical Bar": {"icon": "🗂️", "desc": "Frequency of each category.",
                            "requires": lambda n, c: len(c) >= 1},
        "Box Plot": {"icon": "📦", "desc": "Median, quartiles & outliers of a numeric variable by category.",
                     "requires": lambda n, c: len(n) >= 1 and len(c) >= 1},
        "Violin Plot": {"icon": "🎻", "desc": "Box plot combined with a density curve.",
                        "requires": lambda n, c: len(n) >= 1 and len(c) >= 1},
        "Strip Plot": {"icon": "🔸", "desc": "Every individual point, jittered by category.",
                       "requires": lambda n, c: len(n) >= 1 and len(c) >= 1},
    },
}


CHART_INFO = {name: info for charts in CHART_CATALOG.values() for name, info in charts.items()}
CHART_INFO["Summary Statistics"] = {"icon": "📋", "desc": "Means, std dev, skewness & kurtosis as a table."}


def recommend_components(df: pd.DataFrame, variables: list = None) -> dict:
    """Flag which chart types are a good fit for the currently selected variables."""
    cols = variables if variables else df.columns.tolist()
    data = df[cols]
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

    result = {"Summary Statistics": len(numeric_cols) >= 1}
    for charts in CHART_CATALOG.values():
        for name, info in charts.items():
            result[name] = info["requires"](numeric_cols, cat_cols)
    return result

