"""Descriptive sensory profiling: QDA, temporal (TDS/TI-style) curves, product characterization."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import streamlit as st
from scipy import stats
from statsmodels.formula.api import ols

from ._shared import _columns, _frame, _heatmap, _need, _product_grouping_letters, _product_tests


def _qda(df: pd.DataFrame, title: str = "QDA / Sensory Profile") -> None:
    st.header(title)
    st.write("Summarises product profiles and runs a one-way product ANOVA for each attribute.")
    numeric, _ = _columns(df)
    if not _need(bool(numeric) and df.shape[1] >= 2, "Need a product column and numeric sensory attributes."):
        return
    product = st.selectbox("Product", [str(c) for c in df.columns], key=f"{title}_product")
    attributes = st.multiselect(
        "Attributes", [c for c in numeric if c != product], default=[c for c in numeric if c != product][:6],
        key=f"{title}_attributes",
    )
    if not _need(bool(attributes), "Select at least one attribute."):
        return
    clean = _frame(df, [product, *attributes])
    if not _need(clean[product].nunique() >= 2, "Need at least two products."):
        return
    profile = clean.groupby(product, observed=True)[attributes].agg(["mean", "std", "count"])
    st.subheader("Product summaries")
    st.dataframe(profile, width="stretch")
    tests = _product_tests(clean, product, attributes)
    st.subheader("Inferential product comparisons")
    st.dataframe(tests, width="stretch")
    letters = _product_grouping_letters(clean, product, attributes)
    if not letters.empty:
        st.subheader("Tukey HSD grouping letters")
        st.dataframe(letters, width="stretch")
        st.caption("Within each attribute, products sharing a letter are not significantly different.")
    means = clean.groupby(product, observed=True)[attributes].mean()
    _heatmap(means, "Mean sensory profile")
    significant = tests.loc[tests["p-value"] < 0.05, "Attribute"].tolist() if not tests.empty else []
    st.write(
        f"At the 5% level, product means differ for: **{', '.join(significant)}**."
        if significant else "No selected attribute showed a product difference at the 5% level."
    )


def _temporal(df: pd.DataFrame) -> None:
    st.header("Temporal Sensory Analysis")
    st.write("Displays trajectories over time and tests product, time, and product×time effects.")
    numeric, _ = _columns(df)
    if not _need(len(numeric) >= 2 and df.shape[1] >= 3, "Need time, product, and numeric response columns."):
        return
    product = st.selectbox("Product", df.columns.astype(str).tolist(), key="temporal_product")
    time_candidates = [c for c in numeric if c != product]
    if not _need(bool(time_candidates), "Select a Product column that leaves a numeric Time column available."):
        return
    time = st.selectbox("Time", time_candidates, key="temporal_time")
    response_candidates = [c for c in numeric if c not in {product, time}]
    if not _need(bool(response_candidates), "Select Product/Time columns that leave a numeric response column available."):
        return
    response = st.selectbox("Intensity / dominance score", response_candidates, key="temporal_y")
    clean = _frame(df, [product, time, response])
    summary = clean.groupby([product, time], observed=True)[response].agg(["mean", "std", "count"]).reset_index()
    st.dataframe(summary, width="stretch")
    cell_counts = clean.groupby([product, time], observed=True).size().unstack(fill_value=0)
    if (
        cell_counts.shape[0] >= 2
        and cell_counts.shape[1] >= 2
        and (cell_counts > 0).all().all()
        and len(clean) > cell_counts.size
    ):
        model_data = clean.rename(
            columns={product: "_product", time: "_time", response: "_response"}
        )
        model = ols("_response ~ C(_product) * C(_time)", data=model_data).fit()
        anova = sm.stats.anova_lm(model, typ=2).reset_index()
        anova.columns = ["Effect", *anova.columns[1:]]
        st.subheader("Product × time ANOVA")
        st.dataframe(anova, width="stretch")
        interaction_p = float(
            anova.loc[anova["Effect"] == "C(_product):C(_time)", "PR(>F)"].iloc[0]
        )
        if not np.isfinite(interaction_p):
            st.error("The product × time interaction produced an invalid (non-finite) p-value.")
        elif interaction_p < 0.05:
            st.success("Temporal trajectories differ significantly among products.")
        else:
            st.info("No significant product × time interaction was detected.")
    else:
        st.warning(
            "The product × time ANOVA requires every selected product-time combination; "
            "the trajectory and product-specific trends are still shown."
        )
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.lineplot(data=summary, x=time, y="mean", hue=product, marker="o", ax=ax)
    ax.set(ylabel=f"Mean {response}", title="Temporal sensory curves")
    st.pyplot(fig)
    plt.close(fig)
    trend_rows: list[dict[str, object]] = []
    for product_name, group in clean.groupby(product, observed=True):
        if len(group) >= 3 and group[time].nunique() >= 2:
            trend = stats.linregress(group[time], group[response])
            trend_rows.append(
                {
                    "Product": product_name,
                    "Slope per time unit": trend.slope,
                    "R²": trend.rvalue**2,
                    "Slope p-value": trend.pvalue,
                }
            )
    st.subheader("Product-specific temporal trends")
    st.dataframe(pd.DataFrame(trend_rows), width="stretch")
    st.caption(
        "Positive slopes indicate increasing intensity and negative slopes indicate decay. "
        "Separated or non-parallel curves suggest product-specific temporal evolution."
    )


def _product_characterization(df: pd.DataFrame) -> None:
    _qda(df, "Product Characterization")

