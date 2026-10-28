"""Discrimination and ranking tests: triangle, paired, duo-trio, tetrad, ranking."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy import stats
from statsmodels.stats.multitest import multipletests

from ._shared import _columns, _compact_letters, _interpret_p, _need


def _discrimination(df: pd.DataFrame, name: str, chance: float, key: str) -> None:
    st.header(name)
    st.write(
        f"Tests whether the number of correct responses exceeds chance ({chance:.3g}) "
        "using an exact one-sided binomial test."
    )
    if not _need(df.shape[1] > 0 and len(df) > 0, "Upload response-level data first."):
        return
    outcome = st.selectbox("Response / correct column", df.columns.tolist(), key=f"{key}_outcome")
    values = df[outcome].dropna().unique().tolist()
    if not _need(len(values) >= 1, "The response column must contain at least one observed outcome."):
        return
    if len(values) == 1:
        marks_correct = st.radio(
            f"Does the value '{values[0]}' represent a correct response?",
            ["Yes", "No"],
            key=f"{key}_single_outcome",
            horizontal=True,
        )
        correct_value = values[0] if marks_correct == "Yes" else object()
    else:
        correct_value = st.selectbox("Value meaning correct", values, key=f"{key}_correct")
    valid = df[outcome].notna()
    successes = int((df.loc[valid, outcome] == correct_value).sum())
    total = int(valid.sum())
    result = stats.binomtest(successes, total, chance, alternative="greater")
    critical_correct = next(
        (
            count
            for count in range(total + 1)
            if stats.binomtest(count, total, chance, alternative="greater").pvalue < 0.05
        ),
        total + 1,
    )
    critical_display: int | str = (
        critical_correct if critical_correct <= total else "Not attainable"
    )
    st.dataframe(
        pd.DataFrame(
            {
                "Correct": [successes],
                "Total": [total],
                "Observed proportion": [successes / total],
                "Chance proportion": [chance],
                "Critical correct (α=0.05)": [critical_display],
                "Exact p-value": [result.pvalue],
                "95% CI low": [result.proportion_ci().low],
                "95% CI high": [result.proportion_ci().high],
            }
        ),
        width="stretch",
    )
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(["Observed", "Chance"], [successes / total, chance], color=["#2a9d8f", "#bdbdbd"])
    ax.set(ylim=(0, 1), ylabel="Correct response proportion", title=name)
    st.pyplot(fig)
    plt.close(fig)
    if result.pvalue < 0.05:
        st.success("The panel discriminated above chance.")
    else:
        st.info("The observed correct rate is not convincingly above chance.")


def _ranking(df: pd.DataFrame) -> None:
    st.header("Sensory Ranking Test")
    st.write("Friedman's test compares product ranks while accounting for each assessor.")
    numeric, _ = _columns(df)
    if not _need(bool(numeric) and df.shape[1] >= 3, "Need assessor, product, and numeric rank columns."):
        return
    columns = df.columns.astype(str).tolist()
    assessor = st.selectbox("Assessor", columns, key="rank_assessor")
    product = st.selectbox("Product", [c for c in columns if c != assessor], key="rank_product")
    rank_candidates = [c for c in numeric if c not in {assessor, product}]
    if not _need(bool(rank_candidates), "Select Assessor/Product columns that leave a numeric Rank column available."):
        return
    rank = st.selectbox("Rank", rank_candidates, key="rank_value")
    pivot = df.pivot_table(index=assessor, columns=product, values=rank, aggfunc="mean").dropna()
    if not _need(pivot.shape[0] >= 2 and pivot.shape[1] >= 3, "Need at least 2 complete assessors and 3 products."):
        return
    statistic, p_value = stats.friedmanchisquare(*(pivot[column] for column in pivot))
    summary = pd.DataFrame({"Mean rank": pivot.mean(), "Median rank": pivot.median()}).sort_values("Mean rank")
    st.write(f"Friedman χ²({pivot.shape[1] - 1}) = **{statistic:.3f}**, p = **{p_value:.4g}**")
    _interpret_p(float(p_value))

    pair_metadata: list[tuple[str, str, float, float]] = []
    raw_p_values: list[float] = []
    products = [str(column) for column in pivot.columns]
    for first_index, first in enumerate(pivot.columns):
        for second in pivot.columns[first_index + 1 :]:
            difference = pivot[first] - pivot[second]
            if np.allclose(difference, 0):
                pair_statistic, pair_p = 0.0, 1.0
            else:
                pair_statistic, pair_p = stats.wilcoxon(difference)
            raw_p_values.append(float(pair_p))
            pair_metadata.append(
                (str(first), str(second), float(pair_statistic), float(difference.mean()))
            )
    adjusted = multipletests(raw_p_values, alpha=0.05, method="holm")[1]
    significant_pairs = {
        tuple(sorted((first, second)))
        for (first, second, _, _), adjusted_p in zip(pair_metadata, adjusted)
        if adjusted_p < 0.05
    }
    letters = _compact_letters(products, significant_pairs)
    summary["Grouping letters"] = [letters[str(product_name)] for product_name in summary.index]
    st.subheader("Holm-adjusted pairwise Wilcoxon comparisons")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Product 1": first,
                    "Product 2": second,
                    "W": pair_statistic,
                    "Mean rank difference": mean_difference,
                    "Adjusted p-value": adjusted_p,
                    "Significant (5%)": adjusted_p < 0.05,
                }
                for (first, second, pair_statistic, mean_difference), adjusted_p in zip(
                    pair_metadata, adjusted
                )
            ]
        ),
        width="stretch",
    )
    st.dataframe(summary, width="stretch")
    st.caption("Products sharing a grouping letter are not significantly different.")
    fig, ax = plt.subplots(figsize=(8, 4))
    summary["Mean rank"].plot.bar(ax=ax, color="#457b9d")
    ax.set(ylabel="Mean rank", title="Lower mean rank indicates the more highly ranked product")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

