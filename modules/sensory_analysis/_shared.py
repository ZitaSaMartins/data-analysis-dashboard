"""Shared helpers: data prep, significance labelling, and common plots."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def _columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = df.select_dtypes(include=np.number).columns.astype(str).tolist()
    categorical = [str(column) for column in df.columns if str(column) not in numeric]
    return numeric, categorical


def _need(condition: bool, message: str) -> bool:
    if not condition:
        st.error(message)
    return condition


def _frame(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    return df.loc[:, list(columns)].dropna().copy()


def _interpret_p(p_value: float, subject: str = "products") -> None:
    if p_value < 0.05:
        st.success(f"Evidence of differences among {subject} (p={p_value:.4g}).")
    else:
        st.info(f"No statistically significant difference among {subject} (p={p_value:.4g}).")


def _compact_letters(
    labels: Sequence[str],
    significant_pairs: set[tuple[str, str]],
) -> dict[str, str]:
    """Assign letters so statistically different groups share no letter."""
    ordered = [str(label) for label in labels]

    def pair(first: str, second: str) -> tuple[str, str]:
        return tuple(sorted((first, second)))

    compatible = {
        label: {
            other
            for other in ordered
            if other != label and pair(label, other) not in significant_pairs
        }
        for label in ordered
    }
    cliques: list[set[str]] = []

    def find(current: set[str], candidates: set[str], excluded: set[str]) -> None:
        if not candidates and not excluded:
            cliques.append(current)
            return
        remaining = set(candidates)
        pivot = max(candidates | excluded, key=lambda label: len(candidates & compatible[label]))
        for label in candidates - compatible[pivot]:
            find(
                current | {label},
                remaining & compatible[label],
                excluded & compatible[label],
            )
            remaining.remove(label)
            excluded.add(label)

    find(set(), set(ordered), set())
    cliques.sort(key=lambda values: (-len(values), sorted(values)))
    result: dict[str, list[str]] = {label: [] for label in ordered}
    for index, clique in enumerate(cliques):
        letter = chr(65 + index) if index < 26 else f"A{index - 25}"
        for label in clique:
            result[label].append(letter)
    return {label: "".join(result[label]) or "—" for label in ordered}


def _heatmap(data: pd.DataFrame, title: str, fmt: str = ".2f") -> None:
    width = max(7.0, 0.75 * data.shape[1] + 3)
    height = max(4.0, 0.45 * data.shape[0] + 2)
    fig, ax = plt.subplots(figsize=(width, height))
    sns.heatmap(data, annot=True, fmt=fmt, cmap="RdYlBu_r", ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _scatter_labels(
    coordinates: np.ndarray,
    labels: Sequence[object],
    title: str,
    xlabel: str = "Dimension 1",
    ylabel: str = "Dimension 2",
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(coordinates[:, 0], coordinates[:, 1], s=55)
    for label, (x_value, y_value) in zip(labels, coordinates[:, :2]):
        ax.annotate(str(label), (x_value, y_value), xytext=(4, 4), textcoords="offset points")
    ax.axhline(0, color="grey", linewidth=0.7)
    ax.axvline(0, color="grey", linewidth=0.7)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _product_tests(data: pd.DataFrame, product: str, attributes: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for attribute in attributes:
        groups = [
            group[attribute].dropna().to_numpy()
            for _, group in data.groupby(product, observed=True)
            if group[attribute].notna().sum() > 1
        ]
        if len(groups) < 2:
            continue
        statistic, p_value = stats.f_oneway(*groups)
        rows.append(
            {
                "Attribute": attribute,
                "F": float(statistic),
                "p-value": float(p_value),
                "Significant (5%)": "Yes" if p_value < 0.05 else "No",
                "Groups": len(groups),
            }
        )
    return pd.DataFrame(rows)


def _product_grouping_letters(
    data: pd.DataFrame,
    product: str,
    attributes: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for attribute in attributes:
        clean = data[[product, attribute]].dropna()
        counts = clean.groupby(product, observed=True).size()
        valid_products = counts[counts >= 2].index
        clean = clean[clean[product].isin(valid_products)]
        if clean[product].nunique() < 2:
            continue
        labels = clean[product].astype(str).to_numpy()
        tukey = pairwise_tukeyhsd(clean[attribute].to_numpy(), labels, alpha=0.05)
        first_indices, second_indices = tukey._multicomp.pairindices
        significant_pairs: set[tuple[str, str]] = set()
        for index, (first_index, second_index) in enumerate(zip(first_indices, second_indices)):
            if tukey.reject[index]:
                significant_pairs.add(
                    tuple(
                        sorted(
                            (
                                str(tukey._multicomp.groupsunique[first_index]),
                                str(tukey._multicomp.groupsunique[second_index]),
                            )
                        )
                    )
                )
        products = [str(value) for value in valid_products]
        letters = _compact_letters(products, significant_pairs)
        means = clean.groupby(product, observed=True)[attribute].mean()
        rows.extend(
            {
                "Attribute": attribute,
                "Product": str(product_name),
                "Mean": float(mean),
                "Grouping letters": letters[str(product_name)],
            }
            for product_name, mean in means.items()
        )
    return pd.DataFrame(rows)

