"""Hedonic (liking) comparisons, consumer segmentation, and preference mapping."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests

from ._shared import _columns, _compact_letters, _frame, _heatmap, _interpret_p, _need, _scatter_labels


def _hedonic(df: pd.DataFrame) -> None:
    st.header("Hedonic Product Comparison")
    st.write("Compares consumer liking, using Friedman for complete repeated ratings or one-way ANOVA otherwise.")
    numeric, _ = _columns(df)
    columns = df.columns.astype(str).tolist()
    if not _need(bool(numeric) and len(columns) >= 2, "Need product and numeric liking columns."):
        return
    product = st.selectbox("Product", columns, key="hed_product")
    liking = st.selectbox("Liking", [c for c in numeric if c != product], key="hed_liking")
    consumer = st.selectbox("Consumer (optional)", ["(independent observations)", *[c for c in columns if c not in {product, liking}]],
                            key="hed_consumer")
    clean = _frame(df, [product, liking] + ([] if consumer.startswith("(") else [consumer]))
    summary = clean.groupby(product, observed=True)[liking].agg(["mean", "std", "median", "count"])
    pairwise_rows: list[dict[str, object]] = []
    significant_pairs: set[tuple[str, str]] = set()
    if consumer.startswith("("):
        grouped = {
            str(name): group[liking].to_numpy()
            for name, group in clean.groupby(product, observed=True)
            if len(group) > 1
        }
        if not _need(len(grouped) >= 2, "Need at least two products with two liking scores each."):
            return
        statistic, p_value = stats.f_oneway(*grouped.values())
        method = "One-way ANOVA"
        values = np.concatenate(list(grouped.values()))
        labels = np.concatenate([[name] * len(group) for name, group in grouped.items()])
        tukey = pairwise_tukeyhsd(values, labels, alpha=0.05)
        first_indices, second_indices = tukey._multicomp.pairindices
        for index, (first_index, second_index) in enumerate(zip(first_indices, second_indices)):
            first = str(tukey._multicomp.groupsunique[first_index])
            second = str(tukey._multicomp.groupsunique[second_index])
            significant = bool(tukey.reject[index])
            pairwise_rows.append(
                {
                    "Product 1": first,
                    "Product 2": second,
                    "Mean difference": float(tukey.meandiffs[index]),
                    "Adjusted p-value": float(tukey.pvalues[index]),
                    "Significant (5%)": significant,
                }
            )
            if significant:
                significant_pairs.add(tuple(sorted((first, second))))
        posthoc_method = "Tukey HSD"
    else:
        paired = clean.pivot_table(index=consumer, columns=product, values=liking, aggfunc="mean").dropna()
        if not _need(paired.shape[0] >= 2 and paired.shape[1] >= 3, "Friedman needs 2 consumers rating 3 products."):
            return
        statistic, p_value = stats.friedmanchisquare(*(paired[c] for c in paired))
        method = "Friedman"
        metadata: list[tuple[str, str, float]] = []
        raw_p_values: list[float] = []
        for first_index, first in enumerate(paired.columns):
            for second in paired.columns[first_index + 1 :]:
                difference = paired[first] - paired[second]
                _, pair_p = (0.0, 1.0) if np.allclose(difference, 0) else stats.wilcoxon(difference)
                raw_p_values.append(float(pair_p))
                metadata.append((str(first), str(second), float(difference.mean())))
        adjusted = multipletests(raw_p_values, alpha=0.05, method="holm")[1]
        for (first, second, difference), adjusted_p in zip(metadata, adjusted):
            significant = bool(adjusted_p < 0.05)
            pairwise_rows.append(
                {
                    "Product 1": first,
                    "Product 2": second,
                    "Mean difference": difference,
                    "Adjusted p-value": adjusted_p,
                    "Significant (5%)": significant,
                }
            )
            if significant:
                significant_pairs.add(tuple(sorted((first, second))))
        posthoc_method = "Holm-adjusted Wilcoxon"
    st.write(f"{method}: statistic = **{statistic:.3f}**, p = **{p_value:.4g}**")
    _interpret_p(float(p_value))
    letters = _compact_letters([str(value) for value in summary.index], significant_pairs)
    summary["Grouping letters"] = [letters[str(value)] for value in summary.index]
    st.dataframe(summary, width="stretch")
    st.subheader(f"Pairwise comparisons: {posthoc_method}")
    st.dataframe(pd.DataFrame(pairwise_rows), width="stretch")
    st.caption("Products sharing a grouping letter are not significantly different.")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=clean, x=product, y=liking, ax=ax)
    ax.tick_params(axis="x", rotation=35)
    st.pyplot(fig)
    plt.close(fig)


def _consumer_matrix(df: pd.DataFrame, prefix: str) -> tuple[pd.DataFrame, str, str, str] | None:
    numeric, _ = _columns(df)
    columns = df.columns.astype(str).tolist()
    if not _need(bool(numeric) and len(columns) >= 3, "Need consumer, product, and numeric liking."):
        return None
    consumer = st.selectbox("Consumer", columns, key=f"{prefix}_consumer")
    product = st.selectbox("Product", [c for c in columns if c != consumer], key=f"{prefix}_product")
    liking = st.selectbox("Liking", [c for c in numeric if c not in {consumer, product}], key=f"{prefix}_liking")
    matrix = df.pivot_table(index=consumer, columns=product, values=liking, aggfunc="mean").dropna()
    if not _need(matrix.shape[0] >= 3 and matrix.shape[1] >= 2, "Need at least 3 complete consumers and 2 products."):
        return None
    return matrix, consumer, product, liking


def _segmentation(df: pd.DataFrame) -> None:
    st.header("Consumer Segmentation")
    st.write("K-means groups consumers with similar product-liking patterns.")
    result = _consumer_matrix(df, "segment")
    if result is None:
        return
    matrix, _, _, _ = result
    maximum = min(8, matrix.shape[0] - 1)
    clusters = st.slider("Number of segments", 2, maximum, min(3, maximum), key="segment_k")
    scaled = StandardScaler().fit_transform(matrix)
    model = KMeans(n_clusters=clusters, n_init=20, random_state=42).fit(scaled)
    assignments = pd.DataFrame({"Segment": model.labels_ + 1}, index=matrix.index)
    profiles = matrix.assign(Segment=model.labels_ + 1).groupby("Segment").mean()
    st.dataframe(assignments, width="stretch")
    st.subheader("Segment liking profiles")
    st.dataframe(profiles, width="stretch")
    _heatmap(profiles, "Mean liking by segment")
    st.caption(f"Within-cluster sum of squares: {model.inertia_:.2f}. Segments describe patterns, not quality levels.")


def _internal_preference(df: pd.DataFrame) -> None:
    st.header("Internal Preference Map")
    st.write("PCA maps products from consumer liking alone; consumer vectors indicate preference directions.")
    result = _consumer_matrix(df, "internal")
    if result is None:
        return
    matrix, _, _, _ = result
    product_by_consumer = matrix.T
    scaled = StandardScaler().fit_transform(product_by_consumer)
    if not _need(min(scaled.shape) >= 2, "Need at least two products and two consumers."):
        return
    pca = PCA(n_components=2).fit(scaled)
    scores = pca.transform(scaled)
    _scatter_labels(scores, product_by_consumer.index, "Internal preference map",
                    f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
                    f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    loadings = pd.DataFrame(pca.components_.T, index=product_by_consumer.columns, columns=["PC1", "PC2"])
    st.dataframe(loadings, width="stretch")
    st.caption("Consumers with similar vectors prefer similar products; products near a vector's direction are preferred.")


def _external_preference(df: pd.DataFrame) -> None:
    st.header("External Preference Map")
    st.write("PLS relates sensory descriptors to liking and maps products in sensory latent space.")
    numeric, _ = _columns(df)
    if not _need(len(numeric) >= 3, "Need product, liking, and at least two sensory descriptor columns."):
        return
    product = st.selectbox("Product", df.columns.astype(str).tolist(), key="external_product")
    liking = st.selectbox("Liking", [c for c in numeric if c != product], key="external_liking")
    descriptors = st.multiselect("Sensory descriptors", [c for c in numeric if c not in {product, liking}],
                                 default=[c for c in numeric if c not in {product, liking}][:6], key="external_x")
    means = df.groupby(product, observed=True)[[*descriptors, liking]].mean().dropna()
    if not _need(len(descriptors) >= 2 and means.shape[0] >= 3, "Need 2 descriptors and 3 products with complete means."):
        return
    x_scaled = StandardScaler().fit_transform(means[descriptors])
    model = PLSRegression(n_components=min(2, len(descriptors), means.shape[0] - 1)).fit(x_scaled, means[[liking]])
    if model.x_scores_.shape[1] == 1:
        coordinates = np.column_stack((model.x_scores_[:, 0], np.zeros(len(means))))
    else:
        coordinates = model.x_scores_[:, :2]
    _scatter_labels(coordinates, means.index, "External preference map", "PLS 1", "PLS 2")
    coefficients = pd.DataFrame({"Descriptor": descriptors, "Liking coefficient": model.coef_.reshape(-1)})
    st.dataframe(coefficients.reindex(coefficients["Liking coefficient"].abs().sort_values(ascending=False).index),
                 width="stretch")
    st.caption("Positive coefficients describe sensory properties associated with higher mean liking.")

