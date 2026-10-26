"""Interactive sensory and consumer-science workflows."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy import linalg, stats
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.contingency_tables import cochrans_q, mcnemar
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests


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
        for label in sorted(candidates):
            if label not in remaining:
                continue
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
    if not _need(len(values) >= 2, "The response column must contain correct and incorrect outcomes."):
        return
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
    rank = st.selectbox("Rank", [c for c in numeric if c not in {assessor, product}], key="rank_value")
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
    time = st.selectbox("Time", [c for c in numeric if c != product], key="temporal_time")
    response = st.selectbox("Intensity / dominance score", [c for c in numeric if c not in {product, time}], key="temporal_y")
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
        if interaction_p < 0.05:
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


def _panel_discrimination(df: pd.DataFrame) -> None:
    st.header("Panel Discrimination")
    st.write("For each assessor and attribute, a one-way ANOVA measures separation among products.")
    numeric, _ = _columns(df)
    columns = df.columns.astype(str).tolist()
    if not _need(bool(numeric) and len(columns) >= 3, "Need assessor, product, and numeric attributes."):
        return
    panelist = st.selectbox("Assessor", columns, key="pd_panel")
    product = st.selectbox("Product", [c for c in columns if c != panelist], key="pd_product")
    attributes = st.multiselect("Attributes", [c for c in numeric if c not in {panelist, product}],
                                default=[c for c in numeric if c not in {panelist, product}][:4], key="pd_attr")
    rows: list[dict[str, object]] = []
    for (judge, attribute), group in (
        df.melt(id_vars=[panelist, product], value_vars=attributes, var_name="Attribute", value_name="Rating")
        .dropna()
        .groupby([panelist, "Attribute"], observed=True)
    ):
        samples = [values["Rating"].to_numpy() for _, values in group.groupby(product) if len(values) > 1]
        if len(samples) >= 2:
            statistic, p_value = stats.f_oneway(*samples)
            rows.append({"Assessor": judge, "Attribute": attribute, "F": statistic, "p-value": p_value})
    result = pd.DataFrame(rows)
    if not _need(not result.empty, "Need replicated ratings for at least two products per assessor."):
        return
    result["Discriminates (5%)"] = result["p-value"] < 0.05
    st.dataframe(result, width="stretch")
    _heatmap(result.pivot(index="Assessor", columns="Attribute", values="F").fillna(0), "Product discrimination (F)")
    st.caption("Larger F statistics and p-values below 0.05 indicate stronger product discrimination.")


def _panel_repeatability(df: pd.DataFrame) -> None:
    st.header("Panel Repeatability")
    st.write("Within-assessor/product variation across replicates quantifies repeatability.")
    numeric, _ = _columns(df)
    columns = df.columns.astype(str).tolist()
    if not _need(len(columns) >= 4 and bool(numeric), "Need assessor, product, replicate, and rating columns."):
        return
    panelist = st.selectbox("Assessor", columns, key="pr_panel")
    product = st.selectbox("Product", [c for c in columns if c != panelist], key="pr_product")
    replicate = st.selectbox("Replicate", [c for c in columns if c not in {panelist, product}], key="pr_rep")
    attributes = st.multiselect("Attributes", [c for c in numeric if c not in {panelist, product, replicate}],
                                default=[c for c in numeric if c not in {panelist, product, replicate}][:4], key="pr_attr")
    long = df.melt(id_vars=[panelist, product, replicate], value_vars=attributes,
                   var_name="Attribute", value_name="Rating").dropna()
    within = long.groupby([panelist, product, "Attribute"], observed=True)["Rating"].agg(["std", "count"]).reset_index()
    within = within[within["count"] > 1]
    if not _need(not within.empty, "No repeated assessor-product ratings were found."):
        return
    metrics = within.groupby([panelist, "Attribute"], observed=True)["std"].agg(["mean", "median", "max"]).reset_index()
    metrics.columns = [panelist, "Attribute", "Mean within SD", "Median within SD", "Worst within SD"]
    st.dataframe(metrics, width="stretch")
    _heatmap(metrics.pivot(index=panelist, columns="Attribute", values="Mean within SD"),
             "Repeatability: lower within-cell SD is better")
    st.caption("Assessors with the smallest within-product SD reproduce their scores most consistently.")


def _panel_agreement(df: pd.DataFrame) -> None:
    st.header("Panel Agreement")
    st.write("Kendall's W measures how similarly assessors order products; correlations show pairwise agreement.")
    numeric, _ = _columns(df)
    columns = df.columns.astype(str).tolist()
    if not _need(bool(numeric) and len(columns) >= 3, "Need assessor, product, and numeric rating."):
        return
    panelist = st.selectbox("Assessor", columns, key="pa_panel")
    product = st.selectbox("Product", [c for c in columns if c != panelist], key="pa_product")
    rating = st.selectbox("Rating", [c for c in numeric if c not in {panelist, product}], key="pa_rating")
    matrix = df.pivot_table(index=product, columns=panelist, values=rating, aggfunc="mean").dropna()
    if not _need(matrix.shape[0] >= 3 and matrix.shape[1] >= 2, "Need 3 common products rated by at least 2 assessors."):
        return
    ranks = matrix.rank(axis=0)
    products, judges = ranks.shape
    rank_sums = ranks.sum(axis=1)
    tie_correction = sum(
        sum(count**3 - count for count in column.value_counts())
        for _, column in matrix.items()
    )
    denominator = judges**2 * (products**3 - products) - judges * tie_correction
    w_value = 12 * float(((rank_sums - rank_sums.mean()) ** 2).sum()) / denominator if denominator > 0 else np.nan
    chi_square = judges * (products - 1) * w_value
    p_value = float(stats.chi2.sf(chi_square, products - 1))
    st.metric("Kendall's W", f"{w_value:.3f}")
    st.write(f"χ²({products - 1}) = {chi_square:.3f}, p = {p_value:.4g}")
    _heatmap(matrix.corr(method="spearman"), "Pairwise assessor Spearman correlations")
    st.caption("W ranges from 0 (no common ordering) to 1 (complete agreement).")


def _panel_scale(df: pd.DataFrame) -> None:
    st.header("Panel Scale Usage")
    st.write("Compares assessor location, dispersion, and range to identify scale-use differences.")
    numeric, _ = _columns(df)
    if not _need(bool(numeric) and df.shape[1] >= 2, "Need an assessor column and numeric ratings."):
        return
    panelist = st.selectbox("Assessor", df.columns.astype(str).tolist(), key="ps_panel")
    attributes = st.multiselect("Attributes", [c for c in numeric if c != panelist],
                                default=[c for c in numeric if c != panelist][:5], key="ps_attr")
    long = df.melt(id_vars=panelist, value_vars=attributes, var_name="Attribute", value_name="Rating").dropna()
    metrics = long.groupby([panelist, "Attribute"], observed=True)["Rating"].agg(["mean", "std", "min", "max"]).reset_index()
    metrics["Range"] = metrics["max"] - metrics["min"]
    metrics["Centered mean"] = metrics["mean"] - metrics.groupby("Attribute")["mean"].transform("mean")
    st.dataframe(metrics, width="stretch")
    _heatmap(metrics.pivot(index=panelist, columns="Attribute", values="Range"), "Scale range used")
    st.caption("Narrow ranges suggest scale compression; centered means reveal consistently high or low scoring.")


def _cata(df: pd.DataFrame) -> None:
    st.header("Check-All-That-Apply (CATA)")
    st.write("Attribute frequencies describe products; Cochran's Q tests paired product differences.")
    numeric, _ = _columns(df)
    columns = df.columns.astype(str).tolist()
    if not _need(bool(numeric) and len(columns) >= 3, "Need consumer, product, and binary attribute columns."):
        return
    consumer = st.selectbox("Consumer", columns, key="cata_consumer")
    product = st.selectbox("Product", [c for c in columns if c != consumer], key="cata_product")
    candidates = [
        c for c in numeric if c not in {consumer, product}
        and set(df[c].dropna().unique()).issubset({0, 1, False, True})
    ]
    attributes = st.multiselect("Binary CATA attributes", candidates, default=candidates[:8], key="cata_attr")
    if not _need(bool(attributes), "Select binary 0/1 attribute columns."):
        return
    frequencies = df.groupby(product, observed=True)[attributes].mean() * 100
    st.subheader("Selection frequency (%)")
    st.dataframe(frequencies, width="stretch")
    q_rows: list[dict[str, object]] = []
    for attribute in attributes:
        paired = df.pivot_table(index=consumer, columns=product, values=attribute, aggfunc="max").dropna()
        if paired.shape[0] >= 2 and paired.shape[1] >= 2:
            result = cochrans_q(paired.to_numpy(dtype=int))
            q_rows.append({"Attribute": attribute, "Q": result.statistic, "df": paired.shape[1] - 1,
                           "p-value": result.pvalue})
    q_table = pd.DataFrame(q_rows)
    if not q_table.empty:
        q_table["Significant (5%)"] = q_table["p-value"] < 0.05
        st.subheader("Cochran's Q by attribute")
        st.dataframe(q_table, width="stretch")

    pairwise_rows: list[dict[str, object]] = []
    for attribute in attributes:
        paired = df.pivot_table(
            index=consumer,
            columns=product,
            values=attribute,
            aggfunc="max",
        ).dropna()
        metadata: list[tuple[str, str, float]] = []
        raw_p_values: list[float] = []
        for first_index, first in enumerate(paired.columns):
            for second in paired.columns[first_index + 1 :]:
                table = pd.crosstab(paired[first], paired[second]).reindex(
                    index=[0, 1], columns=[0, 1], fill_value=0
                )
                discordant = int(table.iloc[0, 1] + table.iloc[1, 0])
                result = mcnemar(
                    table.to_numpy(),
                    exact=discordant <= 25,
                    correction=discordant > 25,
                )
                raw_p_values.append(float(result.pvalue))
                metadata.append(
                    (
                        str(first),
                        str(second),
                        float(paired[first].mean() - paired[second].mean()),
                    )
                )
        if raw_p_values:
            adjusted = multipletests(raw_p_values, alpha=0.05, method="holm")[1]
            pairwise_rows.extend(
                {
                    "Attribute": attribute,
                    "Product 1": first,
                    "Product 2": second,
                    "Citation difference": difference,
                    "Adjusted p-value": adjusted_p,
                    "Significant (5%)": adjusted_p < 0.05,
                }
                for (first, second, difference), adjusted_p in zip(metadata, adjusted)
            )
    if pairwise_rows:
        st.subheader("Holm-adjusted pairwise McNemar comparisons")
        st.dataframe(pd.DataFrame(pairwise_rows), width="stretch")
    scaled = StandardScaler().fit_transform(frequencies)
    if min(frequencies.shape) >= 2:
        pca = PCA(n_components=2).fit(scaled)
        coords = pca.transform(scaled)
        _scatter_labels(coords, frequencies.index, "CATA product correspondence/PCA map",
                        f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
                        f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    _heatmap(frequencies, "CATA attribute frequencies", ".1f")
    st.caption("Nearby products have similar attribute citation profiles; significant Q tests vary by product.")


def _jar(df: pd.DataFrame) -> None:
    st.header("Just-About-Right (JAR) Penalty Analysis")
    st.write("Compares liking among low, JAR, and high ratings and weights the drop by group prevalence.")
    numeric, _ = _columns(df)
    if not _need(len(numeric) >= 2, "Need liking and at least one JAR rating column."):
        return
    product_options = ["(all products)", *df.columns.astype(str).tolist()]
    product = st.selectbox("Product column (optional)", product_options, key="jar_product")
    liking = st.selectbox("Overall liking", numeric, key="jar_liking")
    jar_attributes = st.multiselect("JAR attributes", [c for c in numeric if c != liking],
                                    default=[c for c in numeric if c != liking][:4], key="jar_attr")
    target = st.number_input("JAR midpoint", value=3.0, key="jar_target")
    minimum_consumers = st.number_input(
        "Minimum non-JAR consumers (%)",
        min_value=0.0,
        max_value=100.0,
        value=20.0,
        key="jar_minimum_consumers",
    )
    minimum_drop = st.number_input(
        "Minimum mean liking drop",
        min_value=0.0,
        value=1.0,
        key="jar_minimum_drop",
    )
    data = df
    group_columns = [] if product == "(all products)" else [product]
    rows: list[dict[str, object]] = []
    random = np.random.default_rng(42)
    for group_key, group in [(None, data)] if not group_columns else data.groupby(group_columns[0], observed=True):
        base = str(group_key) if group_key is not None else "All"
        for attribute in jar_attributes:
            clean = group[[liking, attribute]].dropna()
            categories = np.where(clean[attribute] < target, "Too low",
                                  np.where(clean[attribute] > target, "Too high", "JAR"))
            jar_values = clean.loc[categories == "JAR", liking].to_numpy(dtype=float)
            for direction in ("Too low", "Too high"):
                mask = categories == direction
                direction_values = clean.loc[mask, liking].to_numpy(dtype=float)
                if len(jar_values) == 0 or len(direction_values) == 0:
                    continue
                prevalence = float(mask.mean())
                mean_drop = float(jar_values.mean() - direction_values.mean())
                bootstrap_drops = np.array(
                    [
                        random.choice(jar_values, len(jar_values), replace=True).mean()
                        - random.choice(
                            direction_values,
                            len(direction_values),
                            replace=True,
                        ).mean()
                        for _ in range(500)
                    ]
                )
                rows.append(
                    {
                        "Product": base,
                        "Attribute": attribute,
                        "Direction": direction,
                        "Consumers (%)": 100 * prevalence,
                        "Mean drop": mean_drop,
                        "95% CI low": float(np.quantile(bootstrap_drops, 0.025)),
                        "95% CI high": float(np.quantile(bootstrap_drops, 0.975)),
                        "Weighted penalty": prevalence * mean_drop,
                        "Priority": (
                            100 * prevalence >= minimum_consumers
                            and mean_drop >= minimum_drop
                        ),
                    }
                )
    result = pd.DataFrame(rows).dropna()
    if not _need(not result.empty, "Need observations in the JAR and non-JAR groups."):
        return
    result = result.sort_values("Weighted penalty", ascending=False)
    st.dataframe(result, width="stretch")
    priorities = result.loc[result["Priority"], ["Product", "Attribute", "Direction"]]
    if priorities.empty:
        st.info("No attribute-direction combination exceeded both priority thresholds.")
    else:
        st.warning(
            f"{len(priorities)} attribute-direction combination(s) exceeded both "
            "priority thresholds."
        )
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.scatterplot(data=result, x="Consumers (%)", y="Mean drop", hue="Attribute", style="Direction", s=90, ax=ax)
    ax.axvline(20, color="grey", linestyle="--")
    ax.set(title="JAR penalty plot")
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Large positive drops affecting many consumers are the strongest reformulation priorities.")


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


def _pca(df: pd.DataFrame) -> None:
    st.header("Principal Component Analysis")
    numeric, _ = _columns(df)
    if not _need(len(numeric) >= 2, "Need at least two numeric variables."):
        return
    label = st.selectbox("Point label", ["(row number)", *df.columns.astype(str).tolist()], key="pca_label")
    features = st.multiselect("Numeric variables", numeric, default=numeric[:8], key="pca_features")
    clean = df[features].dropna()
    if not _need(len(features) >= 2 and clean.shape[0] >= 3, "Select 2 variables with 3 complete rows."):
        return
    pca = PCA(n_components=2).fit(StandardScaler().fit_transform(clean))
    scores = pca.transform(StandardScaler().fit_transform(clean))
    labels = clean.index if label == "(row number)" else df.loc[clean.index, label]
    _scatter_labels(scores, labels, "PCA scores",
                    f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
                    f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    st.dataframe(pd.DataFrame(pca.components_.T, index=features, columns=["PC1 loading", "PC2 loading"]),
                 width="stretch")
    st.caption("Nearby observations have similar standardized profiles; large loadings drive separation.")


def _multiassessor_inputs(df: pd.DataFrame, prefix: str) -> tuple[str, str, list[str]] | None:
    numeric, _ = _columns(df)
    columns = df.columns.astype(str).tolist()
    if not _need(bool(numeric) and len(columns) >= 3, "Need assessor/block, product/object, and numeric attributes."):
        return None
    assessor = st.selectbox("Assessor / table", columns, key=f"{prefix}_assessor")
    product = st.selectbox("Product / object", [c for c in columns if c != assessor], key=f"{prefix}_product")
    attributes = st.multiselect("Common attributes", [c for c in numeric if c not in {assessor, product}],
                                default=[c for c in numeric if c not in {assessor, product}][:6],
                                key=f"{prefix}_attributes")
    if not _need(len(attributes) >= 2, "Select at least two common attributes."):
        return None
    return assessor, product, attributes


def _gpa(df: pd.DataFrame) -> None:
    st.header("Generalized Procrustes Analysis (GPA)")
    st.write("Centers, normalizes, rotates, and iteratively averages assessor configurations.")
    selected = _multiassessor_inputs(df, "gpa")
    if selected is None:
        return
    assessor, product, attributes = selected
    tables: list[np.ndarray] = []
    labels: pd.Index | None = None
    for _, group in df.groupby(assessor, observed=True):
        table = group.groupby(product, observed=True)[attributes].mean().dropna()
        labels = table.index if labels is None else labels.intersection(table.index)
    if not _need(
        df[assessor].nunique() >= 2 and labels is not None and len(labels) >= 3,
        "Need at least 2 assessors and 3 complete products shared by all assessors.",
    ):
        return
    for _, group in df.groupby(assessor, observed=True):
        table = group.groupby(product, observed=True)[attributes].mean().loc[labels]
        values = table.to_numpy(dtype=float)
        values -= values.mean(axis=0)
        norm = np.linalg.norm(values)
        tables.append(values / norm if norm else values)
    consensus = np.mean(tables, axis=0)
    for _ in range(30):
        aligned = [table @ linalg.orthogonal_procrustes(table, consensus)[0] for table in tables]
        updated = np.mean(aligned, axis=0)
        if np.linalg.norm(updated - consensus) < 1e-8:
            break
        consensus = updated
    coordinates = PCA(n_components=2).fit_transform(consensus)
    _scatter_labels(coordinates, labels, "GPA consensus")
    residuals = [float(np.linalg.norm(table - consensus)) for table in aligned]
    st.dataframe(pd.DataFrame({"Assessor table": range(1, len(residuals) + 1), "Residual": residuals}),
                 width="stretch")
    st.caption("Close products have similar consensus profiles; smaller residuals indicate assessor conformity.")


def _statis(df: pd.DataFrame) -> None:
    st.header("STATIS Compromise")
    st.write("RV similarity weights multiple assessor tables into a shared compromise.")
    selected = _multiassessor_inputs(df, "statis")
    if selected is None:
        return
    assessor, product, attributes = selected
    grouped = [(name, group.groupby(product, observed=True)[attributes].mean().dropna())
               for name, group in df.groupby(assessor, observed=True)]
    common = grouped[0][1].index
    for _, table in grouped[1:]:
        common = common.intersection(table.index)
    if not _need(len(grouped) >= 2 and len(common) >= 3, "Need two tables sharing at least three products."):
        return
    tables = [StandardScaler().fit_transform(table.loc[common]) for _, table in grouped]
    cross_products = [table @ table.T for table in tables]
    rv = np.array([
        [np.trace(a @ b) / np.sqrt(np.trace(a @ a) * np.trace(b @ b)) for b in cross_products]
        for a in cross_products
    ])
    eigenvalues, eigenvectors = np.linalg.eigh(rv)
    weights = np.abs(eigenvectors[:, -1])
    weights /= weights.sum()
    compromise = sum(weight * table for weight, table in zip(weights, tables))
    coordinates = PCA(n_components=2).fit_transform(compromise)
    _scatter_labels(coordinates, common, "STATIS compromise")
    st.dataframe(pd.DataFrame(rv, index=[str(x[0]) for x in grouped], columns=[str(x[0]) for x in grouped]),
                 width="stretch")
    st.dataframe(pd.DataFrame({"Table": [x[0] for x in grouped], "Weight": weights}), width="stretch")
    st.caption("High RV values mean similar tables; the compromise emphasizes the most mutually consistent tables.")


def _hac(df: pd.DataFrame) -> None:
    st.header("Hierarchical Agglomerative Clustering")
    numeric, _ = _columns(df)
    if not _need(len(numeric) >= 2, "Need at least two numeric variables."):
        return
    label = st.selectbox("Object label", ["(row number)", *df.columns.astype(str).tolist()], key="hac_label")
    features = st.multiselect("Variables", numeric, default=numeric[:6], key="hac_features")
    clean = df[features].dropna()
    if not _need(len(features) >= 2 and len(clean) >= 3, "Need 2 variables and 3 complete objects."):
        return
    labels = clean.index.astype(str) if label == "(row number)" else df.loc[clean.index, label].astype(str)
    method = st.selectbox("Linkage", ["ward", "average", "complete"], key="hac_method")
    tree = linkage(StandardScaler().fit_transform(clean), method=method)
    cluster_count = st.slider("Clusters", 2, min(8, len(clean)), min(3, len(clean)), key="hac_k")
    assignments = fcluster(tree, cluster_count, criterion="maxclust")
    st.dataframe(pd.DataFrame({"Object": labels.to_numpy(), "Cluster": assignments}), width="stretch")
    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(tree, labels=labels.to_numpy(), leaf_rotation=90, ax=ax)
    ax.set(title="Sensory-object dendrogram", ylabel="Linkage distance")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Objects joined at short distances have more similar standardized profiles.")


def _mfa(df: pd.DataFrame) -> None:
    st.header("Multiple Factor Analysis (MFA)")
    st.write("Each variable block is divided by its first singular value before a global PCA.")
    numeric, _ = _columns(df)
    if not _need(len(numeric) >= 3, "Need at least three numeric variables in two named blocks."):
        return
    delimiter = st.text_input("Block-name delimiter in columns", value="_", key="mfa_delimiter")
    features = st.multiselect("Variables", numeric, default=numeric[:10], key="mfa_features")
    blocks: dict[str, list[str]] = {}
    for feature in features:
        block = feature.split(delimiter, 1)[0] if delimiter and delimiter in feature else "Other"
        blocks.setdefault(block, []).append(feature)
    if not _need(len(blocks) >= 2, "Column names must identify at least two blocks (for example taste_sweet, aroma_fruity)."):
        return
    clean = df[features].dropna()
    pieces: list[np.ndarray] = []
    for columns in blocks.values():
        values = StandardScaler().fit_transform(clean[columns])
        first_singular = np.linalg.svd(values, compute_uv=False)[0]
        pieces.append(values / first_singular if first_singular else values)
    global_data = np.column_stack(pieces)
    pca = PCA(n_components=2).fit(global_data)
    scores = pca.transform(global_data)
    _scatter_labels(scores, clean.index, "MFA global scores",
                    f"Dimension 1 ({pca.explained_variance_ratio_[0]:.1%})",
                    f"Dimension 2 ({pca.explained_variance_ratio_[1]:.1%})")
    st.dataframe(pd.DataFrame({"Block": list(blocks), "Variables": [", ".join(v) for v in blocks.values()]}),
                 width="stretch")
    st.caption("Block scaling prevents large or high-variance blocks from dominating the global map.")


def _pls_coefficient_frame(
    model: PLSRegression,
    predictors: Sequence[str],
    responses: Sequence[str],
) -> pd.DataFrame:
    """Return PLS coefficients as predictors × responses across sklearn versions."""
    values = np.asarray(model.coef_)
    expected_shape = (len(predictors), len(responses))
    transposed_shape = (len(responses), len(predictors))
    if values.shape == expected_shape:
        oriented = values
    elif values.shape == transposed_shape:
        oriented = values.T
    else:
        raise ValueError(
            "Unexpected PLS coefficient shape "
            f"{values.shape}; expected {expected_shape} or {transposed_shape}."
        )
    return pd.DataFrame(oriented, index=predictors, columns=responses)


def _pls(df: pd.DataFrame) -> None:
    st.header("Partial Least Squares (PLS) Regression")
    st.write("Models one or more sensory/consumer outcomes from correlated predictors.")
    numeric, _ = _columns(df)
    if not _need(len(numeric) >= 2, "Need numeric predictor and response columns."):
        return
    responses = st.multiselect("Response (Y)", numeric, default=numeric[-1:], key="pls_y")
    predictors = st.multiselect("Predictors (X)", [c for c in numeric if c not in responses],
                                default=[c for c in numeric if c not in responses][:6], key="pls_x")
    clean = df[[*predictors, *responses]].dropna()
    maximum = min(len(predictors), max(1, len(clean) - 1))
    if not _need(bool(responses) and bool(predictors) and len(clean) >= 3, "Select X and Y with 3 complete rows."):
        return
    components = st.slider("Components", 1, maximum, min(2, maximum), key="pls_components")
    model = PLSRegression(n_components=components, scale=True).fit(clean[predictors], clean[responses])
    predicted = model.predict(clean[predictors])
    rows = []
    for index, response in enumerate(responses):
        observed = clean[response].to_numpy()
        residual_sum = float(np.sum((observed - predicted[:, index]) ** 2))
        total_sum = float(np.sum((observed - observed.mean()) ** 2))
        rows.append({"Response": response, "R² (training)": 1 - residual_sum / total_sum if total_sum else np.nan,
                     "RMSE": np.sqrt(residual_sum / len(observed))})
    st.dataframe(pd.DataFrame(rows), width="stretch")
    coefficients = _pls_coefficient_frame(model, predictors, responses)
    st.dataframe(coefficients.style.background_gradient(cmap="RdBu_r", axis=None), width="stretch")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(clean[responses[0]], predicted[:, 0], alpha=0.75)
    limits = [min(clean[responses[0]].min(), predicted[:, 0].min()),
              max(clean[responses[0]].max(), predicted[:, 0].max())]
    ax.plot(limits, limits, "--", color="grey")
    ax.set(xlabel="Observed", ylabel="Predicted", title=f"PLS fit: {responses[0]}")
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Larger absolute coefficients have more influence; training R² should be confirmed on new data.")


def _product_characterization(df: pd.DataFrame) -> None:
    _qda(df, "Product Characterization")


ROUTES: dict[str, Callable[[pd.DataFrame], None]] = {
    "sensory_triangle": lambda df: _discrimination(df, "Triangle Test", 1 / 3, "triangle"),
    "sensory_paired": lambda df: _discrimination(df, "Paired Comparison", 1 / 2, "paired"),
    "sensory_duo_trio": lambda df: _discrimination(df, "Duo-Trio Test", 1 / 2, "duotrio"),
    "sensory_tetrad": lambda df: _discrimination(df, "Tetrad Test", 1 / 3, "tetrad"),
    "sensory_ranking": _ranking,
    "sensory_qda": _qda,
    "sensory_temporal": _temporal,
    "sensory_panel_discrimination": _panel_discrimination,
    "sensory_panel_repeatability": _panel_repeatability,
    "sensory_panel_agreement": _panel_agreement,
    "sensory_panel_scale": _panel_scale,
    "sensory_product_characterization": _product_characterization,
    "sensory_cata": _cata,
    "sensory_jar": _jar,
    "sensory_hedonic": _hedonic,
    "sensory_segmentation": _segmentation,
    "sensory_internal_preference": _internal_preference,
    "sensory_external_preference": _external_preference,
    "sensory_pca": _pca,
    "sensory_gpa": _gpa,
    "sensory_statis": _statis,
    "sensory_hac": _hac,
    "sensory_mfa": _mfa,
    "sensory_pls": _pls,
}


def run_sensory_page(df: pd.DataFrame, page: str) -> None:
    """Dispatch an exact sensory navigation route to its interactive workflow."""
    workflow = ROUTES.get(page)
    if workflow is None:
        st.error(f"Unknown sensory route: {page}")
        return
    workflow(df)


def run_sensory_analysis(
    df: pd.DataFrame,
    default_analysis_type: str | None = None,
) -> None:
    """Backward-compatible entry point for the original sensory analysis menu."""
    options = {
        "Descriptive Analysis (Sensory Profile)": "sensory_qda",
        "Hedonic Analysis (Liking Scores)": "sensory_hedonic",
        "Attribute Ratings Summary": "sensory_pca",
        "Panelist Performance": "sensory_panel_repeatability",
        "Product Comparison": "sensory_product_characterization",
    }
    labels = list(options)
    index = labels.index(default_analysis_type) if default_analysis_type in options else 0
    selected = st.selectbox(
        "Select sensory analysis type:",
        labels,
        index=index,
        key=f"sensory_analysis_type_{default_analysis_type or 'default'}",
    )
    run_sensory_page(df, options[selected])


def descriptive_analysis(df: pd.DataFrame) -> None:
    """Compatibility wrapper for the original descriptive workflow."""
    _qda(df)


def hedonic_analysis(df: pd.DataFrame) -> None:
    """Compatibility wrapper for the original hedonic workflow."""
    _hedonic(df)


def attribute_summary(df: pd.DataFrame) -> None:
    """Compatibility wrapper for the original attribute-summary workflow."""
    _pca(df)


def panelist_performance(df: pd.DataFrame) -> None:
    """Compatibility wrapper for the original panelist workflow."""
    _panel_repeatability(df)


def product_comparison(df: pd.DataFrame) -> None:
    """Compatibility wrapper for the original product-comparison workflow."""
    _product_characterization(df)
