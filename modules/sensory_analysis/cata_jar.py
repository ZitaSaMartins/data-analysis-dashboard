"""Check-All-That-Apply and Just-About-Right penalty analysis."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.contingency_tables import cochrans_q, mcnemar
from statsmodels.stats.multitest import multipletests

from ._shared import _columns, _heatmap, _need, _scatter_labels


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

