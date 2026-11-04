"""Check-All-That-Apply and Just-About-Right penalty analysis."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy import stats
from statsmodels.stats.contingency_tables import cochrans_q, mcnemar
from statsmodels.stats.multitest import multipletests

from ._cata_math import bootstrap_mean_drop, correspondence_analysis, pcoa
from ._shared import _columns, _compact_letters, _heatmap, _interpret_p, _need, _scatter_labels


def _cata(df: pd.DataFrame) -> None:
    st.header("Check-All-That-Apply (CATA)")
    st.write("Attribute frequencies describe products; Cochran's Q tests paired product differences.")
    numeric, _ = _columns(df)
    columns = df.columns.astype(str).tolist()
    boolean_columns = df.select_dtypes(include="bool").columns.astype(str).tolist()
    binary_eligible = [c for c in columns if c in numeric or c in boolean_columns]
    if not _need(bool(binary_eligible) and len(columns) >= 3, "Need consumer, product, and binary attribute columns."):
        return
    consumer = st.selectbox("Consumer", columns, key="cata_consumer")
    product = st.selectbox("Product", [c for c in columns if c != consumer], key="cata_product")
    candidates = [
        c for c in binary_eligible if c not in {consumer, product}
        and set(df[c].dropna().unique()).issubset({0, 1, False, True})
    ]
    attributes = st.multiselect("Binary CATA attributes", candidates, default=candidates[:8], key="cata_attr")
    if not _need(bool(attributes), "Select binary 0/1 attribute columns."):
        return
    liking_candidates = [c for c in numeric if c not in {consumer, product, *attributes}]
    liking_choice = st.selectbox(
        "Liking / preference column (optional, needed for the ideal-product penalty analysis)",
        ["(none)", *liking_candidates],
        key="cata_liking",
    )
    liking = None if liking_choice == "(none)" else liking_choice

    product_values = df[product].astype(str).unique().tolist()
    default_ideal_position = next(
        (index for index, value in enumerate(product_values) if value.strip().lower() == "ideal"),
        None,
    )
    ideal_options = ["(no ideal product)", *product_values]
    ideal_choice = st.selectbox(
        "Ideal product (optional)",
        ideal_options,
        index=0 if default_ideal_position is None else default_ideal_position + 1,
        key="cata_ideal",
    )
    ideal_product = None if ideal_choice == "(no ideal product)" else ideal_choice

    work_columns = [consumer, product, *attributes] + ([liking] if liking else [])
    work = df.loc[:, work_columns].copy()
    work[product] = work[product].astype(str)

    is_ideal = work[product] == ideal_product if ideal_product else pd.Series(False, index=work.index)
    real = work.loc[~is_ideal].copy()
    if not _need(real[product].nunique() >= 2, "Need at least two non-ideal products."):
        return

    _cata_data_validation(real, consumer, attributes)

    frequencies = real.groupby(product, observed=True)[attributes].mean() * 100
    st.subheader("Selection frequency (%)")
    st.dataframe(frequencies, width="stretch")

    q_rows: list[dict[str, object]] = []
    for attribute in attributes:
        paired = real.pivot_table(index=consumer, columns=product, values=attribute, aggfunc="max").dropna()
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
    letter_rows: list[dict[str, object]] = []
    for attribute in attributes:
        paired = real.pivot_table(
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
        if not raw_p_values:
            continue
        adjusted = multipletests(raw_p_values, alpha=0.05, method="holm")[1]
        significant_pairs = {
            tuple(sorted((first, second)))
            for (first, second, _), adjusted_p in zip(metadata, adjusted)
            if adjusted_p < 0.05
        }
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
        labels = [str(column) for column in paired.columns]
        letters = _compact_letters(labels, significant_pairs)
        rates = paired.mean() * 100
        letter_rows.extend(
            {
                "Attribute": attribute,
                "Product": label,
                "Checked (%)": float(rates.loc[column]),
                "Grouping letters": letters[label],
            }
            for label, column in zip(labels, paired.columns)
        )
    if pairwise_rows:
        st.subheader("Holm-adjusted pairwise McNemar comparisons")
        st.dataframe(pd.DataFrame(pairwise_rows), width="stretch")
    if letter_rows:
        st.subheader("Pairwise comparison letters")
        st.caption("Within each attribute, products sharing a letter are not significantly different.")
        st.dataframe(pd.DataFrame(letter_rows), width="stretch")

    _cata_independence_test(real, product, attributes)
    _cata_correspondence(real, product, attributes)
    _cata_correlation_pcoa(real, attributes, liking)
    if ideal_product:
        _cata_ideal_penalty_analysis(work, consumer, product, attributes, liking, ideal_product)
    else:
        st.info(
            "Select an ideal product above (and a liking column) to run the "
            "must-have / nice-to-have / must-not-have penalty analysis."
        )
    _heatmap(frequencies, "CATA attribute frequencies", ".1f")
    st.caption("Nearby products have similar attribute citation profiles; significant Q tests vary by product.")


def _cata_data_validation(data: pd.DataFrame, consumer: str, attributes: Sequence[str]) -> None:
    """Flag assessors and attributes whose check rate is far from the typical range."""
    st.subheader("CATA data validation")
    st.caption(
        "Checks for assessors who check attributes unusually rarely/often, and attributes "
        "that are strongly over- or under-used overall."
    )
    assessor_usage = (data.groupby(consumer, observed=True)[attributes].mean().mean(axis=1) * 100)
    attribute_usage = data[attributes].mean() * 100
    assessor_mean, assessor_std = float(assessor_usage.mean()), float(assessor_usage.std(ddof=1))
    attribute_mean, attribute_std = float(attribute_usage.mean()), float(attribute_usage.std(ddof=1))
    flagged_assessors = assessor_usage[(assessor_usage - assessor_mean).abs() > 2 * assessor_std]
    flagged_attributes = attribute_usage[(attribute_usage - attribute_mean).abs() > 2 * attribute_std]

    column1, column2 = st.columns(2)
    with column1:
        sorted_assessors = assessor_usage.sort_values()
        colors = ["#d62728" if label in flagged_assessors.index else "#4c78a8" for label in sorted_assessors.index]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(range(len(sorted_assessors)), sorted_assessors.to_numpy(), color=colors)
        ax.axhline(assessor_mean, color="grey", linestyle="--", linewidth=1)
        ax.set_xticks([])
        ax.set(ylabel="Attributes checked (%)", title="Assessor usage rate")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        if flagged_assessors.empty:
            st.success("No assessor's usage rate is unusually far from the average.")
        else:
            st.warning(f"{len(flagged_assessors)} assessor(s) have an unusual usage rate (>2 SD from the mean).")
            st.dataframe(flagged_assessors.rename("Checked (%)"), width="stretch")
    with column2:
        sorted_attributes = attribute_usage.sort_values()
        colors = ["#d62728" if label in flagged_attributes.index else "#4c78a8" for label in sorted_attributes.index]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(range(len(sorted_attributes)), sorted_attributes.to_numpy(), color=colors)
        ax.axhline(attribute_mean, color="grey", linestyle="--", linewidth=1)
        ax.set_xticks(range(len(sorted_attributes)))
        ax.set_xticklabels(sorted_attributes.index, rotation=45, ha="right")
        ax.set(ylabel="Citations (%)", title="Attribute usage rate")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        if flagged_attributes.empty:
            st.success("No attribute's usage rate is unusually far from the average.")
        else:
            st.warning(f"{len(flagged_attributes)} attribute(s) are strongly over- or under-used.")
            st.dataframe(flagged_attributes.rename("Citations (%)"), width="stretch")

    per_assessor_attribute = data.groupby(consumer, observed=True)[attributes].mean() * 100
    long = per_assessor_attribute.melt(var_name="Attribute", value_name="Checked (%)")
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.boxplot(data=long, x="Attribute", y="Checked (%)", ax=ax, color="#72b7b2")
    ax.tick_params(axis="x", rotation=45)
    ax.set_title("Consensus per attribute (spread of assessor usage rates)")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Attributes with wide boxes have the least assessor consensus.")


def _cata_independence_test(data: pd.DataFrame, product: str, attributes: Sequence[str]) -> None:
    """For each product, flag attribute pairs that are redundant (significantly associated)."""
    st.subheader("Attribute independence test")
    st.caption(
        "Fisher's exact test on each attribute pair, within each product, flags pairs that "
        "tend to be checked together or never together (redundant attributes)."
    )
    rows: list[dict[str, object]] = []
    for product_name, group in data.groupby(product, observed=True):
        for first_index, first in enumerate(attributes):
            for second in attributes[first_index + 1 :]:
                table = pd.crosstab(group[first], group[second]).reindex(
                    index=[0, 1], columns=[0, 1], fill_value=0
                )
                if table.to_numpy().sum() == 0:
                    continue
                _, p_value = stats.fisher_exact(table.to_numpy())
                rows.append(
                    {
                        "Product": product_name,
                        "Attribute 1": first,
                        "Attribute 2": second,
                        "p-value": float(p_value),
                        "Redundant (5%)": bool(p_value < 0.05),
                    }
                )
    result = pd.DataFrame(rows)
    if not _need(not result.empty, "Not enough data to test attribute independence."):
        return
    st.dataframe(result.sort_values(["Product", "p-value"]), width="stretch")
    redundant = result.loc[result["Redundant (5%)"]]
    if redundant.empty:
        st.success("No attribute pair shows significant redundancy within any product.")
    else:
        st.warning(
            f"{len(redundant)} attribute pair-product combination(s) show significant redundancy."
        )


def _cata_correspondence(data: pd.DataFrame, product: str, attributes: Sequence[str]) -> None:
    """Chi-square-distance Correspondence Analysis of the product x attribute citation table."""
    st.subheader("Correspondence Analysis (attribute profiles)")
    counts = data.groupby(product, observed=True)[attributes].sum()
    if not _need(counts.shape[0] >= 2 and counts.shape[1] >= 2, "Need at least two products and two attributes."):
        return
    chi2_statistic, chi2_p, _, _ = stats.chi2_contingency(counts)
    row_coords, col_coords, inertia_fraction = correspondence_analysis(counts)
    st.write(f"χ² = {chi2_statistic:.3f}, p = {chi2_p:.4g} for the product × attribute association.")
    _interpret_p(float(chi2_p), subject="product sensory profiles")
    first_two = float(inertia_fraction[:2].sum()) * 100 if len(inertia_fraction) >= 2 else float(inertia_fraction.sum()) * 100
    st.caption(f"The first two dimensions explain {first_two:.2f}% of total inertia.")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(row_coords["Dim1"], row_coords["Dim2"], color="#4c78a8", s=90, label="Products")
    for label, x_value, y_value in zip(row_coords.index, row_coords["Dim1"], row_coords["Dim2"]):
        ax.annotate(str(label), (x_value, y_value), xytext=(5, 5), textcoords="offset points", fontweight="bold")
    ax.scatter(col_coords["Dim1"], col_coords["Dim2"], color="#e45756", marker="^", s=70, label="Attributes")
    for label, x_value, y_value in zip(col_coords.index, col_coords["Dim1"], col_coords["Dim2"]):
        ax.annotate(str(label), (x_value, y_value), xytext=(5, 5), textcoords="offset points", color="#e45756")
    ax.axhline(0, color="grey", linewidth=0.7)
    ax.axvline(0, color="grey", linewidth=0.7)
    dim1_pct = inertia_fraction[0] * 100 if len(inertia_fraction) >= 1 else 0.0
    dim2_pct = inertia_fraction[1] * 100 if len(inertia_fraction) >= 2 else 0.0
    ax.set(
        xlabel=f"Dimension 1 ({dim1_pct:.1f}%)",
        ylabel=f"Dimension 2 ({dim2_pct:.1f}%)",
        title="CATA Correspondence Analysis biplot",
    )
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Products and attributes plotted close together tend to co-occur in consumers' responses.")


def _cata_correlation_pcoa(data: pd.DataFrame, attributes: Sequence[str], liking: str | None) -> None:
    """Attribute (and liking) correlation matrix, plus a PCoA map derived from it."""
    st.subheader("Attribute correlation matrix")
    correlation_columns = [*attributes, *([liking] if liking else [])]
    correlation = data[correlation_columns].corr()
    _heatmap(correlation, "Attribute (and liking) correlations", ".2f")

    st.subheader("Principal Coordinates Analysis (PCoA)")
    st.caption("Maps attributes so that nearby points are more positively correlated with each other.")
    constant_columns = [column for column in correlation.columns if correlation[column].isna().any()]
    if constant_columns:
        st.warning(
            "Excluded constant column(s) with an undefined correlation: "
            f"{', '.join(str(column) for column in constant_columns)}."
        )
        correlation = correlation.drop(index=constant_columns, columns=constant_columns)
    if not _need(correlation.shape[0] >= 3, "Need at least three non-constant attributes for a PCoA map."):
        return
    distance = np.sqrt(np.clip(2 * (1 - correlation.to_numpy()), 0, None))
    coordinates, fraction = pcoa(distance)
    first_two = float(fraction[:2].sum()) * 100 if len(fraction) >= 2 else float(fraction.sum()) * 100
    st.caption(f"The first two axes explain {first_two:.2f}% of the total variance.")
    axis1_pct = fraction[0] * 100 if len(fraction) >= 1 else 0.0
    axis2_pct = fraction[1] * 100 if len(fraction) >= 2 else 0.0
    _scatter_labels(
        coordinates[:, :2],
        correlation.index,
        "Attribute PCoA map",
        f"Axis 1 ({axis1_pct:.1f}%)",
        f"Axis 2 ({axis2_pct:.1f}%)",
    )


def _cata_ideal_penalty_analysis(
    work: pd.DataFrame,
    consumer: str,
    product: str,
    attributes: Sequence[str],
    liking: str | None,
    ideal_product: str,
) -> None:
    """Must-have / nice-to-have / must-not-have penalty analysis against an ideal product."""
    st.subheader("Ideal-product penalty analysis")
    st.caption(
        "Compares liking between consumers who did/didn't find an attribute in the product, "
        "separately for consumers who do/don't want that attribute in their ideal. Significance "
        "uses a bootstrap mean-drop confidence interval, Holm-adjusted across attributes."
    )
    if not _need(liking is not None, "Select a liking column above to run the ideal-product penalty analysis."):
        return
    ideal_rows = work.loc[work[product] == ideal_product].set_index(consumer)
    real_rows = work.loc[work[product] != ideal_product]
    random = np.random.default_rng(42)

    must_have_rows: list[dict[str, object]] = []
    not_nice_rows: list[dict[str, object]] = []
    difference_rows: list[dict[str, object]] = []
    for attribute in attributes:
        merged = real_rows.join(ideal_rows[attribute].rename("_ideal_checked"), on=consumer)
        merged = merged.dropna(subset=[liking, "_ideal_checked"])

        wanted = merged.loc[merged["_ideal_checked"] == 1]
        present = wanted.loc[wanted[attribute] == 1, liking].to_numpy(dtype=float)
        missing = wanted.loc[wanted[attribute] == 0, liking].to_numpy(dtype=float)
        if len(present) >= 2 and len(missing) >= 2:
            drop, ci_low, ci_high, p_value = bootstrap_mean_drop(present, missing, random)
            must_have_rows.append(
                {
                    "Attribute": attribute,
                    "Missing when wanted (%)": 100 * len(missing) / len(wanted),
                    "Mean liking drop": drop,
                    "95% CI low": ci_low,
                    "95% CI high": ci_high,
                    "p-value": p_value,
                }
            )

        unwanted = merged.loc[merged["_ideal_checked"] == 0]
        present_unwanted = unwanted.loc[unwanted[attribute] == 1, liking].to_numpy(dtype=float)
        absent_unwanted = unwanted.loc[unwanted[attribute] == 0, liking].to_numpy(dtype=float)
        if len(present_unwanted) >= 2 and len(absent_unwanted) >= 2:
            drop, ci_low, ci_high, p_value = bootstrap_mean_drop(absent_unwanted, present_unwanted, random)
            not_nice_rows.append(
                {
                    "Attribute": attribute,
                    "Present when unwanted (%)": 100 * len(present_unwanted) / len(unwanted),
                    "Mean liking drop": drop,
                    "95% CI low": ci_low,
                    "95% CI high": ci_high,
                    "p-value": p_value,
                }
            )

        product_rate = float(real_rows[attribute].mean()) * 100
        ideal_rate = float(ideal_rows[attribute].mean()) * 100
        difference_rows.append(
            {
                "Attribute": attribute,
                "Product checked (%)": product_rate,
                "Ideal checked (%)": ideal_rate,
                "Difference": product_rate - ideal_rate,
            }
        )

    must_have_table = pd.DataFrame(must_have_rows)
    if not must_have_table.empty:
        must_have_table["Adjusted p-value"] = multipletests(
            must_have_table["p-value"], alpha=0.05, method="holm"
        )[1]
        must_have_table["Must-have"] = (
            (must_have_table["Adjusted p-value"] < 0.05) & (must_have_table["Mean liking drop"] > 0)
        )
    not_nice_table = pd.DataFrame(not_nice_rows)
    if not not_nice_table.empty:
        not_nice_table["Adjusted p-value"] = multipletests(
            not_nice_table["p-value"], alpha=0.05, method="holm"
        )[1]
        significant = not_nice_table["Adjusted p-value"] < 0.05
        not_nice_table["Must-not-have"] = significant & (not_nice_table["Mean liking drop"] > 0)
        not_nice_table["Nice-to-have"] = significant & (not_nice_table["Mean liking drop"] < 0)
    if must_have_table.empty and not_nice_table.empty:
        st.info("Not enough paired ideal/product responses to run the penalty analysis.")
        return

    if not must_have_table.empty:
        st.markdown("**Must-have check:** liking among consumers who wanted this attribute")
        st.dataframe(must_have_table, width="stretch")
    if not not_nice_table.empty:
        st.markdown("**Must-not-have / nice-to-have check:** liking among consumers who did not want this attribute")
        st.dataframe(not_nice_table, width="stretch")

    summary_rows: list[dict[str, object]] = []
    for attribute in attributes:
        must_have = bool(
            not must_have_table.empty
            and attribute in must_have_table["Attribute"].to_numpy()
            and must_have_table.loc[must_have_table["Attribute"] == attribute, "Must-have"].iloc[0]
        )
        must_not_have = bool(
            not not_nice_table.empty
            and attribute in not_nice_table["Attribute"].to_numpy()
            and not_nice_table.loc[not_nice_table["Attribute"] == attribute, "Must-not-have"].iloc[0]
        )
        nice_to_have = bool(
            not not_nice_table.empty
            and attribute in not_nice_table["Attribute"].to_numpy()
            and not_nice_table.loc[not_nice_table["Attribute"] == attribute, "Nice-to-have"].iloc[0]
        )
        classification = (
            "Must-have & must-not-have (polarizing)" if must_have and must_not_have
            else "Must-have" if must_have
            else "Must-not-have" if must_not_have
            else "Nice-to-have" if nice_to_have
            else "No clear influence"
        )
        summary_rows.append({"Attribute": attribute, "Classification": classification})
    summary_table = pd.DataFrame(summary_rows)
    st.markdown("**Summary classification**")
    st.dataframe(summary_table, width="stretch")
    counts_by_class = summary_table["Classification"].value_counts()
    st.caption(
        ", ".join(f"{count} {label.lower()}" for label, count in counts_by_class.items()) + "."
    )

    if not must_have_table.empty or not not_nice_table.empty:
        impact_rows = []
        if not must_have_table.empty:
            impact_rows.append(must_have_table[["Attribute", "Mean liking drop"]])
        if not not_nice_table.empty:
            impact_rows.append(
                not_nice_table[["Attribute", "Mean liking drop"]].rename(
                    columns={"Mean liking drop": "Mean liking drop"}
                )
            )
        impact = pd.concat(impact_rows, ignore_index=True).groupby("Attribute")["Mean liking drop"].mean()
        impact = impact.sort_values()
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["#4c78a8" if value > 0 else "#e45756" for value in impact.to_numpy()]
        ax.barh(impact.index, impact.to_numpy(), color=colors)
        ax.axvline(0, color="grey", linewidth=0.8)
        ax.set(xlabel="Mean liking impact", title="Mean impact chart")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Blue bars (positive) support must-have attributes; red bars (negative) flag must-not-have ones.")

    st.markdown("**Difference vs. ideal**")
    difference_table = pd.DataFrame(difference_rows).set_index("Attribute")
    _heatmap(difference_table[["Difference"]].T, "Checked (%) difference: product − ideal", ".1f")
    st.caption(
        "Positive values mean the attribute is checked more often in the tested product than in "
        "consumers' ideal; negative values mean it is under-represented relative to the ideal."
    )


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
    ax.axvline(minimum_consumers, color="grey", linestyle="--")
    ax.set(title="JAR penalty plot")
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Large positive drops affecting many consumers are the strongest reformulation priorities.")

