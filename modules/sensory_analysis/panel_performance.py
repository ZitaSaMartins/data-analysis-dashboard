"""Panelist performance: discrimination, repeatability, agreement, scale usage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

from ._shared import _columns, _heatmap, _need


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

