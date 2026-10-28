"""Non-parametric tests: Mann-Whitney, Wilcoxon, Kruskal-Wallis, Friedman, Mood's median."""

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

from ._shared import (
    _paired_columns,
    _require_categorical,
    _require_numeric,
    _show_inference,
    _show_statistic,
    _two_group_inputs,
)
from .posthoc import _display_pairwise_results, _show_pairwise_options


def test_mannwhitney(df: pd.DataFrame) -> None:
    """Run the Mann-Whitney U test for two independent groups."""
    st.markdown("## 🟧 Mann-Whitney U Test")
    st.write("Rank-based comparison of two independent groups.")
    inputs = _two_group_inputs(df, "mannwhitney")
    if inputs is None:
        return
    variable, grouping_variable, group_1, group_2 = inputs
    if st.button("Run Mann-Whitney U", type="primary", key="run_mannwhitney"):
        values_1 = df.loc[
            df[grouping_variable] == group_1, variable
        ].dropna().astype(float)
        values_2 = df.loc[
            df[grouping_variable] == group_2, variable
        ].dropna().astype(float)
        if not len(values_1) or not len(values_2):
            st.error("Both groups need at least one observation.")
            return
        statistic, p_value = stats.mannwhitneyu(
            values_1,
            values_2,
            alternative="two-sided",
        )
        _show_statistic("U-statistic", statistic, p_value)
        _show_inference(
            p_value,
            "The group distributions are significantly different",
            "No significant group-distribution difference was detected",
        )


def test_wilcoxon(df: pd.DataFrame) -> None:
    """Run a paired or one-sample Wilcoxon signed-rank test."""
    st.markdown("## 🟧 Wilcoxon Signed-Rank Test")
    st.write("Rank-based test for paired measurements or one sample versus a reference.")
    mode = st.radio(
        "Test design:",
        ["Paired measurements", "One sample vs reference"],
        horizontal=True,
        key="wilcoxon_mode",
    )
    if mode == "Paired measurements":
        columns = _paired_columns(df, "wilcoxon")
        if columns is None:
            return
        first, second = columns
        if st.button("Run Wilcoxon Test", type="primary", key="run_wilcoxon_paired"):
            paired = df[[first, second]].dropna().astype(float)
            differences = paired[first] - paired[second]
            if len(differences) < 2 or (differences == 0).all():
                st.error("At least two non-identical complete pairs are required.")
                return
            statistic, p_value = stats.wilcoxon(differences)
            _show_statistic("W-statistic", statistic, p_value)
            _show_inference(
                p_value,
                "The paired measurements are significantly different",
                "No significant paired difference was detected",
            )
    else:
        numeric_columns = _require_numeric(df)
        if numeric_columns is None:
            return
        variable = st.selectbox(
            "Numeric variable:",
            numeric_columns,
            key="wilcoxon_one_sample_variable",
        )
        reference = st.number_input(
            "Reference median:",
            value=0.0,
            key="wilcoxon_reference",
        )
        if st.button(
            "Run Wilcoxon Test",
            type="primary",
            key="run_wilcoxon_one_sample",
        ):
            differences = df[variable].dropna().astype(float) - reference
            if len(differences) < 2 or (differences == 0).all():
                st.error("At least two non-zero differences are required.")
                return
            statistic, p_value = stats.wilcoxon(differences)
            _show_statistic("W-statistic", statistic, p_value)
            _show_inference(
                p_value,
                "The sample differs significantly from the reference",
                "No significant difference from the reference was detected",
            )


def test_kruskal_wallis(df: pd.DataFrame) -> None:
    """Run the Kruskal-Wallis test for independent groups."""
    st.markdown("## 🟧 Kruskal-Wallis Test")
    st.write("Rank-based comparison of three or more independent groups.")
    numeric_columns = _require_numeric(df)
    categorical_columns = _require_categorical(df)
    if numeric_columns is None or categorical_columns is None:
        return
    selected_variables = st.multiselect(
        "Numeric outcome variables:",
        numeric_columns,
        default=numeric_columns[:1],
        key="kruskal_numeric_variables",
    )
    grouping_variable = st.selectbox(
        "Grouping variable:",
        categorical_columns,
        key="kruskal_group",
    )
    pairwise_method = _show_pairwise_options("nonparametric", "kruskal")
    if st.button("Run Kruskal-Wallis", type="primary", key="run_kruskal"):
        if not selected_variables:
            st.error("Select at least one numeric outcome variable.")
            return
        for variable in selected_variables:
            st.markdown(f"### Outcome: {variable}")
            grouped_values = {
                str(group_name): group[variable].dropna().astype(float)
                for group_name, group in df.groupby(grouping_variable, observed=True)
            }
            grouped_values = {
                group: values
                for group, values in grouped_values.items()
                if len(values) >= 1
            }
            if len(grouped_values) < 3:
                st.warning(f"'{variable}' needs at least three non-empty groups.")
                continue
            groups = list(grouped_values.values())
            try:
                statistic, p_value = stats.kruskal(*groups)
            except ValueError as error:
                st.warning(f"'{variable}' could not be tested: {error}")
                continue
            _show_statistic("H-statistic", statistic, p_value)
            _show_inference(
                p_value,
                "At least one group distribution is significantly different",
                "No significant difference among group distributions was detected",
            )
            _display_pairwise_results(grouped_values, pairwise_method, "Median")


def test_friedman(df: pd.DataFrame) -> None:
    """Run Friedman's rank test for three or more related measurements."""
    st.markdown("## 🟧 Friedman Test")
    st.write("Rank-based comparison of three or more related numeric measurements.")
    numeric_columns = _require_numeric(df, minimum=3)
    if numeric_columns is None:
        return
    selected_columns = st.multiselect(
        "Related measurements:",
        numeric_columns,
        default=numeric_columns[:3],
        key="friedman_variables",
    )
    if len(selected_columns) < 3:
        st.info("Select at least three related measurements.")
        return
    if st.button("Run Friedman Test", type="primary", key="run_friedman"):
        complete = df[selected_columns].dropna().astype(float)
        if len(complete) < 2:
            st.error("At least two complete cases are required.")
            return
        statistic, p_value = stats.friedmanchisquare(
            *(complete[column] for column in selected_columns)
        )
        _show_statistic("χ² statistic", statistic, p_value)
        _show_inference(
            p_value,
            "At least one related measurement is significantly different",
            "No significant difference among related measurements was detected",
        )


def test_mood_median(df: pd.DataFrame) -> None:
    """Run Mood's median test for independent groups."""
    st.markdown("## 🟧 Mood's Median Test")
    st.write("Tests whether independent groups have the same median.")
    numeric_columns = _require_numeric(df)
    categorical_columns = _require_categorical(df)
    if numeric_columns is None or categorical_columns is None:
        return
    selected_variables = st.multiselect(
        "Numeric outcome variables:",
        numeric_columns,
        default=numeric_columns[:1],
        key="mood_numeric_variables",
    )
    grouping_variable = st.selectbox(
        "Grouping variable:",
        categorical_columns,
        key="mood_group",
    )
    pairwise_method = _show_pairwise_options("nonparametric", "mood")
    if st.button("Run Mood's Median Test", type="primary", key="run_mood"):
        if not selected_variables:
            st.error("Select at least one numeric outcome variable.")
            return
        for variable in selected_variables:
            st.markdown(f"### Outcome: {variable}")
            grouped_values = {
                str(group_name): group[variable].dropna().astype(float)
                for group_name, group in df.groupby(grouping_variable, observed=True)
            }
            grouped_values = {
                group: values
                for group, values in grouped_values.items()
                if len(values) >= 1
            }
            if len(grouped_values) < 3:
                st.warning(f"'{variable}' needs at least three non-empty groups.")
                continue
            groups = list(grouped_values.values())
            try:
                statistic, p_value, median, contingency = stats.median_test(*groups)
            except ValueError as error:
                st.warning(f"'{variable}' could not be tested: {error}")
                continue
            statistic_col, p_value_col, median_col = st.columns(3)
            statistic_col.metric("χ² statistic", f"{statistic:.4f}")
            p_value_col.metric("p-value", f"{p_value:.4f}")
            median_col.metric("Grand median", f"{median:.4f}")
            st.dataframe(pd.DataFrame(contingency), use_container_width=True)
            _show_inference(
                p_value,
                "The group medians are significantly different",
                "No significant difference among group medians was detected",
            )
            _display_pairwise_results(grouped_values, pairwise_method, "Median")

