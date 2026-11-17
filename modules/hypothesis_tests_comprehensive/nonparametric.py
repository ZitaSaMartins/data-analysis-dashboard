"""Non-parametric tests: Mann-Whitney, Wilcoxon, Kruskal-Wallis, Friedman, Mood's median."""

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

from modules.stat_format import format_alpha, select_significance_level, with_formatted_p_values

from ._shared import (
    _paired_columns,
    _require_categorical,
    _require_numeric,
    _show_inference,
    _show_statistic,
    _show_variable_section_header,
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
    variables, grouping_variable, group_1, group_2 = inputs
    alpha = select_significance_level()
    if st.button("Run Mann-Whitney U", type="primary", key="run_mannwhitney"):
        if not variables:
            st.error("Select at least one numeric outcome variable.")
            return
        summary_rows: list[dict[str, object]] = []
        for variable in variables:
            values_1 = df.loc[
                df[grouping_variable] == group_1, variable
            ].dropna().astype(float)
            values_2 = df.loc[
                df[grouping_variable] == group_2, variable
            ].dropna().astype(float)
            if not len(values_1) or not len(values_2):
                st.warning(f"'{variable}': both groups need at least one observation; skipped.")
                continue
            statistic, p_value = stats.mannwhitneyu(
                values_1,
                values_2,
                alternative="two-sided",
            )
            is_significant = np.isfinite(p_value) and p_value <= alpha
            summary_rows.append({
                "Variable": variable,
                "U-statistic": statistic,
                "p-value": p_value,
                "Interpretation": (
                    "Group distributions differ significantly ✅"
                    if is_significant
                    else "No significant group-distribution difference ❌"
                ),
            })
        if not summary_rows:
            st.error("No valid Mann-Whitney results were produced.")
            return
        st.markdown("#### Summary")
        st.dataframe(
            with_formatted_p_values(pd.DataFrame(summary_rows), ["p-value"]),
            hide_index=True,
            width="stretch",
            column_config={"U-statistic": st.column_config.NumberColumn(format="%.4f")},
        )
        st.caption(f"Interpretation uses the selected significance level α = {format_alpha(alpha)}.")


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
        first, seconds = columns
        alpha = select_significance_level()
        if st.button("Run Wilcoxon Test", type="primary", key="run_wilcoxon_paired"):
            if not seconds:
                st.error("Select at least one second measurement.")
                return
            summary_rows: list[dict[str, object]] = []
            for second in seconds:
                paired = df[[first, second]].dropna().astype(float)
                differences = paired[first] - paired[second]
                if len(differences) < 2 or (differences == 0).all():
                    st.warning(
                        f"'{second}': at least two non-identical complete pairs "
                        "are required; skipped."
                    )
                    continue
                statistic, p_value = stats.wilcoxon(differences)
                is_significant = np.isfinite(p_value) and p_value <= alpha
                summary_rows.append({
                    "Variable": second,
                    "W-statistic": statistic,
                    "p-value": p_value,
                    "Interpretation": (
                        f"Significantly different from {first} ✅"
                        if is_significant
                        else f"No significant difference from {first} ❌"
                    ),
                })
            if not summary_rows:
                st.error("No valid Wilcoxon results were produced.")
                return
            st.markdown("#### Summary")
            st.dataframe(
                with_formatted_p_values(pd.DataFrame(summary_rows), ["p-value"]),
                hide_index=True,
                width="stretch",
                column_config={"W-statistic": st.column_config.NumberColumn(format="%.4f")},
            )
            st.caption(
                f"Each row compares {first} to the listed variable. Interpretation "
                f"uses the selected significance level α = {format_alpha(alpha)}."
            )
    else:
        numeric_columns = _require_numeric(df)
        if numeric_columns is None:
            return
        variables = st.multiselect(
            "Numeric variable(s):",
            numeric_columns,
            default=numeric_columns[:1],
            key="wilcoxon_one_sample_variable",
        )
        reference = st.number_input(
            "Reference median:",
            value=0.0,
            key="wilcoxon_reference",
        )
        alpha = select_significance_level()
        if st.button(
            "Run Wilcoxon Test",
            type="primary",
            key="run_wilcoxon_one_sample",
        ):
            if not variables:
                st.error("Select at least one numeric variable.")
                return
            summary_rows: list[dict[str, object]] = []
            for variable in variables:
                differences = df[variable].dropna().astype(float) - reference
                if len(differences) < 2 or (differences == 0).all():
                    st.warning(
                        f"'{variable}': at least two non-zero differences are "
                        "required; skipped."
                    )
                    continue
                statistic, p_value = stats.wilcoxon(differences)
                is_significant = np.isfinite(p_value) and p_value <= alpha
                summary_rows.append({
                    "Variable": variable,
                    "W-statistic": statistic,
                    "p-value": p_value,
                    "Interpretation": (
                        "Significantly different from the reference ✅"
                        if is_significant
                        else "No significant difference from the reference ❌"
                    ),
                })
            if not summary_rows:
                st.error("No valid Wilcoxon results were produced.")
                return
            st.markdown("#### Summary")
            st.dataframe(
                with_formatted_p_values(pd.DataFrame(summary_rows), ["p-value"]),
                hide_index=True,
                width="stretch",
                column_config={"W-statistic": st.column_config.NumberColumn(format="%.4f")},
            )
            st.caption(f"Interpretation uses the selected significance level α = {format_alpha(alpha)}.")


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
    alpha = select_significance_level()
    if st.button("Run Kruskal-Wallis", type="primary", key="run_kruskal"):
        if not selected_variables:
            st.error("Select at least one numeric outcome variable.")
            return
        summary_rows: list[dict[str, object]] = []
        pairwise_inputs: dict[str, dict[str, pd.Series]] = {}
        for variable in selected_variables:
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
            is_significant = np.isfinite(p_value) and p_value <= alpha
            summary_rows.append({
                "Variable": variable,
                "Groups": len(grouped_values),
                "H-statistic": statistic,
                "p-value": p_value,
                "Interpretation": (
                    "At least one group distribution differs significantly ✅"
                    if is_significant
                    else "No significant difference among group distributions ❌"
                ),
            })
            pairwise_inputs[variable] = grouped_values

        if not summary_rows:
            st.error("No valid Kruskal-Wallis results were produced.")
            return

        st.markdown("#### Summary")
        st.dataframe(
            with_formatted_p_values(pd.DataFrame(summary_rows), ["p-value"]),
            hide_index=True,
            width="stretch",
            column_config={
                "H-statistic": st.column_config.NumberColumn(format="%.4f"),
            },
        )
        st.caption(f"Interpretation uses the selected significance level α = {format_alpha(alpha)}.")

        st.markdown("#### Pairwise comparisons by variable")
        for index, (variable, grouped_values) in enumerate(pairwise_inputs.items()):
            _show_variable_section_header(variable, index)
            _display_pairwise_results(grouped_values, pairwise_method, "Median", alpha)


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
    alpha = select_significance_level()
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
            alpha,
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
    alpha = select_significance_level()
    if st.button("Run Mood's Median Test", type="primary", key="run_mood"):
        if not selected_variables:
            st.error("Select at least one numeric outcome variable.")
            return
        summary_rows: list[dict[str, object]] = []
        pairwise_inputs: dict[str, tuple[dict[str, pd.Series], np.ndarray]] = {}
        for variable in selected_variables:
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
            is_significant = np.isfinite(p_value) and p_value <= alpha
            summary_rows.append({
                "Variable": variable,
                "Groups": len(grouped_values),
                "χ² statistic": statistic,
                "Grand median": median,
                "p-value": p_value,
                "Interpretation": (
                    "Group medians differ significantly ✅"
                    if is_significant
                    else "No significant difference among group medians ❌"
                ),
            })
            pairwise_inputs[variable] = (grouped_values, contingency)

        if not summary_rows:
            st.error("No valid Mood's median results were produced.")
            return

        st.markdown("#### Summary")
        st.dataframe(
            with_formatted_p_values(pd.DataFrame(summary_rows), ["p-value"]),
            hide_index=True,
            width="stretch",
            column_config={
                "χ² statistic": st.column_config.NumberColumn(format="%.4f"),
                "Grand median": st.column_config.NumberColumn(format="%.4f"),
            },
        )
        st.caption(f"Interpretation uses the selected significance level α = {format_alpha(alpha)}.")

        st.markdown("#### Pairwise comparisons by variable")
        for index, (variable, (grouped_values, contingency)) in enumerate(pairwise_inputs.items()):
            _show_variable_section_header(variable, index)
            st.caption("Contingency table: counts above/below the grand median, by group.")
            st.dataframe(pd.DataFrame(contingency), use_container_width=True)
            _display_pairwise_results(grouped_values, pairwise_method, "Median", alpha)


