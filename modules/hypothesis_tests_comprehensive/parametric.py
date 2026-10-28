"""Parametric tests: t-tests and ANOVA family."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import AnovaRM

from ._shared import (
    ALPHA,
    _paired_columns,
    _require_categorical,
    _require_numeric,
    _show_inference,
    _show_statistic,
    _two_group_inputs,
)
from .posthoc import _display_pairwise_results, _show_pairwise_options


def test_one_sample_ttest(df: pd.DataFrame) -> None:
    """Run a one-sample t-test against a user-provided reference mean."""
    st.markdown("## 🟦 One-Sample t-Test")
    st.write("Tests whether one numeric sample mean differs from a reference value.")
    numeric_columns = _require_numeric(df)
    if numeric_columns is None:
        return
    variable = st.selectbox(
        "Numeric variable:",
        numeric_columns,
        key="one_sample_t_variable",
    )
    reference = st.number_input(
        "Reference mean:",
        value=0.0,
        key="one_sample_t_reference",
    )
    if st.button("Run One-Sample t-Test", type="primary", key="run_one_sample_t"):
        values = df[variable].dropna().astype(float)
        if len(values) < 2:
            st.error("At least two observations are required.")
            return
        statistic, p_value = stats.ttest_1samp(values, popmean=reference)
        _show_statistic("t-statistic", statistic, p_value)
        st.metric("Sample mean", f"{values.mean():.4f}")
        _show_inference(
            p_value,
            "The sample mean is significantly different from the reference mean",
            "No significant difference from the reference mean was detected",
        )


def test_independent_ttest(df: pd.DataFrame) -> None:
    """Run an independent samples t-test with Cohen's d."""
    st.markdown("## 🟦 Independent Samples t-Test")
    st.write("Compares the means of two independent groups.")
    inputs = _two_group_inputs(df, "independent_t")
    if inputs is None:
        return
    variable, grouping_variable, group_1, group_2 = inputs
    equal_variance = st.checkbox(
        "Assume equal variances",
        value=True,
        key="independent_t_equal_variance",
    )

    if st.button(
        "Run Independent t-Test",
        type="primary",
        key="run_independent_t",
    ):
        values_1 = df.loc[
            df[grouping_variable] == group_1, variable
        ].dropna().astype(float)
        values_2 = df.loc[
            df[grouping_variable] == group_2, variable
        ].dropna().astype(float)
        if len(values_1) < 2 or len(values_2) < 2:
            st.error("Each group needs at least two observations.")
            return

        statistic, p_value = stats.ttest_ind(
            values_1,
            values_2,
            equal_var=equal_variance,
        )
        pooled_variance = (
            ((len(values_1) - 1) * values_1.var(ddof=1))
            + ((len(values_2) - 1) * values_2.var(ddof=1))
        ) / (len(values_1) + len(values_2) - 2)
        cohens_d = (
            (values_1.mean() - values_2.mean()) / np.sqrt(pooled_variance)
            if pooled_variance > 0
            else np.nan
        )

        mean_1, mean_2, effect = st.columns(3)
        mean_1.metric(f"{group_1} mean", f"{values_1.mean():.4f}")
        mean_2.metric(f"{group_2} mean", f"{values_2.mean():.4f}")
        effect.metric("Cohen's d", f"{cohens_d:.4f}")
        _show_statistic("t-statistic", statistic, p_value)
        _show_inference(
            p_value,
            "The group means are significantly different",
            "No significant difference between group means was detected",
        )


def test_paired_ttest(df: pd.DataFrame) -> None:
    """Run a paired samples t-test."""
    st.markdown("## 🟦 Paired Samples t-Test")
    st.write("Compares two related numeric measurements from the same cases.")
    columns = _paired_columns(df, "paired_t")
    if columns is None:
        return
    first, second = columns
    if st.button("Run Paired t-Test", type="primary", key="run_paired_t"):
        paired = df[[first, second]].dropna().astype(float)
        if len(paired) < 2:
            st.error("At least two complete pairs are required.")
            return
        statistic, p_value = stats.ttest_rel(paired[first], paired[second])
        _show_statistic("t-statistic", statistic, p_value)
        st.metric("Mean paired difference", f"{(paired[first] - paired[second]).mean():.4f}")
        _show_inference(
            p_value,
            "The paired measurements are significantly different",
            "No significant paired difference was detected",
        )


def test_anova(df: pd.DataFrame) -> None:
    """Run a one-way independent-groups ANOVA."""
    st.markdown("## 🟦 One-Way ANOVA")
    st.write("Compares a numeric mean across three or more independent groups.")
    numeric_columns = _require_numeric(df)
    categorical_columns = _require_categorical(df)
    if numeric_columns is None or categorical_columns is None:
        return
    selected_variables = st.multiselect(
        "Numeric outcome variables:",
        numeric_columns,
        default=numeric_columns[:1],
        key="anova_numeric_variables",
    )
    grouping_variable = st.selectbox(
        "Grouping variable:",
        categorical_columns,
        key="anova_group",
    )
    pairwise_method = _show_pairwise_options("anova", "anova")
    if st.button("Run One-Way ANOVA", type="primary", key="run_anova"):
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
                if len(values) >= 2
            }
            if len(grouped_values) < 3:
                st.warning(
                    f"'{variable}' needs at least three groups with two observations each."
                )
                continue
            groups = list(grouped_values.values())
            statistic, p_value = stats.f_oneway(*groups)
            all_values = np.concatenate(groups)
            grand_mean = all_values.mean()
            between_ss = sum(
                len(group) * (group.mean() - grand_mean) ** 2 for group in groups
            )
            total_ss = ((all_values - grand_mean) ** 2).sum()
            eta_squared = between_ss / total_ss if total_ss > 0 else np.nan
            _show_statistic("F-statistic", statistic, p_value)
            st.metric("Eta squared", f"{eta_squared:.4f}")
            _show_inference(
                p_value,
                "At least one group mean is significantly different",
                "No significant difference among group means was detected",
            )
            _display_pairwise_results(grouped_values, pairwise_method, "Mean")


def _fit_two_way_anova(
    data: pd.DataFrame,
    sum_squares_type: int,
):
    if sum_squares_type == 3:
        formula = (
            "_outcome ~ C(_factor_1, Sum) * C(_factor_2, Sum)"
        )
        factor_1_term = "C(_factor_1, Sum)"
        factor_2_term = "C(_factor_2, Sum)"
    else:
        formula = "_outcome ~ C(_factor_1) * C(_factor_2)"
        factor_1_term = "C(_factor_1)"
        factor_2_term = "C(_factor_2)"

    model = ols(formula, data=data).fit()
    table = sm.stats.anova_lm(model, typ=sum_squares_type)
    return model, table, factor_1_term, factor_2_term


def _two_way_diagnostic_figure(
    model,
    factor_1: str,
    factor_2: str,
) -> go.Figure:
    residuals = np.asarray(model.resid, dtype=float)
    fitted = np.asarray(model.fittedvalues, dtype=float)
    (theoretical, ordered), (slope, intercept, _) = stats.probplot(
        residuals,
        dist="norm",
        fit=True,
    )
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Residuals vs Fitted", "Residual Normal Q-Q Plot"),
    )
    figure.add_trace(
        go.Scatter(
            x=fitted,
            y=residuals,
            mode="markers",
            marker={"color": "#4C78A8", "size": 8},
            name="Residuals",
        ),
        row=1,
        col=1,
    )
    figure.add_hline(y=0, line_dash="dash", line_color="#E45756", row=1, col=1)
    figure.add_trace(
        go.Scatter(
            x=theoretical,
            y=ordered,
            mode="markers",
            marker={"color": "#4C78A8", "size": 8},
            name="Observed quantiles",
        ),
        row=1,
        col=2,
    )
    figure.add_trace(
        go.Scatter(
            x=theoretical,
            y=slope * theoretical + intercept,
            mode="lines",
            line={"color": "#E45756", "width": 2},
            name="Normal reference",
        ),
        row=1,
        col=2,
    )
    figure.update_xaxes(title_text="Fitted values", row=1, col=1)
    figure.update_yaxes(title_text="Residuals", row=1, col=1)
    figure.update_xaxes(title_text="Theoretical quantiles", row=1, col=2)
    figure.update_yaxes(title_text="Observed residuals", row=1, col=2)
    figure.update_layout(
        height=420,
        margin={"l": 30, "r": 20, "t": 60, "b": 30},
        showlegend=False,
        title={"text": f"Model diagnostics: {factor_1} × {factor_2}", "x": 0.02},
    )
    return figure


def test_two_way_anova(df: pd.DataFrame) -> None:
    """Run a factorial two-way independent-groups ANOVA."""
    st.markdown("## 🟦 Two-Way ANOVA")
    st.write(
        "Tests two categorical factors simultaneously: each factor's main effect "
        "and whether the factors interact in their effect on a numeric outcome."
    )
    numeric_columns = _require_numeric(df)
    categorical_columns = _require_categorical(df, minimum=2)
    if numeric_columns is None or categorical_columns is None:
        return
    factor_columns = [
        column
        for column in categorical_columns
        if 2 <= df[column].nunique(dropna=True) < len(df)
    ]
    factor_columns.sort(key=lambda column: df[column].nunique(dropna=True))
    if len(factor_columns) < 2:
        st.error(
            "Two-Way ANOVA requires at least two categorical factors with repeated "
            "levels. Unique identifier columns cannot be used as factors."
        )
        return

    if st.session_state.get("two_way_anova_factor_1") not in factor_columns:
        st.session_state.two_way_anova_factor_1 = factor_columns[0]
    if st.session_state.get("two_way_anova_outcome") not in numeric_columns:
        st.session_state.two_way_anova_outcome = numeric_columns[0]
    outcome_column, factor_1_column, factor_2_column = st.columns(3)
    with outcome_column:
        outcome = st.selectbox(
            "Numeric outcome:",
            numeric_columns,
            key="two_way_anova_outcome",
        )
    with factor_1_column:
        factor_1 = st.selectbox(
            "Factor 1:",
            factor_columns,
            key="two_way_anova_factor_1",
        )
    with factor_2_column:
        factor_2_options = [
            column for column in factor_columns if column != factor_1
        ]
        if st.session_state.get("two_way_anova_factor_2") not in factor_2_options:
            st.session_state.two_way_anova_factor_2 = factor_2_options[0]
        factor_2 = st.selectbox(
            "Factor 2:",
            factor_2_options,
            key="two_way_anova_factor_2",
        )

    st.caption(
        "Use two independent categorical factors with repeated observations in "
        "every factor combination. For repeated or matched participants, use a "
        "repeated-measures or mixed-design method instead."
    )

    if st.button(
        "Run Two-Way ANOVA",
        type="primary",
        key="run_two_way_anova",
    ):
        analysis_data = (
            df[[outcome, factor_1, factor_2]]
            .dropna()
            .rename(columns={
                outcome: "_outcome",
                factor_1: "_factor_1",
                factor_2: "_factor_2",
            })
        )
        analysis_data["_outcome"] = analysis_data["_outcome"].astype(float)
        analysis_data["_factor_1"] = analysis_data["_factor_1"].astype(str)
        analysis_data["_factor_2"] = analysis_data["_factor_2"].astype(str)

        factor_1_levels = analysis_data["_factor_1"].nunique()
        factor_2_levels = analysis_data["_factor_2"].nunique()
        if factor_1_levels < 2 or factor_2_levels < 2:
            st.error("Each factor must contain at least two observed levels.")
            return
        if analysis_data["_outcome"].nunique() < 2:
            st.error("The numeric outcome must contain variation.")
            return

        cell_counts = (
            analysis_data
            .groupby(["_factor_1", "_factor_2"], observed=True)
            .size()
        )
        expected_cell_count = factor_1_levels * factor_2_levels
        if len(cell_counts) < expected_cell_count:
            st.error(
                "Every Factor 1 × Factor 2 combination must contain observations. "
                f"Found {len(cell_counts)} of {expected_cell_count} required cells."
            )
            return
        if int(cell_counts.min()) < 2:
            st.error(
                "Every factor combination needs at least two observations so the "
                "interaction can be tested with residual error."
            )
            return

        balanced_design = cell_counts.nunique() == 1
        sum_squares_type = 2 if balanced_design else 3
        model, anova_table, factor_1_term, factor_2_term = _fit_two_way_anova(
            analysis_data,
            sum_squares_type,
        )
        interaction_term = f"{factor_1_term}:{factor_2_term}"
        residual_sum_squares = float(anova_table.loc["Residual", "sum_sq"])

        effect_terms = {
            factor_1: factor_1_term,
            factor_2: factor_2_term,
            f"{factor_1} × {factor_2}": interaction_term,
        }
        result_rows = []
        for effect_label, term in effect_terms.items():
            effect_sum_squares = float(anova_table.loc[term, "sum_sq"])
            partial_eta_squared = (
                effect_sum_squares / (effect_sum_squares + residual_sum_squares)
                if effect_sum_squares + residual_sum_squares > 0
                else np.nan
            )
            result_rows.append({
                "Effect": effect_label,
                "Sum of squares": effect_sum_squares,
                "df": float(anova_table.loc[term, "df"]),
                "F": float(anova_table.loc[term, "F"]),
                "p-value": float(anova_table.loc[term, "PR(>F)"]),
                "Partial eta squared": partial_eta_squared,
            })

        st.info(
            f"Used **Type {sum_squares_type} sums of squares** because the "
            f"factorial design is {'balanced' if balanced_design else 'unbalanced'}."
        )
        st.markdown("### ANOVA Results")
        st.dataframe(
            pd.DataFrame(result_rows),
            hide_index=True,
            width="stretch",
            column_config={
                "Sum of squares": st.column_config.NumberColumn(format="%.4f"),
                "df": st.column_config.NumberColumn(format="%.2f"),
                "F": st.column_config.NumberColumn(format="%.4f"),
                "p-value": st.column_config.NumberColumn(format="%.4f"),
                "Partial eta squared": st.column_config.NumberColumn(format="%.4f"),
            },
        )

        metric_columns = st.columns(3)
        for metric_column, result in zip(metric_columns, result_rows):
            with metric_column:
                st.markdown(f"**{result['Effect']}**")
                st.metric("p-value", f"{result['p-value']:.4f}")
                st.metric(
                    "Partial η²",
                    f"{result['Partial eta squared']:.4f}",
                )
                if result["p-value"] <= ALPHA:
                    st.success("Significant effect detected")
                else:
                    st.info("No significant effect detected")

        interaction_result = result_rows[2]
        if interaction_result["p-value"] <= ALPHA:
            st.warning(
                "The interaction is significant. Interpret each main effect "
                "cautiously because one factor's effect changes across levels of "
                "the other factor."
            )
        else:
            st.info(
                "No significant interaction was detected. Main effects can be "
                "interpreted without evidence that they change across the other factor."
            )

        cell_samples = [
            group["_outcome"].to_numpy()
            for _, group in analysis_data.groupby(
                ["_factor_1", "_factor_2"],
                observed=True,
            )
        ]
        levene_statistic, levene_p = stats.levene(*cell_samples, center="median")
        shapiro_residuals = (
            pd.Series(model.resid).sample(5000, random_state=0)
            if len(model.resid) > 5000
            else model.resid
        )
        shapiro_statistic, shapiro_p = stats.shapiro(shapiro_residuals)
        st.markdown("### Assumption Checks")
        assumption_columns = st.columns(2)
        assumption_columns[0].metric(
            "Residual normality (Shapiro p)",
            f"{shapiro_p:.4f}",
        )
        assumption_columns[1].metric(
            "Equal cell variance (Levene p)",
            f"{levene_p:.4f}",
        )
        if shapiro_p <= ALPHA:
            st.warning("Residuals show significant evidence against normality.")
        if levene_p <= ALPHA:
            st.warning("Cell variances differ significantly.")

        means = (
            analysis_data
            .groupby(["_factor_1", "_factor_2"], observed=True)["_outcome"]
            .mean()
            .reset_index()
        )
        interaction_figure = go.Figure()
        for factor_2_level, group in means.groupby("_factor_2", sort=False):
            interaction_figure.add_trace(
                go.Scatter(
                    x=group["_factor_1"],
                    y=group["_outcome"],
                    mode="lines+markers",
                    name=str(factor_2_level),
                )
            )
        interaction_figure.update_layout(
            title=f"Interaction Plot: {factor_1} × {factor_2}",
            xaxis_title=factor_1,
            yaxis_title=f"Mean {outcome}",
            legend_title=factor_2,
            height=440,
            margin={"l": 30, "r": 20, "t": 60, "b": 30},
        )
        st.markdown("### Visual Inspection")
        st.plotly_chart(
            interaction_figure,
            width="stretch",
            key="two_way_anova_interaction_plot",
        )
        st.plotly_chart(
            _two_way_diagnostic_figure(model, factor_1, factor_2),
            width="stretch",
            key="two_way_anova_diagnostics",
        )
        st.caption(
            "Parallel interaction lines support little interaction; clearly "
            "non-parallel or crossing lines suggest that factor effects may depend "
            "on each other. Residuals should show no systematic fitted-value pattern "
            "and should approximately follow the Q-Q reference line."
        )


def test_repeated_anova(df: pd.DataFrame) -> None:
    """Run a one-factor repeated measures ANOVA using long-format data."""
    st.markdown("## 🟦 Repeated Measures ANOVA")
    st.write(
        "Compares three or more related conditions. Data must be in long format "
        "with subject, condition, and numeric outcome columns."
    )
    numeric_columns = _require_numeric(df)
    if numeric_columns is None or len(df.columns) < 3:
        st.error("Subject, condition, and numeric outcome columns are required.")
        return
    outcome = st.selectbox(
        "Numeric outcome:",
        numeric_columns,
        key="repeated_anova_outcome",
    )
    subject_options = [column for column in df.columns if column != outcome]
    subject = st.selectbox(
        "Subject identifier:",
        subject_options,
        key="repeated_anova_subject",
    )
    condition_options = [
        column for column in df.columns if column not in {outcome, subject}
    ]
    condition = st.selectbox(
        "Within-subject condition:",
        condition_options,
        key="repeated_anova_condition",
    )
    if st.button(
        "Run Repeated Measures ANOVA",
        type="primary",
        key="run_repeated_anova",
    ):
        data = df[[subject, condition, outcome]].dropna().copy()
        if data[condition].nunique() < 3:
            st.error("The condition variable must contain at least three levels.")
            return
        if data[subject].nunique() < 2:
            st.error("At least two subjects are required.")
            return
        try:
            result = AnovaRM(
                data.rename(columns={outcome: "_outcome", subject: "_subject", condition: "_condition"}),
                depvar="_outcome",
                subject="_subject",
                within=["_condition"],
                aggregate_func="mean",
            ).fit()
        except ValueError as error:
            st.error(f"Repeated measures ANOVA could not be calculated: {error}")
            return
        table = result.anova_table.reset_index(names="Effect")
        st.dataframe(table, use_container_width=True)
        p_value = float(table.loc[0, "Pr > F"])
        statistic = float(table.loc[0, "F Value"])
        _show_statistic("F-statistic", statistic, p_value)
        _show_inference(
            p_value,
            "The repeated conditions differ significantly",
            "No significant repeated-condition difference was detected",
        )

