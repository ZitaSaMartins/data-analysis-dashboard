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

from modules.stat_format import (
    format_alpha,
    format_p_value,
    select_significance_level,
    with_formatted_p_values,
)

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
    alpha = select_significance_level()
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
            alpha,
        )


def test_independent_ttest(df: pd.DataFrame) -> None:
    """Run an independent samples t-test with Cohen's d."""
    st.markdown("## 🟦 Independent Samples t-Test")
    st.write("Compares the means of two independent groups.")
    inputs = _two_group_inputs(df, "independent_t")
    if inputs is None:
        return
    variables, grouping_variable, group_1, group_2 = inputs
    equal_variance = st.checkbox(
        "Assume equal variances",
        value=True,
        key="independent_t_equal_variance",
    )
    alpha = select_significance_level()

    if st.button(
        "Run Independent t-Test",
        type="primary",
        key="run_independent_t",
    ):
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
            if len(values_1) < 2 or len(values_2) < 2:
                st.warning(f"'{variable}': each group needs at least two observations; skipped.")
                continue

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
            is_significant = np.isfinite(p_value) and p_value <= alpha
            summary_rows.append({
                "Variable": variable,
                f"{group_1} mean": values_1.mean(),
                f"{group_2} mean": values_2.mean(),
                "Cohen's d": cohens_d,
                "t-statistic": statistic,
                "p-value": p_value,
                "Interpretation": (
                    "Group means differ significantly ✅"
                    if is_significant
                    else "No significant difference between group means ❌"
                ),
            })
        if not summary_rows:
            st.error("No valid t-test results were produced.")
            return
        st.markdown("#### Summary")
        st.dataframe(
            with_formatted_p_values(pd.DataFrame(summary_rows), ["p-value"]),
            hide_index=True,
            width="stretch",
            column_config={
                f"{group_1} mean": st.column_config.NumberColumn(format="%.4f"),
                f"{group_2} mean": st.column_config.NumberColumn(format="%.4f"),
                "Cohen's d": st.column_config.NumberColumn(format="%.4f"),
                "t-statistic": st.column_config.NumberColumn(format="%.4f"),
            },
        )
        st.caption(f"Interpretation uses the selected significance level α = {format_alpha(alpha)}.")


def test_paired_ttest(df: pd.DataFrame) -> None:
    """Run a paired samples t-test."""
    st.markdown("## 🟦 Paired Samples t-Test")
    st.write("Compares two or more related numeric measurements from the same cases.")
    columns = _paired_columns(df, "paired_t")
    if columns is None:
        return
    first, seconds = columns
    alpha = select_significance_level()
    if st.button("Run Paired t-Test", type="primary", key="run_paired_t"):
        if not seconds:
            st.error("Select at least one second measurement.")
            return
        summary_rows: list[dict[str, object]] = []
        for second in seconds:
            paired = df[[first, second]].dropna().astype(float)
            if len(paired) < 2:
                st.warning(f"'{second}': at least two complete pairs are required; skipped.")
                continue
            statistic, p_value = stats.ttest_rel(paired[first], paired[second])
            mean_difference = (paired[first] - paired[second]).mean()
            is_significant = np.isfinite(p_value) and p_value <= alpha
            summary_rows.append({
                "Variable": second,
                "Mean difference": mean_difference,
                "t-statistic": statistic,
                "p-value": p_value,
                "Interpretation": (
                    f"Significantly different from {first} ✅"
                    if is_significant
                    else f"No significant difference from {first} ❌"
                ),
            })
        if not summary_rows:
            st.error("No valid paired t-test results were produced.")
            return
        st.markdown("#### Summary")
        st.dataframe(
            with_formatted_p_values(pd.DataFrame(summary_rows), ["p-value"]),
            hide_index=True,
            width="stretch",
            column_config={
                "Mean difference": st.column_config.NumberColumn(format="%.4f"),
                "t-statistic": st.column_config.NumberColumn(format="%.4f"),
            },
        )
        st.caption(
            f"Each row compares {first} to the listed variable. Interpretation "
            f"uses the selected significance level α = {format_alpha(alpha)}."
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
    alpha = select_significance_level()
    if st.button("Run One-Way ANOVA", type="primary", key="run_anova"):
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
            is_significant = np.isfinite(p_value) and p_value <= alpha
            summary_rows.append({
                "Variable": variable,
                "Groups": len(grouped_values),
                "F-statistic": statistic,
                "Eta squared": eta_squared,
                "p-value": p_value,
                "Interpretation": (
                    "At least one group mean differs significantly ✅"
                    if is_significant
                    else "No significant difference among group means ❌"
                ),
            })
            pairwise_inputs[variable] = grouped_values

        if not summary_rows:
            st.error("No valid ANOVA results were produced.")
            return

        st.markdown("#### Summary")
        st.dataframe(
            with_formatted_p_values(pd.DataFrame(summary_rows), ["p-value"]),
            hide_index=True,
            width="stretch",
            column_config={
                "F-statistic": st.column_config.NumberColumn(format="%.4f"),
                "Eta squared": st.column_config.NumberColumn(format="%.4f"),
            },
        )
        st.caption(f"Interpretation uses the selected significance level α = {format_alpha(alpha)}.")

        st.markdown("#### Pairwise comparisons by variable")
        for index, (variable, grouped_values) in enumerate(pairwise_inputs.items()):
            _show_variable_section_header(variable, index)
            _display_pairwise_results(grouped_values, pairwise_method, "Mean", alpha)


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
        "and whether the factors interact in their effect on one or more numeric "
        "outcomes."
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
    outcome_column, factor_1_column, factor_2_column = st.columns(3)
    with outcome_column:
        outcomes = st.multiselect(
            "Numeric outcome(s):",
            numeric_columns,
            default=numeric_columns[:1],
            key="two_way_anova_outcomes",
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
    alpha = select_significance_level()

    if st.button(
        "Run Two-Way ANOVA",
        type="primary",
        key="run_two_way_anova",
    ):
        if not outcomes:
            st.error("Select at least one numeric outcome.")
            return
        summary_rows: list[dict[str, object]] = []
        outcome_details: dict[str, dict[str, object]] = {}
        for outcome in outcomes:
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
                st.warning(f"'{outcome}': each factor must contain at least two observed levels; skipped.")
                continue
            if analysis_data["_outcome"].nunique() < 2:
                st.warning(f"'{outcome}': the numeric outcome must contain variation; skipped.")
                continue

            cell_counts = (
                analysis_data
                .groupby(["_factor_1", "_factor_2"], observed=True)
                .size()
            )
            expected_cell_count = factor_1_levels * factor_2_levels
            if len(cell_counts) < expected_cell_count:
                st.warning(
                    f"'{outcome}': every Factor 1 × Factor 2 combination must contain "
                    f"observations. Found {len(cell_counts)} of {expected_cell_count} "
                    "required cells; skipped."
                )
                continue
            if int(cell_counts.min()) < 2:
                st.warning(
                    f"'{outcome}': every factor combination needs at least two "
                    "observations so the interaction can be tested with residual "
                    "error; skipped."
                )
                continue

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

            significant_effects = [
                result["Effect"]
                for result in result_rows
                if np.isfinite(result["p-value"]) and result["p-value"] <= alpha
            ]
            summary_rows.append({
                "Variable": outcome,
                "Design": f"Type {sum_squares_type} ({'balanced' if balanced_design else 'unbalanced'})",
                f"{factor_1} p-value": result_rows[0]["p-value"],
                f"{factor_2} p-value": result_rows[1]["p-value"],
                "Interaction p-value": result_rows[2]["p-value"],
                "Significant effects": ", ".join(significant_effects) if significant_effects else "None",
            })
            outcome_details[outcome] = {
                "sum_squares_type": sum_squares_type,
                "balanced_design": balanced_design,
                "result_rows": result_rows,
                "analysis_data": analysis_data,
                "model": model,
            }

        if not summary_rows:
            st.error("No valid Two-Way ANOVA results were produced.")
            return

        st.markdown("#### Summary")
        summary_df = pd.DataFrame(summary_rows)
        p_value_columns = [
            column for column in summary_df.columns if column.endswith("p-value")
        ]
        st.dataframe(
            with_formatted_p_values(summary_df, p_value_columns),
            hide_index=True,
            width="stretch",
        )
        st.caption(f"Interpretation uses the selected significance level α = {format_alpha(alpha)}.")

        st.markdown("#### Detailed results by variable")
        for index, (outcome, details) in enumerate(outcome_details.items()):
            _show_variable_section_header(outcome, index)
            _render_two_way_anova_details(
                outcome=outcome,
                factor_1=factor_1,
                factor_2=factor_2,
                alpha=alpha,
                **details,
            )


def _render_two_way_anova_details(
    outcome: str,
    factor_1: str,
    factor_2: str,
    alpha: float,
    sum_squares_type: int,
    balanced_design: bool,
    result_rows: list[dict[str, object]],
    analysis_data: pd.DataFrame,
    model,
) -> None:
    st.info(
        f"Used **Type {sum_squares_type} sums of squares** because the "
        f"factorial design is {'balanced' if balanced_design else 'unbalanced'}."
    )
    st.markdown("### ANOVA Results")
    st.dataframe(
        with_formatted_p_values(pd.DataFrame(result_rows), ["p-value"]),
        hide_index=True,
        width="stretch",
        column_config={
            "Sum of squares": st.column_config.NumberColumn(format="%.4f"),
            "df": st.column_config.NumberColumn(format="%.2f"),
            "F": st.column_config.NumberColumn(format="%.4f"),
            "Partial eta squared": st.column_config.NumberColumn(format="%.4f"),
        },
    )

    metric_columns = st.columns(3)
    for metric_column, result in zip(metric_columns, result_rows, strict=False):
        with metric_column:
            st.markdown(f"**{result['Effect']}**")
            st.metric("p-value", format_p_value(result['p-value']))
            st.metric(
                "Partial η²",
                f"{result['Partial eta squared']:.4f}",
            )
            if result["p-value"] <= alpha:
                st.success("Significant effect detected")
            else:
                st.info("No significant effect detected")

    interaction_result = result_rows[2]
    if interaction_result["p-value"] <= alpha:
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
        format_p_value(shapiro_p),
    )
    assumption_columns[1].metric(
        "Equal cell variance (Levene p)",
        format_p_value(levene_p),
    )
    if shapiro_p <= alpha:
        st.warning("Residuals show significant evidence against normality.")
    if levene_p <= alpha:
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
        key=f"two_way_anova_interaction_plot_{outcome}",
    )
    st.plotly_chart(
        _two_way_diagnostic_figure(model, factor_1, factor_2),
        width="stretch",
        key=f"two_way_anova_diagnostics_{outcome}",
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
        "Compares three or more related conditions for one or more numeric "
        "outcomes. Data must be in long format with subject, condition, and "
        "numeric outcome columns."
    )
    numeric_columns = _require_numeric(df)
    if numeric_columns is None or len(df.columns) < 3:
        st.error("Subject, condition, and numeric outcome columns are required.")
        return
    outcomes = st.multiselect(
        "Numeric outcome(s):",
        numeric_columns,
        default=numeric_columns[:1],
        key="repeated_anova_outcomes",
    )
    subject_options = [column for column in df.columns if column not in outcomes]
    subject = st.selectbox(
        "Subject identifier:",
        subject_options,
        key="repeated_anova_subject",
    )
    condition_options = [
        column for column in df.columns if column not in {*outcomes, subject}
    ]
    condition = st.selectbox(
        "Within-subject condition:",
        condition_options,
        key="repeated_anova_condition",
    )
    alpha = select_significance_level()
    if st.button(
        "Run Repeated Measures ANOVA",
        type="primary",
        key="run_repeated_anova",
    ):
        if not outcomes:
            st.error("Select at least one numeric outcome.")
            return
        summary_rows: list[dict[str, object]] = []
        for outcome in outcomes:
            data = df[[subject, condition, outcome]].dropna().copy()
            if data[condition].nunique() < 3:
                st.warning(f"'{outcome}': the condition variable must contain at least three levels; skipped.")
                continue
            if data[subject].nunique() < 2:
                st.warning(f"'{outcome}': at least two subjects are required; skipped.")
                continue
            try:
                result = AnovaRM(
                    data.rename(columns={outcome: "_outcome", subject: "_subject", condition: "_condition"}),
                    depvar="_outcome",
                    subject="_subject",
                    within=["_condition"],
                    aggregate_func="mean",
                ).fit()
            except ValueError as error:
                st.warning(f"'{outcome}' could not be tested: {error}")
                continue
            table = result.anova_table.reset_index(names="Effect")
            p_value = float(table.loc[0, "Pr > F"])
            statistic = float(table.loc[0, "F Value"])
            is_significant = np.isfinite(p_value) and p_value <= alpha
            summary_rows.append({
                "Variable": outcome,
                "F-statistic": statistic,
                "p-value": p_value,
                "Interpretation": (
                    "Repeated conditions differ significantly ✅"
                    if is_significant
                    else "No significant repeated-condition difference ❌"
                ),
            })
        if not summary_rows:
            st.error("No valid repeated measures ANOVA results were produced.")
            return
        st.markdown("#### Summary")
        st.dataframe(
            with_formatted_p_values(pd.DataFrame(summary_rows), ["p-value"]),
            hide_index=True,
            width="stretch",
            column_config={"F-statistic": st.column_config.NumberColumn(format="%.4f")},
        )
        st.caption(f"Interpretation uses the selected significance level α = {format_alpha(alpha)}.")


