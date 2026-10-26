"""Comprehensive, student-friendly statistical tests for the dashboard."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests


ALPHA = 0.05


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()


def _categorical_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()


def _show_inference(
    p_value: float,
    significant_text: str,
    nonsignificant_text: str,
) -> None:
    if not np.isfinite(p_value):
        st.error("The test produced an invalid p-value. Check the selected data.")
    elif p_value <= ALPHA:
        st.success(f"✅ {significant_text} (p ≤ {ALPHA}).")
    else:
        st.info(f"❌ {nonsignificant_text} (p > {ALPHA}).")


def _show_statistic(
    statistic_label: str,
    statistic: float,
    p_value: float,
) -> None:
    statistic_col, p_value_col = st.columns(2)
    statistic_col.metric(statistic_label, f"{statistic:.4f}")
    p_value_col.metric("p-value", f"{p_value:.4f}")


def _require_numeric(df: pd.DataFrame, minimum: int = 1) -> list[str] | None:
    columns = _numeric_columns(df)
    if len(columns) < minimum:
        st.error(f"This test requires at least {minimum} numeric variable(s).")
        return None
    return columns


def _require_categorical(df: pd.DataFrame, minimum: int = 1) -> list[str] | None:
    columns = _categorical_columns(df)
    if len(columns) < minimum:
        st.error(f"This test requires at least {minimum} categorical variable(s).")
        return None
    return columns


def _two_group_inputs(
    df: pd.DataFrame,
    key_prefix: str,
) -> tuple[str, str, object, object] | None:
    numeric_columns = _require_numeric(df)
    categorical_columns = _require_categorical(df)
    if numeric_columns is None or categorical_columns is None:
        return None

    numeric_variable = st.selectbox(
        "Numeric outcome variable:",
        numeric_columns,
        key=f"{key_prefix}_numeric",
    )
    grouping_variable = st.selectbox(
        "Grouping variable:",
        categorical_columns,
        key=f"{key_prefix}_grouping",
    )
    groups = df[grouping_variable].dropna().unique().tolist()
    if len(groups) < 2:
        st.error("The grouping variable must contain at least two groups.")
        return None

    group_col1, group_col2 = st.columns(2)
    group_1 = group_col1.selectbox(
        "Group 1:",
        groups,
        key=f"{key_prefix}_group_1",
    )
    remaining_groups = [group for group in groups if group != group_1]
    group_2 = group_col2.selectbox(
        "Group 2:",
        remaining_groups,
        key=f"{key_prefix}_group_2",
    )
    return numeric_variable, grouping_variable, group_1, group_2


def _paired_columns(
    df: pd.DataFrame,
    key_prefix: str,
) -> tuple[str, str] | None:
    numeric_columns = _require_numeric(df, minimum=2)
    if numeric_columns is None:
        return None
    col1, col2 = st.columns(2)
    first = col1.selectbox(
        "First measurement:",
        numeric_columns,
        key=f"{key_prefix}_first",
    )
    second_options = [column for column in numeric_columns if column != first]
    second = col2.selectbox(
        "Second measurement:",
        second_options,
        key=f"{key_prefix}_second",
    )
    return first, second


def _suggest_normality_test(sample_size: int) -> tuple[str, str]:
    if sample_size < 8:
        return (
            "Shapiro-Wilk",
            "Only a very small sample is available; combine the test with visual inspection.",
        )
    if sample_size < 50:
        return (
            "Shapiro-Wilk",
            "Recommended for small samples and generally powerful for detecting non-normality.",
        )
    return (
        "D'Agostino-Pearson K²",
        "The sample is large enough to assess skewness and kurtosis reliably.",
    )


def _distribution_diagnostic_figure(values: pd.Series, variable: str) -> go.Figure:
    data = values.to_numpy(dtype=float)
    figure = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Histogram", "Normal Q-Q Plot", "Box Plot"),
        column_widths=[0.4, 0.4, 0.2],
    )
    figure.add_trace(
        go.Histogram(
            x=data,
            histnorm="probability density",
            marker_color="#4C78A8",
            opacity=0.78,
            name="Observed",
        ),
        row=1,
        col=1,
    )

    standard_deviation = float(np.std(data, ddof=1))
    if standard_deviation > 0:
        x_values = np.linspace(float(np.min(data)), float(np.max(data)), 200)
        normal_density = stats.norm.pdf(
            x_values,
            loc=float(np.mean(data)),
            scale=standard_deviation,
        )
        figure.add_trace(
            go.Scatter(
                x=x_values,
                y=normal_density,
                mode="lines",
                line={"color": "#E45756", "width": 3},
                name="Fitted normal curve",
            ),
            row=1,
            col=1,
        )

    (theoretical, ordered), (slope, intercept, _) = stats.probplot(
        data,
        dist="norm",
        fit=True,
    )
    figure.add_trace(
        go.Scatter(
            x=theoretical,
            y=ordered,
            mode="markers",
            marker={"color": "#4C78A8", "size": 7},
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
    figure.add_trace(
        go.Box(
            y=data,
            boxpoints="outliers",
            marker_color="#4C78A8",
            name=variable,
        ),
        row=1,
        col=3,
    )
    figure.update_xaxes(title_text=variable, row=1, col=1)
    figure.update_xaxes(title_text="Theoretical quantiles", row=1, col=2)
    figure.update_yaxes(title_text="Density", row=1, col=1)
    figure.update_yaxes(title_text="Observed quantiles", row=1, col=2)
    figure.update_layout(
        height=420,
        margin={"l": 30, "r": 20, "t": 70, "b": 30},
        showlegend=False,
        title={"text": variable, "x": 0.02},
    )
    return figure


def _store_assumption_variables(widget_key: str) -> None:
    st.session_state.assumption_variables = list(
        st.session_state.get(widget_key, [])
    )


def _select_assumption_variables(
    numeric_columns: list[str],
    widget_key: str,
    legacy_key: str,
) -> list[str]:
    if "assumption_variables" not in st.session_state:
        legacy_selection = st.session_state.get(legacy_key)
        st.session_state.assumption_variables = list(
            legacy_selection
            if legacy_selection is not None
            else numeric_columns[: min(3, len(numeric_columns))]
        )

    selected_variables = [
        variable
        for variable in st.session_state.assumption_variables
        if variable in numeric_columns
    ]
    st.session_state.assumption_variables = selected_variables
    st.session_state[widget_key] = selected_variables
    return st.multiselect(
        "Numeric variables:",
        numeric_columns,
        key=widget_key,
        on_change=_store_assumption_variables,
        args=(widget_key,),
    )


def run_distribution_tests(df: pd.DataFrame) -> None:
    """Run a selected normality test and show visual distribution diagnostics."""
    st.markdown("## 📐 Test Distribution")
    st.write(
        "Checks whether selected numeric variables are consistent with a normal "
        "distribution. Use the statistical result together with the visual checks."
    )
    numeric_columns = _require_numeric(df)
    if numeric_columns is None:
        return

    selected_columns = _select_assumption_variables(
        numeric_columns,
        "_comprehensive_distribution_variables",
        "comprehensive_distribution_variables",
    )
    if not selected_columns:
        st.info("Select at least one numeric variable.")
        return

    test_options = {
        "Automatic — recommended test for each sample size": "automatic",
        "Shapiro-Wilk — strongest choice for small samples": "shapiro",
        "D'Agostino-Pearson K² — skewness and kurtosis for larger samples": "dagostino",
        "Anderson-Darling — emphasizes tails and outliers": "anderson",
        "Kolmogorov-Smirnov — exploratory comparison with a normal curve": "ks",
    }
    selected_test_label = st.selectbox(
        "Normality test:",
        list(test_options),
        key="comprehensive_distribution_test",
    )
    selected_test = test_options[selected_test_label]

    guidance = []
    for column in selected_columns:
        sample_size = int(df[column].notna().sum())
        suggested_test, reason = _suggest_normality_test(sample_size)
        guidance.append({
            "Variable": column,
            "Usable observations": sample_size,
            "Suggested test": suggested_test,
            "Why": reason,
        })
    st.markdown("#### Suggested test for the selected variables")
    st.dataframe(
        pd.DataFrame(guidance),
        hide_index=True,
        width="stretch",
    )

    with st.expander("ℹ️ How should I choose a test?"):
        st.markdown(
            """
            - **Shapiro-Wilk:** preferred for small samples and valid from 3
              observations. For very small samples, its power is limited, so inspect
              the plots as well.
            - **D'Agostino-Pearson K²:** evaluates skewness and kurtosis. Use it with
              at least 20 observations; it is especially useful for larger samples.
            - **Anderson-Darling:** gives extra weight to distribution tails. Choose it
              when tail behavior or extreme observations are important.
            - **Kolmogorov-Smirnov:** compares the sample with a fitted normal curve.
              Because the mean and standard deviation are estimated from the same
              sample, treat its p-value as exploratory rather than definitive.
            - **Automatic:** uses Shapiro-Wilk below 50 observations and
              D'Agostino-Pearson K² from 50 observations upward.

            Large samples can make minor, unimportant deviations statistically
            significant. Always review the histogram, Q-Q plot, and box plot.
            """
        )

    if st.button(
        "Run Distribution Tests",
        type="primary",
        key="run_comprehensive_distribution",
    ):
        results: list[dict[str, object]] = []
        saved_results: list[dict[str, object]] = []
        plotted_values: dict[str, pd.Series] = {}
        for column in selected_columns:
            values = df[column].dropna().astype(float)
            sample_size = len(values)
            if sample_size < 3:
                st.warning(f"'{column}' has fewer than three observations and was skipped.")
                continue
            plotted_values[column] = values
            if values.std(ddof=1) == 0:
                st.warning(f"'{column}' is constant and cannot be tested for normality.")
                continue

            test_to_run = selected_test
            if test_to_run == "automatic":
                test_to_run = "shapiro" if sample_size < 50 else "dagostino"

            critical_value: float | None = None
            if test_to_run == "shapiro":
                tested_values = (
                    values.sample(5000, random_state=0)
                    if sample_size > 5000
                    else values
                )
                if sample_size > 5000:
                    st.info(
                        f"'{column}': Shapiro-Wilk used a reproducible sample of "
                        "5,000 observations because its p-value is less reliable above 5,000."
                    )
                statistic, p_value = stats.shapiro(tested_values)
                test_name = "Shapiro-Wilk"
            elif test_to_run == "dagostino":
                if sample_size < 20:
                    st.warning(
                        f"'{column}': D'Agostino-Pearson is recommended for n ≥ 20; "
                        "Shapiro-Wilk was used instead."
                    )
                    statistic, p_value = stats.shapiro(values)
                    test_name = "Shapiro-Wilk (fallback)"
                else:
                    statistic, p_value = stats.normaltest(values)
                    test_name = "D'Agostino-Pearson K²"
            elif test_to_run == "anderson":
                anderson_result = stats.anderson(values, dist="norm")
                statistic = float(anderson_result.statistic)
                five_percent_index = int(
                    np.argmin(np.abs(anderson_result.significance_level - 5.0))
                )
                critical_value = float(
                    anderson_result.critical_values[five_percent_index]
                )
                p_value = None
                test_name = "Anderson-Darling"
            else:
                standardized = (values - values.mean()) / values.std(ddof=1)
                statistic, p_value = stats.kstest(standardized, "norm")
                test_name = "Kolmogorov-Smirnov"

            if p_value is None:
                if critical_value is None:
                    st.error(f"'{column}' did not produce a critical value.")
                    continue
                supports_normality = statistic < critical_value
                reason = (
                    f"statistic {statistic:.4f} "
                    f"{'<' if supports_normality else '≥'} "
                    f"5% critical value {critical_value:.4f}"
                )
            else:
                if not np.isfinite(p_value):
                    st.warning(f"'{column}' produced an invalid p-value and was skipped.")
                    continue
                supports_normality = p_value > ALPHA
                reason = (
                    f"p = {p_value:.4f} "
                    f"{'>' if supports_normality else '≤'} {ALPHA}"
                )

            results.append({
                "Variable": column,
                "n": sample_size,
                "Test": test_name,
                "Statistic": statistic,
                "p-value": p_value,
                "5% critical value": critical_value,
                "Interpretation": (
                    "Consistent with normality ✅"
                    if supports_normality
                    else "Evidence against normality ❌"
                ),
            })
            saved_results.append({
                "variable": column,
                "is_normal": supports_normality,
                "test_name": test_name,
                "reason": f"{test_name}: {reason}.",
            })

        if not results:
            st.error("No valid distribution test results were produced.")
            return
        st.session_state.distribution_test_result = {
            "variables": saved_results,
            "alpha": ALPHA,
        }
        result_df = pd.DataFrame(results)
        st.dataframe(
            result_df,
            hide_index=True,
            width="stretch",
            column_config={
                "Statistic": st.column_config.NumberColumn(format="%.4f"),
                "p-value": st.column_config.NumberColumn(format="%.4f"),
                "5% critical value": st.column_config.NumberColumn(format="%.4f"),
            },
        )
        st.caption(
            "Interpretation: p > 0.05 means the test did not find significant "
            "evidence against normality. For Anderson-Darling, the statistic must "
            "be below the 5% critical value. Neither result proves perfect normality."
        )

        st.markdown("### 👁️ Visual Distribution Checks")
        st.caption(
            "A normal distribution is approximately bell-shaped, follows the Q-Q "
            "reference line, and has no severe asymmetry or extreme outliers."
        )
        plot_tabs = st.tabs(list(plotted_values))
        for tab, (column, values) in zip(plot_tabs, plotted_values.items()):
            with tab:
                st.plotly_chart(
                    _distribution_diagnostic_figure(values, column),
                    width="stretch",
                    key=f"distribution_diagnostics_{column}",
                )


def _suggest_variance_test(
    variable: str,
    samples: list[pd.Series],
) -> tuple[str, str]:
    if len(samples) < 2:
        return "Unavailable", "At least two groups with two observations are required."

    distribution_results = st.session_state.get("distribution_test_result")
    normality_by_variable = {
        result["variable"]: result["is_normal"]
        for result in (distribution_results or {}).get("variables", [])
    }
    minimum_group_size = min(len(sample) for sample in samples)
    has_extreme_outliers = False
    for sample in samples:
        first_quartile, third_quartile = np.percentile(sample, [25, 75])
        interquartile_range = third_quartile - first_quartile
        if interquartile_range == 0:
            continue
        lower_limit = first_quartile - 3 * interquartile_range
        upper_limit = third_quartile + 3 * interquartile_range
        if ((sample < lower_limit) | (sample > upper_limit)).any():
            has_extreme_outliers = True
            break

    if normality_by_variable.get(variable) is False or has_extreme_outliers:
        return (
            "Fligner-Killeen",
            "Non-normality or extreme outliers were detected; use the most robust option.",
        )
    if normality_by_variable.get(variable) is True and minimum_group_size >= 5:
        return (
            "Bartlett",
            "The distribution check supports normality and every group has at least five observations.",
        )
    if minimum_group_size < 5:
        return (
            "Levene",
            "At least one group is small; use the robust default and inspect the plots.",
        )
    return (
        "Levene",
        "Normality has not been established, so the robust default is recommended.",
    )


def _variance_diagnostic_figure(
    grouped_values: dict[str, pd.Series],
    variable: str,
    grouping_variable: str,
) -> go.Figure:
    group_labels: list[str] = []
    combined_values: list[float] = []
    variances: list[float] = []
    for group, values in grouped_values.items():
        group_labels.extend([group] * len(values))
        combined_values.extend(values.astype(float).tolist())
        variances.append(float(values.var(ddof=1)))

    figure = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Grouped Box Plot", "Grouped Violin Plot", "Sample Variance"),
        column_widths=[0.36, 0.36, 0.28],
    )
    figure.add_trace(
        go.Box(
            x=group_labels,
            y=combined_values,
            boxpoints="outliers",
            marker_color="#4C78A8",
            name="Box plot",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Violin(
            x=group_labels,
            y=combined_values,
            box_visible=True,
            meanline_visible=True,
            points=False,
            line_color="#72B7B2",
            name="Violin plot",
        ),
        row=1,
        col=2,
    )
    figure.add_trace(
        go.Bar(
            x=list(grouped_values),
            y=variances,
            marker_color="#F58518",
            text=[f"{variance:.3f}" for variance in variances],
            textposition="outside",
            name="Variance",
        ),
        row=1,
        col=3,
    )
    figure.update_xaxes(title_text=grouping_variable, row=1, col=1)
    figure.update_xaxes(title_text=grouping_variable, row=1, col=2)
    figure.update_xaxes(title_text=grouping_variable, row=1, col=3)
    figure.update_yaxes(title_text=variable, row=1, col=1)
    figure.update_yaxes(title_text=variable, row=1, col=2)
    figure.update_yaxes(title_text="Variance", row=1, col=3)
    figure.update_layout(
        height=440,
        margin={"l": 30, "r": 20, "t": 70, "b": 30},
        showlegend=False,
        title={"text": variable, "x": 0.02},
    )
    return figure


def run_variance_tests(df: pd.DataFrame) -> None:
    """Run a selected equality-of-variance test with visual diagnostics."""
    st.markdown("## 📏 Test Variance")
    st.write(
        "Checks whether selected numeric variables have similar variance across "
        "groups. Use the statistical result together with the visual comparisons."
    )
    numeric_columns = _require_numeric(df)
    categorical_columns = _require_categorical(df)
    if numeric_columns is None or categorical_columns is None:
        return

    selected_columns = _select_assumption_variables(
        numeric_columns,
        "_comprehensive_variance_variables",
        "comprehensive_variance_variables",
    )
    grouping_variable = st.selectbox(
        "Grouping variable:",
        categorical_columns,
        key="comprehensive_variance_group",
    )
    if not selected_columns:
        st.info("Select at least one numeric variable.")
        return

    group_names = df[grouping_variable].dropna().unique().tolist()
    if len(group_names) < 2:
        st.error("The grouping variable must contain at least two groups.")
        return

    samples_by_variable: dict[str, dict[str, pd.Series]] = {}
    guidance = []
    for column in selected_columns:
        grouped_values = {
            str(group): df.loc[
                df[grouping_variable] == group,
                column,
            ].dropna().astype(float)
            for group in group_names
        }
        grouped_values = {
            group: values
            for group, values in grouped_values.items()
            if len(values) >= 2
        }
        samples_by_variable[column] = grouped_values
        suggested_test, reason = _suggest_variance_test(
            column,
            list(grouped_values.values()),
        )
        group_sizes = [len(values) for values in grouped_values.values()]
        guidance.append({
            "Variable": column,
            "Usable groups": len(grouped_values),
            "Smallest group": min(group_sizes) if group_sizes else 0,
            "Largest group": max(group_sizes) if group_sizes else 0,
            "Suggested test": suggested_test,
            "Why": reason,
        })

    test_options = {
        "Automatic — recommended test for each variable": "automatic",
        "Levene — robust recommended default": "levene",
        "Bartlett — best when every group is approximately normal": "bartlett",
        "Fligner-Killeen — most robust to non-normality and outliers": "fligner",
    }
    selected_test_label = st.selectbox(
        "Variance test:",
        list(test_options),
        key="comprehensive_variance_test",
    )
    selected_test = test_options[selected_test_label]

    st.markdown("#### Suggested test for the selected variables")
    st.dataframe(
        pd.DataFrame(guidance),
        hide_index=True,
        width="stretch",
    )

    with st.expander("ℹ️ How should I choose a test?"):
        st.markdown(
            """
            - **Levene:** the robust general-purpose choice. Use it when normality
              has not been established, group sizes are small, or assumptions are
              uncertain.
            - **Bartlett:** usually the most powerful option when every group is
              approximately normal. It is sensitive to non-normality and outliers.
            - **Fligner-Killeen:** a rank-based, highly robust option for clearly
              non-normal data or influential outliers.
            - **Automatic:** uses prior results from **Test Distribution**, group
              sizes, and extreme-outlier checks. It chooses Bartlett when normality
              is supported, Fligner-Killeen when non-normality or extreme outliers
              are detected, and Levene when assumptions remain uncertain.

            A non-significant result does not prove identical variances. Review the
            grouped box plot, violin plot, and variance bars for practically important
            differences.
            """
        )

    if st.button(
        "Run Variance Tests",
        type="primary",
        key="run_comprehensive_variance",
    ):
        results: list[dict[str, object]] = []
        saved_results: list[dict[str, object]] = []
        group_summaries: list[dict[str, object]] = []
        plotted_values: dict[str, dict[str, pd.Series]] = {}
        for column in selected_columns:
            grouped_values = samples_by_variable[column]
            samples = list(grouped_values.values())
            if len(samples) < 2:
                st.warning(
                    f"'{column}' needs at least two groups with two observations each."
                )
                continue
            plotted_values[column] = grouped_values

            test_to_run = selected_test
            if test_to_run == "automatic":
                suggested_test, _ = _suggest_variance_test(column, samples)
                test_to_run = {
                    "Bartlett": "bartlett",
                    "Fligner-Killeen": "fligner",
                    "Levene": "levene",
                }[suggested_test]

            if test_to_run == "bartlett":
                statistic, p_value = stats.bartlett(*samples)
                test_name = "Bartlett"
            elif test_to_run == "fligner":
                statistic, p_value = stats.fligner(*samples)
                test_name = "Fligner-Killeen"
            else:
                statistic, p_value = stats.levene(*samples, center="median")
                test_name = "Levene"

            if not np.isfinite(statistic) or not np.isfinite(p_value):
                st.warning(
                    f"{test_name} was invalid for '{column}'. Check for constant "
                    "groups or insufficient variation."
                )
                continue

            supports_equal_variance = p_value > ALPHA
            results.append({
                "Variable": column,
                "Groups": len(grouped_values),
                "Test": test_name,
                "Statistic": statistic,
                "p-value": p_value,
                "Interpretation": (
                    "No significant variance difference ✅"
                    if supports_equal_variance
                    else "Variances differ significantly ❌"
                ),
            })
            saved_results.append({
                "variable": column,
                "group_count": len(grouped_values),
                "has_equal_variance": supports_equal_variance,
                "test_name": test_name,
                "p_value": p_value,
                "reason": (
                    f"{test_name}: p = {p_value:.4f} "
                    f"({'>' if supports_equal_variance else '≤'} {ALPHA})."
                ),
            })
            for group, values in grouped_values.items():
                group_summaries.append({
                    "Variable": column,
                    "Group": group,
                    "n": len(values),
                    "Variance": values.var(ddof=1),
                    "Std. deviation": values.std(ddof=1),
                })

        if not results:
            st.error("No valid variance test results were produced.")
            return
        st.session_state.variance_test_result = {
            "variables": saved_results,
            "grouping_variable": grouping_variable,
            "alpha": ALPHA,
        }
        result_df = pd.DataFrame(results)
        st.dataframe(
            result_df,
            hide_index=True,
            width="stretch",
            column_config={
                "Statistic": st.column_config.NumberColumn(format="%.4f"),
                "p-value": st.column_config.NumberColumn(format="%.4f"),
            },
        )
        st.caption(
            "Interpretation: p > 0.05 means the test did not find a significant "
            "variance difference; it does not prove that group variances are identical."
        )

        st.markdown("#### Group variance details")
        st.dataframe(
            pd.DataFrame(group_summaries),
            hide_index=True,
            width="stretch",
            column_config={
                "Variance": st.column_config.NumberColumn(format="%.4f"),
                "Std. deviation": st.column_config.NumberColumn(format="%.4f"),
            },
        )

        st.markdown("### 👁️ Visual Variance Checks")
        st.caption(
            "Similar vertical spread and similarly sized variance bars support the "
            "equal-variance assumption. Also inspect outliers and group sample sizes."
        )
        plot_tabs = st.tabs(list(plotted_values))
        for tab, (column, grouped_values) in zip(
            plot_tabs,
            plotted_values.items(),
        ):
            with tab:
                st.plotly_chart(
                    _variance_diagnostic_figure(
                        grouped_values,
                        column,
                        grouping_variable,
                    ),
                    width="stretch",
                    key=f"variance_diagnostics_{column}",
                )


def _inference_recommendations(
    is_normal: bool,
    has_equal_variance: bool,
    group_count: int,
) -> dict[str, str]:
    if not is_normal:
        return {
            "family": "Non-parametric",
            "primary_test": (
                "Mann-Whitney U" if group_count == 2 else "Kruskal-Wallis"
            ),
            "primary_page": (
                "nonparametric_mannwhitney"
                if group_count == 2
                else "nonparametric_kruskal_wallis"
            ),
            "primary_reason": (
                "The distribution result indicates that the normality assumption "
                "is not supported."
            ),
            "related_test": (
                "Wilcoxon signed-rank" if group_count == 2 else "Friedman"
            ),
            "relationship_test": "Spearman correlation or Kendall's tau",
            "one_sample_test": "One-sample Wilcoxon signed-rank",
        }

    if not has_equal_variance and group_count == 2:
        return {
            "family": "Parametric with unequal-variance adjustment",
            "primary_test": "Welch independent t-test",
            "primary_page": "parametric_ttest_independent",
            "primary_reason": (
                "Normality is supported, but group variances differ. Open the "
                "independent t-test and turn off 'Assume equal variances'."
            ),
            "related_test": "Paired t-test, if observations are matched",
            "relationship_test": "Pearson correlation",
            "one_sample_test": "One-sample t-test",
        }

    if not has_equal_variance:
        return {
            "family": "Robust non-parametric",
            "primary_test": "Kruskal-Wallis",
            "primary_page": "nonparametric_kruskal_wallis",
            "primary_reason": (
                "Normality is supported, but variances differ across three or more "
                "groups. Kruskal-Wallis is the robust option available in this dashboard."
            ),
            "related_test": "Repeated-measures ANOVA, if observations are matched",
            "relationship_test": "Pearson correlation",
            "one_sample_test": "One-sample t-test",
        }

    return {
        "family": "Parametric",
        "primary_test": (
            "Independent t-test" if group_count == 2 else "One-way ANOVA"
        ),
        "primary_page": (
            "parametric_ttest_independent"
            if group_count == 2
            else "parametric_anova"
        ),
        "primary_reason": (
            "Both normality and equal-variance assumptions are supported by the "
            "latest checks."
        ),
        "related_test": (
            "Paired t-test" if group_count == 2 else "Repeated-measures ANOVA"
        ),
        "relationship_test": "Pearson correlation",
        "one_sample_test": "One-sample t-test",
    }


def _prepare_recommended_test(
    page: str,
    variable: str,
    grouping_variable: str,
    assume_equal_variance: bool = True,
) -> None:
    test_widget_keys = {
        "parametric_ttest_independent": (
            "independent_t_numeric",
            "independent_t_grouping",
        ),
        "parametric_anova": ("anova_numeric_variables", "anova_group"),
        "nonparametric_mannwhitney": (
            "mannwhitney_numeric",
            "mannwhitney_grouping",
        ),
        "nonparametric_kruskal_wallis": (
            "kruskal_numeric_variables",
            "kruskal_group",
        ),
    }
    numeric_key, grouping_key = test_widget_keys[page]
    if page in {"parametric_anova", "nonparametric_kruskal_wallis"}:
        st.session_state[numeric_key] = [variable]
    else:
        st.session_state[numeric_key] = variable
    st.session_state[grouping_key] = grouping_variable
    if page == "parametric_ttest_independent":
        st.session_state.independent_t_equal_variance = assume_equal_variance


def show_suggested_analysis(df: pd.DataFrame) -> None:
    """Recommend inference tests from the latest assumption-check results."""
    st.markdown("## 🧭 Suggested Analysis")
    st.write(
        "Combines the latest distribution and variance results to suggest inference "
        "tests that can be carried out in this dashboard."
    )

    distribution_result = st.session_state.get("distribution_test_result") or {}
    variance_result = st.session_state.get("variance_test_result") or {}
    distribution_by_variable = {
        result["variable"]: result
        for result in distribution_result.get("variables", [])
    }
    variance_by_variable = {
        result["variable"]: result
        for result in variance_result.get("variables", [])
    }
    assumption_variables = [
        variable
        for variable in st.session_state.get("assumption_variables", [])
        if variable in df.columns
    ]
    if not assumption_variables:
        assumption_variables = list(dict.fromkeys([
            *distribution_by_variable,
            *variance_by_variable,
        ]))
        assumption_variables = [
            variable for variable in assumption_variables if variable in df.columns
        ]
    if not assumption_variables:
        st.info(
            "Select numeric variables in Test Distribution or Test Variance first. "
            "The same selection will be used by both pages and by Suggested Analysis."
        )
        return

    current_selection = st.session_state.get("_suggested_analysis_variables")
    if current_selection is None:
        st.session_state._suggested_analysis_variables = assumption_variables
    else:
        st.session_state._suggested_analysis_variables = [
            variable
            for variable in current_selection
            if variable in assumption_variables
        ]
    selected_variables = st.multiselect(
        "Variables to include:",
        assumption_variables,
        key="_suggested_analysis_variables",
    )
    if not selected_variables:
        st.info("Select at least one variable to review.")
        return

    missing_distribution = [
        variable
        for variable in selected_variables
        if variable not in distribution_by_variable
    ]
    missing_variance = [
        variable
        for variable in selected_variables
        if variable not in variance_by_variable
    ]
    if missing_distribution or missing_variance:
        st.warning(
            "A complete recommendation requires both assumption checks for every "
            "selected variable."
        )
        action_columns = st.columns(2)
        with action_columns[0]:
            if missing_distribution:
                st.error(
                    "**Distribution analysis missing:** "
                    + ", ".join(missing_distribution)
                )
                if st.button(
                    "Complete Test Distribution",
                    key="suggested_open_distribution",
                    width="stretch",
                ):
                    st.session_state.page = "test_distribution"
                    st.rerun()
            else:
                st.success("Distribution analysis is complete.")
        with action_columns[1]:
            if missing_variance:
                st.error(
                    "**Variance analysis missing:** "
                    + ", ".join(missing_variance)
                )
                if st.button(
                    "Complete Test Variance",
                    key="suggested_open_variance",
                    width="stretch",
                ):
                    st.session_state.page = "test_variance"
                    st.rerun()
            else:
                st.success("Variance analysis is complete.")

    grouping_variable = variance_result.get(
        "grouping_variable",
        "the selected grouping variable",
    )
    summary_rows = []
    recommendations_by_variable: dict[str, dict[str, str]] = {}
    for variable in selected_variables:
        distribution = distribution_by_variable.get(variable)
        variance = variance_by_variable.get(variable)
        if distribution is None or variance is None:
            summary_rows.append({
                "Variable": variable,
                "Distribution": (
                    "Missing" if distribution is None else "Complete"
                ),
                "Variance": "Missing" if variance is None else "Complete",
                "Groups": str(variance["group_count"]) if variance else "—",
                "Suggested test": "Complete missing checks",
            })
            continue

        recommendation = _inference_recommendations(
            distribution["is_normal"],
            variance["has_equal_variance"],
            variance["group_count"],
        )
        recommendations_by_variable[variable] = recommendation
        summary_rows.append({
            "Variable": variable,
            "Distribution": (
                "Normality supported"
                if distribution["is_normal"]
                else "Normality not supported"
            ),
            "Variance": (
                "Equal variance supported"
                if variance["has_equal_variance"]
                else "Unequal variance"
            ),
            "Groups": str(variance["group_count"]),
            "Suggested test": recommendation["primary_test"],
        })

    st.markdown(f"#### Recommendation summary by `{grouping_variable}`")
    st.dataframe(
        pd.DataFrame(summary_rows),
        hide_index=True,
        width="stretch",
    )

    for variable in selected_variables:
        distribution = distribution_by_variable.get(variable)
        variance = variance_by_variable.get(variable)
        with st.container(border=True):
            st.markdown(f"### {variable}")
            distribution_column, variance_column = st.columns(2)
            with distribution_column:
                if distribution is None:
                    st.error("Distribution analysis: missing")
                elif distribution["is_normal"]:
                    st.success("Distribution: normality supported")
                else:
                    st.warning("Distribution: normality not supported")
                if distribution is not None:
                    st.caption(distribution["reason"])
            with variance_column:
                if variance is None:
                    st.error("Variance analysis: missing")
                elif variance["has_equal_variance"]:
                    st.success("Variance: no significant difference detected")
                else:
                    st.warning("Variance: significant difference detected")
                if variance is not None:
                    st.caption(variance["reason"])

            if distribution is None or variance is None:
                st.info(
                    "Complete the missing assumption analysis before selecting an "
                    "inference test for this variable."
                )
                continue

            recommendation = recommendations_by_variable[variable]
            st.markdown(
                f"**Primary recommendation: {recommendation['primary_test']}**  \n"
                f"Path: **{recommendation['family']}**  \n"
                f"{recommendation['primary_reason']}"
            )
            st.markdown(
                "**Other study designs:**\n"
                f"- Related or repeated observations: "
                f"**{recommendation['related_test']}**\n"
                f"- Association between numeric variables: "
                f"**{recommendation['relationship_test']}**\n"
                f"- One sample versus a reference value: "
                f"**{recommendation['one_sample_test']}**"
            )
            if st.button(
                f"Open {recommendation['primary_test']}",
                key=f"suggested_open_{variable}",
                type="primary",
                width="stretch",
            ):
                _prepare_recommended_test(
                    recommendation["primary_page"],
                    variable,
                    grouping_variable,
                    recommendation["primary_test"] != "Welch independent t-test",
                )
                st.session_state.page = recommendation["primary_page"]
                st.rerun()

    st.caption(
        "Recommendations assume independent groups for the primary test. Confirm "
        "your study design, observation independence, pairing, outliers, and sample "
        "size before drawing conclusions."
    )


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


def _compact_letter_display(
    group_labels: list[str],
    significant_pairs: set[tuple[str, str]],
) -> dict[str, str]:
    """Create letters where groups sharing a letter are not significantly different."""
    labels = list(group_labels)
    if not labels:
        return {}

    def pair_key(first: str, second: str) -> tuple[str, str]:
        return tuple(sorted((first, second)))

    compatible = {
        label: {
            other
            for other in labels
            if other != label and pair_key(label, other) not in significant_pairs
        }
        for label in labels
    }

    maximal_cliques: list[set[str]] = []

    def find_cliques(
        current: set[str],
        candidates: set[str],
        excluded: set[str],
    ) -> None:
        if not candidates and not excluded:
            maximal_cliques.append(current)
            return
        remaining = set(candidates)
        while remaining:
            label = next(iter(remaining))
            find_cliques(
                current | {label},
                remaining & compatible[label],
                excluded & compatible[label],
            )
            remaining.remove(label)
            excluded.add(label)

    find_cliques(set(), set(labels), set())
    if not maximal_cliques:
        maximal_cliques = [{label} for label in labels]

    all_pairs = {
        pair_key(first, second)
        for index, first in enumerate(labels)
        for second in labels[index + 1 :]
        if pair_key(first, second) not in significant_pairs
    }
    uncovered_labels = set(labels)
    uncovered_pairs = set(all_pairs)
    selected_cliques: list[set[str]] = []
    while uncovered_labels or uncovered_pairs:
        best = max(
            maximal_cliques,
            key=lambda clique: (
                len(clique & uncovered_labels)
                + sum(
                    pair_key(first, second) in uncovered_pairs
                    for index, first in enumerate(clique)
                    for second in list(clique)[index + 1 :]
                ),
                len(clique),
            ),
        )
        selected_cliques.append(best)
        uncovered_labels -= best
        uncovered_pairs -= {
            pair_key(first, second)
            for index, first in enumerate(best)
            for second in list(best)[index + 1 :]
        }
        maximal_cliques.remove(best)
        if not maximal_cliques:
            break

    letters: dict[str, list[str]] = {label: [] for label in labels}
    for index, clique in enumerate(selected_cliques):
        letter = (
            chr(ord("A") + index)
            if index < 26
            else f"A{index - 25}"
        )
        for label in clique:
            letters[label].append(letter)
    return {label: "".join(letters[label]) or "—" for label in labels}


def _run_pairwise_comparisons(
    grouped_values: dict[str, pd.Series],
    method: str,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Run a selected post-hoc method and return pairwise results and letters."""
    labels = list(grouped_values)
    pair_rows: list[dict[str, object]] = []
    significant_pairs: set[tuple[str, str]] = set()

    if method == "tukey":
        values = np.concatenate([grouped_values[label].to_numpy() for label in labels])
        groups = np.concatenate(
            [[label] * len(grouped_values[label]) for label in labels]
        )
        tukey = pairwise_tukeyhsd(values, groups, alpha=ALPHA)
        pair_indices = tukey._multicomp.pairindices
        for index, (first_index, second_index) in enumerate(
            zip(pair_indices[0], pair_indices[1])
        ):
            first = str(tukey._multicomp.groupsunique[first_index])
            second = str(tukey._multicomp.groupsunique[second_index])
            adjusted_p = float(tukey.pvalues[index])
            significant = bool(tukey.reject[index])
            pair_rows.append(
                {
                    "Group 1": first,
                    "Group 2": second,
                    "Difference": float(tukey.meandiffs[index]),
                    "p-value": adjusted_p,
                    "Significant": "Yes" if significant else "No",
                }
            )
            if significant:
                significant_pairs.add(tuple(sorted((first, second))))
    elif method in {"bonferroni", "scheffe", "fisher_lsd"}:
        sample_count = sum(len(values) for values in grouped_values.values())
        group_count = len(grouped_values)
        residual_degrees_freedom = sample_count - group_count
        residual_sum_squares = sum(
            float(((values - values.mean()) ** 2).sum())
            for values in grouped_values.values()
        )
        mean_square_error = residual_sum_squares / residual_degrees_freedom
        raw_p_values: list[float] = []
        pair_metadata: list[tuple[str, str, float]] = []
        for index, first in enumerate(labels):
            for second in labels[index + 1 :]:
                difference = float(
                    grouped_values[first].mean() - grouped_values[second].mean()
                )
                standard_error = np.sqrt(
                    mean_square_error
                    * (
                        1 / len(grouped_values[first])
                        + 1 / len(grouped_values[second])
                    )
                )
                if np.isclose(standard_error, 0):
                    raw_p = 1.0 if np.isclose(difference, 0) else 0.0
                else:
                    t_statistic = difference / standard_error
                    if method == "scheffe":
                        contrast_f = t_statistic**2
                        raw_p = stats.f.sf(
                            contrast_f / (group_count - 1),
                            group_count - 1,
                            residual_degrees_freedom,
                        )
                    else:
                        raw_p = 2 * stats.t.sf(
                            abs(t_statistic),
                            residual_degrees_freedom,
                        )
                raw_p_values.append(float(raw_p))
                pair_metadata.append((first, second, difference))

        if method == "bonferroni":
            p_values = multipletests(
                raw_p_values,
                alpha=ALPHA,
                method="bonferroni",
            )[1]
        else:
            p_values = np.asarray(raw_p_values)
        for (first, second, difference), p_value in zip(pair_metadata, p_values):
            significant = bool(p_value <= ALPHA)
            pair_rows.append(
                {
                    "Group 1": first,
                    "Group 2": second,
                    "Difference": difference,
                    "p-value": float(p_value),
                    "Significant": "Yes" if significant else "No",
                }
            )
            if significant:
                significant_pairs.add(tuple(sorted((first, second))))
    elif method == "holm_t":
        raw_p_values = []
        pair_metadata = []
        for index, first in enumerate(labels):
            for second in labels[index + 1 :]:
                _, raw_p = stats.ttest_ind(
                    grouped_values[first],
                    grouped_values[second],
                    equal_var=False,
                )
                difference = float(
                    grouped_values[first].mean() - grouped_values[second].mean()
                )
                raw_p_values.append(float(raw_p))
                pair_metadata.append((first, second, difference))
        adjusted = multipletests(raw_p_values, alpha=ALPHA, method="holm")[1]
        for (first, second, difference), adjusted_p in zip(pair_metadata, adjusted):
            significant = bool(adjusted_p <= ALPHA)
            pair_rows.append(
                {
                    "Group 1": first,
                    "Group 2": second,
                    "Difference": difference,
                    "p-value": float(adjusted_p),
                    "Significant": "Yes" if significant else "No",
                }
            )
            if significant:
                significant_pairs.add(tuple(sorted((first, second))))
    else:
        combined_values = np.concatenate(
            [grouped_values[label].to_numpy() for label in labels]
        )
        ranks = stats.rankdata(combined_values)
        sample_count = len(combined_values)
        group_count = len(labels)
        offsets = np.cumsum(
            [0] + [len(grouped_values[label]) for label in labels]
        )
        mean_ranks = {
            label: float(ranks[offsets[index] : offsets[index + 1]].mean())
            for index, label in enumerate(labels)
        }
        _, tie_counts = np.unique(combined_values, return_counts=True)
        tie_sum = float(np.sum(tie_counts**3 - tie_counts))
        tie_correction = 1 - tie_sum / (sample_count**3 - sample_count)
        base_variance = (
            sample_count * (sample_count + 1) / 12
            - tie_sum / (12 * (sample_count - 1))
        )

        if method == "conover":
            rank_square_variance = (
                float(np.sum(ranks**2))
                - sample_count * (sample_count + 1) ** 2 / 4
            ) / (sample_count - 1)
            kruskal_statistic = (
                12
                / (sample_count * (sample_count + 1))
                * sum(
                    len(grouped_values[label]) * mean_ranks[label] ** 2
                    for label in labels
                )
                - 3 * (sample_count + 1)
            ) / tie_correction
            conover_scale = (
                rank_square_variance
                * (sample_count - 1 - kruskal_statistic)
                / (sample_count - group_count)
            )

        raw_p_values = []
        pair_metadata = []
        for index, first in enumerate(labels):
            for second in labels[index + 1 :]:
                size_factor = (
                    1 / len(grouped_values[first])
                    + 1 / len(grouped_values[second])
                )
                if method == "dunn":
                    standard_error = np.sqrt(base_variance * size_factor)
                    z_statistic = (
                        abs(mean_ranks[first] - mean_ranks[second]) / standard_error
                    )
                    raw_p = 2 * stats.norm.sf(z_statistic)
                else:
                    standard_error = np.sqrt(conover_scale * size_factor)
                    t_statistic = (
                        abs(mean_ranks[first] - mean_ranks[second]) / standard_error
                    )
                    raw_p = 2 * stats.t.sf(
                        t_statistic,
                        sample_count - group_count,
                    )
                raw_p_values.append(float(raw_p))
                pair_metadata.append(
                    (
                        first,
                        second,
                        float(
                            grouped_values[first].median()
                            - grouped_values[second].median()
                        ),
                    )
                )
        adjusted = multipletests(raw_p_values, alpha=ALPHA, method="holm")[1]
        for (first, second, difference), adjusted_p in zip(pair_metadata, adjusted):
            significant = bool(adjusted_p <= ALPHA)
            pair_rows.append(
                {
                    "Group 1": first,
                    "Group 2": second,
                    "Difference": difference,
                    "p-value": float(adjusted_p),
                    "Significant": "Yes" if significant else "No",
                }
            )
            if significant:
                significant_pairs.add(tuple(sorted((first, second))))

    letters = _compact_letter_display(labels, significant_pairs)
    return pd.DataFrame(pair_rows), letters


def _show_pairwise_options(
    test_type: str,
    key_prefix: str,
) -> str:
    if test_type == "anova":
        options = {
            "Tukey HSD — recommended when variances are similar": "tukey",
            "Bonferroni — conservative control of Type I error": "bonferroni",
            "Scheffé — flexible and conservative for unbalanced designs": "scheffe",
            "Fisher's LSD — powerful but least conservative": "fisher_lsd",
            "Holm-adjusted Welch t-tests — robust to unequal variances": "holm_t",
            "No pairwise comparison": "none",
        }
        description = (
            "- **Tukey HSD:** recommended general-purpose method when ANOVA "
            "assumptions hold; controls family-wise error across all pairs.\n"
            "- **Bonferroni:** divides the error allowance across comparisons. It "
            "strongly protects against false positives but can have lower power.\n"
            "- **Scheffé:** a conservative method that remains valid for any planned "
            "or unplanned contrast and works with unequal group sizes.\n"
            "- **Fisher's LSD:** pooled pairwise t-tests without multiplicity "
            "correction. It is powerful but should normally follow a significant "
            "overall ANOVA and is less protective against false positives.\n"
            "- **Holm-adjusted Welch:** use when group variances or sample sizes "
            "differ; it does not rely on a common pooled variance."
        )
    else:
        options = {
            "Dunn test with Holm correction — recommended": "dunn",
            "Conover-Iman with Holm correction — more powerful alternative": "conover",
            "No pairwise comparison": "none",
        }
        description = (
            "- **Dunn:** standard rank-based follow-up after Kruskal-Wallis. It "
            "compares mean ranks and applies a tie correction.\n"
            "- **Conover-Iman:** also compares mean ranks but incorporates the "
            "overall Kruskal-Wallis result, often providing more power.\n\n"
            "Both options use Holm correction to control false positives across "
            "the pairwise comparisons. After Mood's median test, these methods "
            "should be interpreted as complementary rank-distribution comparisons."
        )
    selected = st.selectbox(
        "Pairwise comparison method:",
        list(options),
        key=f"{key_prefix}_pairwise_method",
    )
    with st.expander("ℹ️ How should I choose a pairwise test?"):
        st.write(description)
        st.write(
            "Grouping letters summarize the adjusted results: groups that share "
            "a letter are not significantly different at α = 0.05. Groups with "
            "no letters in common are significantly different."
        )
    return options[selected]


def _display_pairwise_results(
    grouped_values: dict[str, pd.Series],
    method: str,
    center_label: str,
) -> None:
    if method == "none":
        return
    pairwise, letters = _run_pairwise_comparisons(grouped_values, method)
    summary = pd.DataFrame(
        [
            {
                "Group": group,
                "n": len(values),
                center_label: float(
                    values.mean() if center_label == "Mean" else values.median()
                ),
                "Grouping letters": letters[group],
            }
            for group, values in grouped_values.items()
        ]
    )
    st.markdown("#### Pairwise comparisons")
    st.caption(
        "Groups sharing a letter are not significantly different after multiple-"
        "comparison correction."
    )
    st.dataframe(summary, hide_index=True, width="stretch")
    st.dataframe(pairwise, hide_index=True, width="stretch")


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
                data,
                depvar=outcome,
                subject=subject,
                within=[condition],
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


def test_pearson_correlation(df: pd.DataFrame) -> None:
    """Run Pearson's linear correlation test."""
    _test_correlation(
        df,
        method="pearson",
        title="🟦 Pearson Correlation",
        description="Measures a linear relationship between two numeric variables.",
    )


def test_linear_regression(df: pd.DataFrame) -> None:
    """Run simple linear regression for one predictor and one outcome."""
    st.markdown("## 📈 Linear Regression")
    st.write("Models a numeric outcome from one numeric predictor.")
    numeric_columns = _require_numeric(df, minimum=2)
    if numeric_columns is None:
        return
    col1, col2 = st.columns(2)
    predictor = col1.selectbox(
        "Predictor (X):",
        numeric_columns,
        key="regression_predictor",
    )
    outcome_options = [column for column in numeric_columns if column != predictor]
    outcome = col2.selectbox(
        "Outcome (Y):",
        outcome_options,
        key="regression_outcome",
    )
    if st.button("Run Linear Regression", type="primary", key="run_regression"):
        data = df[[predictor, outcome]].dropna().astype(float)
        if len(data) < 3 or data[predictor].nunique() < 2:
            st.error("At least three paired observations and two X values are required.")
            return
        result = stats.linregress(data[predictor], data[outcome])
        slope, intercept, r_squared = st.columns(3)
        slope.metric("Slope", f"{result.slope:.4f}")
        intercept.metric("Intercept", f"{result.intercept:.4f}")
        r_squared.metric("R²", f"{result.rvalue ** 2:.4f}")
        _show_statistic("t-statistic", result.slope / result.stderr, result.pvalue)
        predictions = result.intercept + result.slope * data[predictor]
        output = data.copy()
        output["Predicted"] = predictions
        output["Residual"] = data[outcome] - predictions
        st.dataframe(output.head(100), use_container_width=True)
        _show_inference(
            result.pvalue,
            "The predictor has a significant linear relationship with the outcome",
            "No significant linear relationship was detected",
        )


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
            if len(differences) < 2 or np.allclose(differences, 0):
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
            if len(differences) < 2 or np.allclose(differences, 0):
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
            statistic, p_value = stats.kruskal(*groups)
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


def test_spearman(df: pd.DataFrame) -> None:
    """Run Spearman's rank correlation."""
    _test_correlation(
        df,
        method="spearman",
        title="🟧 Spearman Correlation",
        description="Measures a monotonic rank-based relationship.",
    )


def test_kendall(df: pd.DataFrame) -> None:
    """Run Kendall's tau rank correlation."""
    _test_correlation(
        df,
        method="kendall",
        title="🟧 Kendall's Tau",
        description="Measures ordinal association and handles small samples well.",
    )


def _test_correlation(
    df: pd.DataFrame,
    method: str,
    title: str,
    description: str,
) -> None:
    st.markdown(f"## {title}")
    st.write(description)
    numeric_columns = _require_numeric(df, minimum=2)
    if numeric_columns is None:
        return
    col1, col2 = st.columns(2)
    first = col1.selectbox(
        "First variable:",
        numeric_columns,
        key=f"{method}_first",
    )
    second_options = [column for column in numeric_columns if column != first]
    second = col2.selectbox(
        "Second variable:",
        second_options,
        key=f"{method}_second",
    )
    if st.button(
        f"Run {title.removeprefix('🟦 ').removeprefix('🟧 ')}",
        type="primary",
        key=f"run_{method}",
    ):
        paired = df[[first, second]].dropna().astype(float)
        if len(paired) < 3:
            st.error("At least three complete pairs are required.")
            return
        if method == "pearson":
            statistic, p_value = stats.pearsonr(paired[first], paired[second])
            statistic_label = "Pearson r"
        elif method == "spearman":
            statistic, p_value = stats.spearmanr(paired[first], paired[second])
            statistic_label = "Spearman ρ"
        else:
            statistic, p_value = stats.kendalltau(paired[first], paired[second])
            statistic_label = "Kendall τ"
        _show_statistic(statistic_label, statistic, p_value)
        _show_inference(
            p_value,
            "A significant association was detected",
            "No significant association was detected",
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
            statistic, p_value, median, contingency = stats.median_test(*groups)
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


def test_chi_square(df: pd.DataFrame) -> None:
    """Run a chi-square test of independence for two categorical variables."""
    st.markdown("## 📋 Chi-Square Test")
    st.write("Tests whether two categorical variables are associated.")
    categorical_columns = _require_categorical(df, minimum=2)
    if categorical_columns is None:
        return
    col1, col2 = st.columns(2)
    first = col1.selectbox(
        "First categorical variable:",
        categorical_columns,
        key="chi_square_first",
    )
    second_options = [column for column in categorical_columns if column != first]
    second = col2.selectbox(
        "Second categorical variable:",
        second_options,
        key="chi_square_second",
    )
    if st.button("Run Chi-Square Test", type="primary", key="run_chi_square"):
        contingency = pd.crosstab(df[first], df[second])
        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            st.error("Both variables must contain at least two observed categories.")
            return
        statistic, p_value, degrees_freedom, expected = stats.chi2_contingency(
            contingency
        )
        statistic_col, p_value_col, df_col = st.columns(3)
        statistic_col.metric("χ² statistic", f"{statistic:.4f}")
        p_value_col.metric("p-value", f"{p_value:.4f}")
        df_col.metric("Degrees of freedom", str(degrees_freedom))
        st.markdown("**Observed counts**")
        st.dataframe(contingency, use_container_width=True)
        st.markdown("**Expected counts**")
        st.dataframe(
            pd.DataFrame(
                expected,
                index=contingency.index,
                columns=contingency.columns,
            ).round(2),
            use_container_width=True,
        )
        _show_inference(
            p_value,
            "The categorical variables are significantly associated",
            "No significant categorical association was detected",
        )
