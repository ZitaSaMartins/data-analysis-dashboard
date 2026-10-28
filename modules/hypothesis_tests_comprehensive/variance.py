"""Equal-variance assumption checks."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats

from ._shared import ALPHA, _require_categorical, _require_numeric, _select_assumption_variables


def _groups_are_normal(samples: list[pd.Series]) -> bool | None:
    """Check whether every supplied group is individually consistent with normality.

    Bartlett's assumption is normality within each compared group, not in the
    pooled outcome, so this tests the same per-group samples used for the
    variance test rather than relying on the earlier pooled distribution check.
    Returns ``None`` when at least one group cannot be tested.
    """
    for sample in samples:
        values = sample.dropna().astype(float)
        if len(values) < 3 or values.std(ddof=1) == 0:
            return None
        test = stats.shapiro if len(values) < 50 else stats.normaltest
        _, p_value = test(values)
        if p_value <= ALPHA:
            return False
    return True


def _suggest_variance_test(
    variable: str,
    samples: list[pd.Series],
) -> tuple[str, str]:
    if len(samples) < 2:
        return "Unavailable", "At least two groups with two observations are required."

    groups_normal = _groups_are_normal(samples)
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

    if groups_normal is False or has_extreme_outliers:
        return (
            "Fligner-Killeen",
            "Non-normality or extreme outliers were detected; use the most robust option.",
        )
    if groups_normal is True and minimum_group_size >= 5:
        return (
            "Bartlett",
            "Each group is individually consistent with normality and has at least five observations.",
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
            - **Automatic:** checks normality within each group and for extreme
              outliers. It chooses Bartlett when every group is individually
              consistent with normality, Fligner-Killeen when non-normality or
              extreme outliers are detected, and Levene when assumptions remain
              uncertain.

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
        st.session_state.variance_test_result = None
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

