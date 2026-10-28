"""Normality / distribution assumption checks."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats

from ._shared import ALPHA, _require_numeric, _select_assumption_variables


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
        st.session_state.distribution_test_result = None
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
                if sample_size < 4:
                    st.warning(f"'{column}': the corrected KS test requires at least four observations.")
                    continue
                statistic, p_value = sm.stats.lilliefors(values, dist="norm")
                test_name = "Kolmogorov-Smirnov (Lilliefors correction)"

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

