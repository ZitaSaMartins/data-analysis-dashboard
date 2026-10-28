"""Legacy "Test Distribution" (normality) page."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats


def run_test_distribution(df: pd.DataFrame):
    """Normality testing: Shapiro-Wilk or an alternative, chosen by sample size."""
    st.markdown("### 📐 Test Distribution (Normality)")
    st.caption("Checks whether a numeric variable follows a normal distribution — many parametric "
               "tests rely on this assumption.")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.error("No numeric columns found.")
        return

    selected_cols = st.multiselect(
        "Select numeric variable(s) to test:",
        numeric_cols,
        default=numeric_cols[:min(3, len(numeric_cols))],
        key="dist_vars"
    )

    test_options = {
        "Auto (recommended based on sample size)": "auto",
        "Shapiro-Wilk — best for small/medium samples (n < 50)": "shapiro",
        "D'Agostino-Pearson K² — good for larger samples (n ≥ 20)": "dagostino",
        "Anderson-Darling — more sensitive to tail/outlier deviations": "anderson",
        "Kolmogorov-Smirnov — for large samples, less powerful": "ks",
    }
    test_label = st.selectbox("Normality test:", list(test_options.keys()), key="dist_test_choice")
    test_choice = test_options[test_label]

    with st.expander("ℹ️ Which test should I use?"):
        st.markdown("""
        - **Shapiro-Wilk**: the most widely recommended test for small-to-moderate samples
          (roughly n < 50, works up to ~5000). Most powerful in this range.
        - **D'Agostino-Pearson K²**: combines skewness and kurtosis; a solid general-purpose
          choice, especially for larger samples (n ≥ 20).
        - **Anderson-Darling**: gives more weight to the tails of the distribution; useful when
          outliers/tail behavior matter.
        - **Kolmogorov-Smirnov**: compares your data to a normal curve; less powerful than the
          above, mostly used for large samples.
        - **Auto**: picks Shapiro-Wilk for n < 50, otherwise D'Agostino-Pearson K².
        """)

    if not selected_cols:
        return

    if st.button("Run Distribution Test", type="primary", key="run_distribution"):
        st.session_state.distribution_test_result = None
        results = []
        saved_results = []
        for col in selected_cols:
            data = df[col].dropna()
            n = len(data)
            if n < 3:
                st.warning(f"'{col}' has fewer than 3 observations — skipped.")
                continue

            chosen = test_choice
            if chosen == "auto":
                chosen = "shapiro" if n < 50 else "dagostino"

            crit_5pct = None
            if chosen == "shapiro":
                stat, p_value = stats.shapiro(data)
                test_name = "Shapiro-Wilk"
            elif chosen == "dagostino":
                if n < 20:
                    st.warning(f"'{col}': D'Agostino-Pearson needs n ≥ 20; using Shapiro-Wilk instead.")
                    stat, p_value = stats.shapiro(data)
                    test_name = "Shapiro-Wilk (fallback)"
                else:
                    stat, p_value = stats.normaltest(data)
                    test_name = "D'Agostino-Pearson K²"
            elif chosen == "anderson":
                res = stats.anderson(data, dist='norm')
                stat = res.statistic
                crit_5pct = res.critical_values[2]  # 5% significance level
                p_value = None
                test_name = "Anderson-Darling"
            else:
                stat, p_value = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
                test_name = "Kolmogorov-Smirnov"

            if p_value is not None:
                if not np.isfinite(stat) or not np.isfinite(p_value):
                    st.warning(f"'{col}' produced an invalid test result — skipped.")
                    continue
                is_normal = "Yes" if p_value > 0.05 else "No"
                results.append({
                    'Variable': col, 'n': n, 'Test': test_name,
                    'Statistic': round(stat, 4), 'p-value': round(p_value, 4),
                    'Normal (α=0.05)': is_normal
                })
                saved_results.append({
                    "variable": col,
                    "is_normal": is_normal == "Yes",
                    "test_name": test_name,
                    "reason": (
                        f"{test_name}: p = {p_value:.4f} "
                        f"({'>' if p_value > 0.05 else '≤'} 0.05)."
                    ),
                })
            else:
                if not np.isfinite(stat) or not np.isfinite(crit_5pct):
                    st.warning(f"'{col}' produced an invalid test result — skipped.")
                    continue
                is_normal = "Yes" if stat < crit_5pct else "No"
                results.append({
                    'Variable': col, 'n': n, 'Test': test_name,
                    'Statistic': round(stat, 4), 'Critical value (5%)': round(crit_5pct, 4),
                    'Normal (5% level)': is_normal
                })
                saved_results.append({
                    "variable": col,
                    "is_normal": is_normal == "Yes",
                    "test_name": test_name,
                    "reason": (
                        f"{test_name}: statistic = {stat:.4f} "
                        f"({'<' if stat < crit_5pct else '≥'} "
                        f"critical value {crit_5pct:.4f})."
                    ),
                })

        if not results:
            return

        st.session_state.distribution_test_result = {
            "variables": saved_results,
            "alpha": 0.05,
        }
        st.dataframe(pd.DataFrame(results), use_container_width=True)

        st.markdown("#### Visual check")
        for col in selected_cols:
            data = df[col].dropna()
            if len(data) < 3:
                continue
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].hist(data, bins=20, color='steelblue', edgecolor='black', alpha=0.75)
            axes[0].set_title(f'Histogram: {col}')
            stats.probplot(data, dist="norm", plot=axes[1])
            axes[1].set_title(f'Q-Q Plot: {col}')
            plt.tight_layout()
            st.pyplot(fig)

