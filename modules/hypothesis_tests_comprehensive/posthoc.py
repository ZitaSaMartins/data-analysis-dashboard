"""Shared post-hoc / pairwise-comparison engine used by ANOVA, Kruskal-Wallis and Mood's median."""

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests

from ._shared import ALPHA


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
        pivot = max(candidates | excluded, key=lambda label: len(candidates & compatible[label]))
        for label in candidates - compatible[pivot]:
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
                if standard_error == 0:
                    raw_p = 1.0 if difference == 0 else 0.0
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
    if method == "conover" and sum(len(values) for values in grouped_values.values()) <= len(grouped_values):
        st.warning("Conover-Iman requires positive residual degrees of freedom; choose Dunn instead.")
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

