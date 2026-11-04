"""Pure computational helpers for advanced CATA analyses (no Streamlit calls).

Kept separate from cata_jar.py so the underlying statistics can be understood
and unit-tested independently of the page-rendering code.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def correspondence_analysis(
    counts: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Simple (classical) Correspondence Analysis via SVD of chi-square standardized residuals.

    ``counts`` is a contingency table (for CATA: products x attribute citation counts).
    Returns (row_coordinates, column_coordinates, inertia_fraction), where
    ``inertia_fraction[k]`` is the share of total inertia (chi-square / N) explained
    by axis k. Row/column principal coordinates follow the standard CA convention
    (scaled by the inverse square root of their mass).
    """
    values = counts.to_numpy(dtype=float)
    total = values.sum()
    relative = values / total
    row_masses = relative.sum(axis=1)
    col_masses = relative.sum(axis=0)
    expected = np.outer(row_masses, col_masses)
    standardized_residuals = (relative - expected) / np.sqrt(expected)
    u, singular_values, vt = np.linalg.svd(standardized_residuals, full_matrices=False)
    inertia = singular_values**2
    inertia_fraction = inertia / inertia.sum() if inertia.sum() > 0 else inertia
    row_coords = (u * singular_values) / np.sqrt(row_masses)[:, None]
    col_coords = (vt.T * singular_values) / np.sqrt(col_masses)[:, None]
    axis_names = [f"Dim{i + 1}" for i in range(row_coords.shape[1])]
    return (
        pd.DataFrame(row_coords, index=counts.index, columns=axis_names),
        pd.DataFrame(col_coords, index=counts.columns, columns=axis_names),
        inertia_fraction,
    )


def pcoa(distance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Classical (Torgerson) Principal Coordinates Analysis on a distance matrix.

    Returns (coordinates, inertia_fraction). Negative eigenvalues (possible for
    non-Euclidean distances, e.g. derived from correlations) are clipped to zero.
    """
    n = distance.shape[0]
    centering = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centering @ (distance**2) @ centering
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    positive = np.clip(eigenvalues, 0, None)
    coordinates = eigenvectors * np.sqrt(positive)
    fraction = positive / positive.sum() if positive.sum() > 0 else positive
    return coordinates, fraction


def bootstrap_mean_drop(
    reference_values: np.ndarray,
    penalty_values: np.ndarray,
    random: np.random.Generator,
    n_resamples: int = 10000,
) -> tuple[float, float, float, float]:
    """Mean(reference) - mean(penalty) with a bootstrap 95% CI and two-sided p-value.

    Mirrors the JAR penalty-analysis convention already used elsewhere in this
    package: a positive drop means the reference group liked the product more
    than the penalty group. The p-value is the bootstrap proportion of resamples
    on the opposite side of zero from the observed drop (doubled for a two-sided
    test), intended to be combined with Holm correction across several attributes
    tested at once rather than read as significant on its own. Uses a vectorized
    resample so a high resample count (needed for enough p-value resolution to
    survive Holm correction across many attributes) stays fast.
    """
    mean_drop = float(reference_values.mean() - penalty_values.mean())
    reference_means = random.choice(
        reference_values, size=(n_resamples, len(reference_values)), replace=True
    ).mean(axis=1)
    penalty_means = random.choice(
        penalty_values, size=(n_resamples, len(penalty_values)), replace=True
    ).mean(axis=1)
    resampled = reference_means - penalty_means
    side_proportion = float((resampled <= 0).mean() if mean_drop > 0 else (resampled >= 0).mean())
    p_value = min(1.0, 2 * max(side_proportion, 1 / n_resamples))
    return (
        mean_drop,
        float(np.quantile(resampled, 0.025)),
        float(np.quantile(resampled, 0.975)),
        p_value,
    )
