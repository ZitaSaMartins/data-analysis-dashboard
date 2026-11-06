"""Smoke tests for modules.sensory_analysis._cata_math (pure statistical helpers)."""

import numpy as np
import pandas as pd

from modules.sensory_analysis._cata_math import (
    bootstrap_mean_drop,
    correspondence_analysis,
    pcoa,
)


def test_correspondence_analysis_returns_expected_shapes_and_inertia():
    counts = pd.DataFrame(
        {
            "Sweet": [40, 5, 20],
            "Bitter": [5, 35, 15],
            "Crunchy": [20, 15, 30],
        },
        index=["Prod1", "Prod2", "Prod3"],
    )
    row_coords, col_coords, inertia_fraction = correspondence_analysis(counts)
    assert list(row_coords.index) == list(counts.index)
    assert list(col_coords.index) == list(counts.columns)
    # Inertia fractions should sum to (approximately) 1 and be non-negative.
    assert np.isclose(inertia_fraction.sum(), 1.0)
    assert (inertia_fraction >= 0).all()


def test_pcoa_recovers_known_euclidean_configuration():
    # Four points on a unit square; pairwise Euclidean distances are known exactly.
    points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    distance = np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=-1))
    coordinates, fraction = pcoa(distance)
    assert coordinates.shape[0] == 4
    # The square is 2-dimensional, so only the first two axes should carry inertia.
    assert fraction[0] > 0
    assert fraction[1] > 0
    assert np.allclose(fraction[2:], 0, atol=1e-8)


def test_bootstrap_mean_drop_detects_a_clear_difference():
    rng = np.random.default_rng(0)
    reference = rng.normal(8, 0.5, size=200)
    penalty = rng.normal(4, 0.5, size=200)
    mean_drop, ci_low, ci_high, p_value = bootstrap_mean_drop(reference, penalty, rng, n_resamples=2000)
    assert mean_drop > 3
    assert ci_low < mean_drop < ci_high
    assert p_value < 0.05


def test_bootstrap_mean_drop_no_difference_is_not_significant():
    rng = np.random.default_rng(1)
    reference = rng.normal(5, 1.0, size=200)
    penalty = rng.normal(5, 1.0, size=200)
    _, _, _, p_value = bootstrap_mean_drop(reference, penalty, rng, n_resamples=2000)
    assert p_value > 0.05
