"""Smoke tests for scripts.generate_sample_data.

Calls the individual ``make_*`` generators directly (rather than ``main()``,
which writes to the real ``synthetic_test_data/`` paths) to check shapes,
expected columns, determinism, and that each dataset's intentional effect
(baked in when the sample data was designed) is actually detectable.
"""

import numpy as np
import pandas as pd
from scipy import stats

from scripts.generate_sample_data import (
    SEED,
    make_cata,
    make_discrimination,
    make_hedonic_consumer,
    make_jar,
    make_panel_profile,
    make_pls,
    make_product_profile_wide,
    make_ranking,
    make_repeated_measures,
    make_student_performance,
    make_temporal,
)


def _rng():
    return np.random.default_rng(SEED)


def test_generation_is_deterministic_for_a_fixed_seed():
    df_a = make_student_performance(_rng())
    df_b = make_student_performance(_rng())
    pd.testing.assert_frame_equal(df_a, df_b)


def test_student_performance_has_expected_shape_and_significant_group_effect():
    df = make_student_performance(_rng())
    assert len(df) == 500
    assert {"StudyGroup", "TeachingMethod", "ExamScore", "PassFail"} <= set(df.columns)

    groups = [g["ExamScore"].to_numpy() for _, g in df.groupby("TeachingMethod")]
    _, p_value = stats.f_oneway(*groups)
    assert p_value < 0.01


def test_repeated_measures_wide_and_long_are_consistent_and_show_improvement():
    wide, long = make_repeated_measures(_rng())
    assert set(wide["ParticipantID"]) == set(long["ParticipantID"])
    assert len(long) == len(wide) * 4

    _, p_value = stats.ttest_rel(wide["Baseline"], wide["Week12"])
    assert p_value < 0.01
    assert wide["Week12"].mean() < wide["Baseline"].mean()


def test_discrimination_data_is_above_chance():
    df = make_discrimination(_rng())
    correct = (df["Response"] == "Correct").sum()
    assert stats.binomtest(correct, len(df), 1 / 3, alternative="greater").pvalue < 0.01


def test_ranking_data_recovers_a_consistent_product_order():
    df = make_ranking(_rng())
    pivot = df.pivot_table(index="Assessor", columns="Product", values="Rank")
    statistic, p_value = stats.friedmanchisquare(*[pivot[c] for c in pivot.columns])
    assert p_value < 0.01


def test_panel_profile_shows_significant_product_differences():
    df = make_panel_profile(_rng())
    groups = [g["Overall_Liking"].to_numpy() for _, g in df.groupby("Product")]
    _, p_value = stats.f_oneway(*groups)
    assert p_value < 0.01


def test_product_profile_wide_has_two_numeric_clusters():
    df = make_product_profile_wide(_rng())
    assert len(df) == 10
    numeric_columns = df.columns.drop("Product")
    assert len(numeric_columns) >= 2


def test_temporal_products_have_distinct_peak_intensity():
    df = make_temporal(_rng())
    peaks = df.groupby("Product")["Intensity"].max()
    assert peaks.max() - peaks.min() > 1.0


def test_cata_binary_attributes_and_liking_vary_by_product():
    df = make_cata(_rng())
    attribute_columns = [c for c in df.columns if c not in {"Consumer", "Product", "Liking"}]
    assert len(attribute_columns) == 10
    assert set(df[attribute_columns[0]].unique()) <= {0, 1}

    groups = [g["Liking"].to_numpy() for _, g in df.groupby("Product")]
    _, p_value = stats.f_oneway(*groups)
    assert p_value < 0.01


def test_jar_liking_peaks_near_the_jar_midpoint():
    df = make_jar(_rng())
    mean_by_jar = df.groupby("Sweetness_JAR")["Overall_Liking"].mean()
    assert mean_by_jar.idxmax() == 3


def test_hedonic_consumer_shows_segment_preference_differences():
    df = make_hedonic_consumer(_rng())
    pivot = df.pivot_table(index="Consumer", columns="Product", values="Liking")
    statistic, p_value = stats.friedmanchisquare(*[pivot[c] for c in pivot.columns])
    assert p_value < 0.01


def test_pls_predictors_correlate_with_intended_responses():
    df = make_pls(_rng())
    sweetener_correlation = df["SweetenerLevel"].corr(df["SweetnessPerception"])
    fat_correlation = df["FatContent"].corr(df["Creaminess"])
    assert sweetener_correlation > 0.5
    assert fat_correlation > 0.5
