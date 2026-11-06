"""Smoke tests for modules.model_recommender."""

import pandas as pd

from modules.model_recommender import (
    recommend_classification,
    recommend_eda,
    recommend_hypothesis_tests,
    recommend_models,
    recommend_regression,
    recommend_sensory,
)


def _recommendation_names(recommendations):
    return {item["model"] for item in recommendations}


def test_recommend_models_dispatches_by_analysis_type():
    df = pd.DataFrame({"a": range(20)})
    recommendations = recommend_models("EDA", df, [])
    assert recommendations == recommend_eda(df, [])


def test_recommend_models_unknown_analysis_type_returns_empty_list():
    df = pd.DataFrame({"a": range(5)})
    assert recommend_models("Not A Real Type", df, []) == []


def test_recommend_eda_returns_nonempty_recommendations_with_required_keys():
    df = pd.DataFrame({"a": range(10)})
    recommendations = recommend_eda(df, [])
    assert len(recommendations) > 0
    for rec in recommendations:
        assert {"model", "suitability", "reason", "pros", "cons", "when_to_use"} <= rec.keys()


def test_recommend_hypothesis_tests_returns_nonempty_recommendations():
    df = pd.DataFrame({"a": range(10)})
    recommendations = recommend_hypothesis_tests(df, [])
    assert len(recommendations) > 0


def test_recommend_classification_sorted_descending_by_suitability():
    df = pd.DataFrame({"a": range(200)})
    recommendations = recommend_classification(df, n_samples=200, n_features=5, characteristics=[])
    suitabilities = [rec["suitability"] for rec in recommendations]
    assert suitabilities == sorted(suitabilities, reverse=True)


def test_recommend_classification_small_sample_lowers_random_forest_suitability():
    df = pd.DataFrame({"a": range(10)})
    small = {rec["model"]: rec["suitability"] for rec in recommend_classification(df, 10, 3, [])}
    large = {rec["model"]: rec["suitability"] for rec in recommend_classification(df, 500, 3, [])}
    assert small["Random Forest"] < large["Random Forest"]


def test_recommend_classification_imbalanced_classes_lowers_random_forest_suitability():
    df = pd.DataFrame({"a": range(200)})
    baseline = {rec["model"]: rec["suitability"] for rec in recommend_classification(df, 200, 5, [])}
    imbalanced = {
        rec["model"]: rec["suitability"]
        for rec in recommend_classification(df, 200, 5, ["Imbalanced classes"])
    }
    assert imbalanced["Random Forest"] == baseline["Random Forest"] - 1


def test_recommend_regression_sorted_descending_by_suitability():
    df = pd.DataFrame({"a": range(200)})
    recommendations = recommend_regression(df, n_samples=200, n_features=5, characteristics=[])
    suitabilities = [rec["suitability"] for rec in recommendations]
    assert suitabilities == sorted(suitabilities, reverse=True)


def test_recommend_regression_nonlinear_characteristic_favors_random_forest():
    df = pd.DataFrame({"a": range(200)})
    recommendations = recommend_regression(df, 200, 5, ["Non-linear relationships"])
    by_model = {rec["model"]: rec["suitability"] for rec in recommendations}
    assert by_model["Random Forest Regression"] == 10


def test_recommend_sensory_returns_nonempty_recommendations():
    df = pd.DataFrame({"a": range(10)})
    recommendations = recommend_sensory(df, [])
    assert len(recommendations) > 0
    assert all("model" in rec for rec in recommendations)
