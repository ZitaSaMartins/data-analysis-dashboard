"""Generates synthetic CSV datasets that exercise every analysis page in the
Data Analysis Dashboard (general statistics + the full Sensory Analysis suite).

Each dataset is built with a real, intentional effect baked in (a group
difference, a correlation, a product preference, etc.) so that running the
corresponding analysis on it produces a meaningful, usually statistically
significant, result rather than pure noise.

Run with:
    python scripts/generate_sample_data.py

Outputs CSV files under ``synthetic_test_data/general_stats/`` and
``synthetic_test_data/sensory_analysis/``. See ``synthetic_test_data/README.md``
for a map of which file to use for which dashboard page.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT_GENERAL = ROOT / "synthetic_test_data" / "general_stats"
OUT_SENSORY = ROOT / "synthetic_test_data" / "sensory_analysis"

SEED = 42


# ---------------------------------------------------------------------------
# General statistics
# ---------------------------------------------------------------------------


def make_student_performance(rng: np.random.Generator) -> pd.DataFrame:
    """Flagship dataset covering EDA, hypothesis tests, correlation,
    regression, chi-square, and machine learning pages."""
    n = 500

    regions = rng.choice(["North", "South", "East", "West"], size=n)
    departments = rng.choice(
        ["Math", "Science", "Engineering", "Business"], size=n, p=[0.3, 0.3, 0.25, 0.15]
    )
    study_group = rng.choice(["In-Person", "Online"], size=n, p=[0.5, 0.5])
    teaching_method = rng.choice(
        ["Lecture", "Flipped", "Blended"], size=n, p=[0.4, 0.3, 0.3]
    )

    study_hours = np.clip(rng.normal(10, 3, n), 0, None)
    prior_gpa = np.clip(rng.normal(3.0, 0.4, n), 1.5, 4.0)
    # Right-skewed attendance rate (good candidate for non-parametric tests).
    attendance_rate = np.clip(rng.beta(5, 2, n) * 100, 0, 100)
    # Skewed 1-10 satisfaction score.
    satisfaction_score = np.clip(np.round(rng.beta(4, 1.5, n) * 9 + 1), 1, 10)

    group_effect = np.where(study_group == "In-Person", 4.0, 0.0)
    method_effect = np.select(
        [teaching_method == "Flipped", teaching_method == "Blended"],
        [6.0, 3.0],
        default=0.0,
    )
    # Interaction: the Flipped method gives an extra boost specifically to
    # Online students, so a two-way ANOVA shows a real interaction effect.
    interaction_effect = np.where(
        (study_group == "Online") & (teaching_method == "Flipped"), 5.0, 0.0
    )

    noise = rng.normal(0, 6, n)
    exam_score = (
        45
        + 1.6 * study_hours
        + 8 * (prior_gpa - 3.0)
        + group_effect
        + method_effect
        + interaction_effect
        + noise
    )
    exam_score = np.clip(exam_score, 0, 100).round(1)

    pass_fail = np.where(exam_score >= 60, "Pass", "Fail")

    dates = pd.date_range("2024-01-06", periods=104, freq="W-MON")
    enrollment_date = rng.choice(dates, size=n)

    df = pd.DataFrame(
        {
            "StudentID": [f"S{i:04d}" for i in range(1, n + 1)],
            "Date": pd.to_datetime(enrollment_date),
            "Region": regions,
            "Department": departments,
            "StudyGroup": study_group,
            "TeachingMethod": teaching_method,
            "StudyHours": study_hours.round(1),
            "PriorGPA": prior_gpa.round(2),
            "AttendanceRate": attendance_rate.round(1),
            "SatisfactionScore": satisfaction_score.astype(int),
            "ExamScore": exam_score,
            "PassFail": pass_fail,
        }
    ).sort_values("Date").reset_index(drop=True)
    return df


def make_repeated_measures(rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A 12-week stress-reduction program measured at 4 time points.

    Returns (wide, long) versions of the same simulated data:
    - wide: ParticipantID, Baseline, Week4, Week8, Week12 (paired t-test,
      Wilcoxon signed-rank, Friedman)
    - long: ParticipantID, TimePoint, StressScore (Repeated Measures ANOVA)
    """
    n = 50
    baseline = np.clip(rng.normal(70, 10, n), 30, 100)
    # Each participant improves at their own rate, with diminishing returns.
    improvement_rate = np.clip(rng.normal(12, 4, n), 2, None)

    week4 = baseline - improvement_rate * 0.6 + rng.normal(0, 4, n)
    week8 = baseline - improvement_rate * 1.0 + rng.normal(0, 4, n)
    week12 = baseline - improvement_rate * 1.3 + rng.normal(0, 4, n)

    wide = pd.DataFrame(
        {
            "ParticipantID": [f"P{i:03d}" for i in range(1, n + 1)],
            "Baseline": baseline.round(1),
            "Week4": np.clip(week4, 0, 100).round(1),
            "Week8": np.clip(week8, 0, 100).round(1),
            "Week12": np.clip(week12, 0, 100).round(1),
        }
    )

    long = wide.melt(
        id_vars="ParticipantID", var_name="TimePoint", value_name="StressScore"
    )
    time_order = ["Baseline", "Week4", "Week8", "Week12"]
    long["TimePoint"] = pd.Categorical(long["TimePoint"], categories=time_order, ordered=True)
    long = long.sort_values(["ParticipantID", "TimePoint"]).reset_index(drop=True)
    return wide, long


# ---------------------------------------------------------------------------
# Sensory analysis
# ---------------------------------------------------------------------------


def make_discrimination(rng: np.random.Generator) -> pd.DataFrame:
    """Serves Triangle, Paired Comparison, Duo-Trio, and Tetrad tests.

    A single correct/incorrect response column above both the 1/3 and 1/2
    chance baselines, so all four discrimination tests show significance.
    """
    n = 60
    correct_prob = 0.72
    responses = rng.choice(
        ["Correct", "Incorrect"], size=n, p=[correct_prob, 1 - correct_prob]
    )
    return pd.DataFrame(
        {
            "Trial": range(1, n + 1),
            "Panelist": [f"J{((i - 1) % 20) + 1:02d}" for i in range(1, n + 1)],
            "Response": responses,
        }
    )


def make_ranking(rng: np.random.Generator) -> pd.DataFrame:
    """Assessor x Product ranking data for the Ranking Test."""
    products = ["Product_A", "Product_B", "Product_C", "Product_D", "Product_E"]
    assessors = [f"J{i:02d}" for i in range(1, 13)]
    # True latent quality order (lower = preferred -> rank 1).
    true_quality = {"Product_A": 1, "Product_B": 3, "Product_C": 5, "Product_D": 2, "Product_E": 4}

    rows = []
    for assessor in assessors:
        noise = rng.normal(0, 0.8, len(products))
        scores = {p: true_quality[p] + noise[i] for i, p in enumerate(products)}
        ranked = sorted(scores, key=scores.get)
        for rank, product in enumerate(ranked, start=1):
            rows.append({"Assessor": assessor, "Product": product, "Rank": rank})
    return pd.DataFrame(rows)


def make_panel_profile(rng: np.random.Generator) -> pd.DataFrame:
    """Long assessor x product x replicate sensory-profile data.

    Serves QDA / Product Characterization, Panel Discrimination, Panel
    Repeatability, Panel Agreement, Panel Scale Use, GPA, and STATIS.
    """
    assessors = [f"J{i:02d}" for i in range(1, 7)]
    products = ["P1", "P2", "P3", "P4", "P5"]
    replicates = ["Rep1", "Rep2"]
    attributes = ["Sweetness", "Bitterness", "Aroma_Intensity", "Crunchiness", "Overall_Liking"]

    # True product signature per attribute (0-10 scale).
    true_profile = {
        "P1": {"Sweetness": 7.5, "Bitterness": 2.0, "Aroma_Intensity": 6.0, "Crunchiness": 3.0, "Overall_Liking": 7.8},
        "P2": {"Sweetness": 3.0, "Bitterness": 6.5, "Aroma_Intensity": 4.5, "Crunchiness": 7.5, "Overall_Liking": 4.5},
        "P3": {"Sweetness": 5.5, "Bitterness": 4.0, "Aroma_Intensity": 7.5, "Crunchiness": 5.0, "Overall_Liking": 6.2},
        "P4": {"Sweetness": 8.5, "Bitterness": 1.0, "Aroma_Intensity": 5.0, "Crunchiness": 2.0, "Overall_Liking": 8.3},
        "P5": {"Sweetness": 2.0, "Bitterness": 7.5, "Aroma_Intensity": 3.5, "Crunchiness": 8.0, "Overall_Liking": 3.6},
    }
    # Each assessor has a consistent scale-use bias (harsh vs lenient).
    assessor_bias = {a: rng.normal(0, 0.8) for a in assessors}

    rows = []
    for assessor in assessors:
        bias = assessor_bias[assessor]
        for product in products:
            for replicate in replicates:
                row = {"Assessor": assessor, "Product": product, "Replicate": replicate}
                for attribute in attributes:
                    true_value = true_profile[product][attribute]
                    value = true_value + bias + rng.normal(0, 0.6)
                    row[attribute] = round(float(np.clip(value, 0, 10)), 2)
                rows.append(row)
    return pd.DataFrame(rows)


def make_product_profile_wide(rng: np.random.Generator) -> pd.DataFrame:
    """Product-level wide matrix with block-named attributes.

    Serves PCA, Hierarchical Clustering (HAC), and MFA.
    """
    products = [f"Product_{i:02d}" for i in range(1, 11)]
    # Two latent clusters of products (e.g. "indulgent" vs "light") so PCA/HAC
    # show clear structure.
    cluster = np.array([0] * 5 + [1] * 5)

    taste_sweet = np.where(cluster == 0, rng.normal(7.5, 0.6, 10), rng.normal(3.0, 0.6, 10))
    taste_sour = np.where(cluster == 0, rng.normal(2.5, 0.5, 10), rng.normal(5.5, 0.6, 10))
    taste_bitter = np.where(cluster == 0, rng.normal(2.0, 0.5, 10), rng.normal(6.0, 0.6, 10))
    aroma_fruity = np.where(cluster == 0, rng.normal(7.0, 0.6, 10), rng.normal(3.5, 0.6, 10))
    aroma_floral = np.where(cluster == 0, rng.normal(5.5, 0.6, 10), rng.normal(2.5, 0.6, 10))
    texture_crunchy = np.where(cluster == 0, rng.normal(3.0, 0.6, 10), rng.normal(7.0, 0.6, 10))
    texture_creamy = np.where(cluster == 0, rng.normal(7.5, 0.6, 10), rng.normal(3.0, 0.6, 10))

    df = pd.DataFrame(
        {
            "Product": products,
            "taste_sweet": taste_sweet.round(2),
            "taste_sour": taste_sour.round(2),
            "taste_bitter": taste_bitter.round(2),
            "aroma_fruity": aroma_fruity.round(2),
            "aroma_floral": aroma_floral.round(2),
            "texture_crunchy": texture_crunchy.round(2),
            "texture_creamy": texture_creamy.round(2),
        }
    )
    numeric_columns = df.columns.drop("Product")
    df[numeric_columns] = df[numeric_columns].clip(lower=0, upper=10)
    return df


def make_temporal(rng: np.random.Generator) -> pd.DataFrame:
    """Time-intensity style data for the Temporal Methods page."""
    products = {
        # name: (peak_time, decay_rate, amplitude)
        "Product_A": (12, 0.09, 9.0),   # fast burst, fades quickly
        "Product_B": (24, 0.045, 7.5),  # moderate build-up and lingering
        "Product_C": (36, 0.03, 8.5),   # slow build, long linger
        "Product_D": (18, 0.06, 6.0),   # mild, moderate profile
    }
    time_points = list(range(0, 61, 6))
    replicates = ["Rep1", "Rep2", "Rep3"]

    rows = []
    for product, (peak_time, decay_rate, amplitude) in products.items():
        for replicate in replicates:
            panelist_noise = rng.normal(0, 0.4)
            for t in time_points:
                if t <= peak_time:
                    intensity = amplitude * (t / peak_time)
                else:
                    intensity = amplitude * np.exp(-(t - peak_time) * decay_rate)
                intensity = max(0.0, intensity + panelist_noise + rng.normal(0, 0.3))
                rows.append(
                    {
                        "Product": product,
                        "Time": t,
                        "Replicate": replicate,
                        "Intensity": round(intensity, 2),
                    }
                )
    return pd.DataFrame(rows)


def make_cata(rng: np.random.Generator) -> pd.DataFrame:
    """Consumer x Product CATA data, matching the demo_CATA_EN.xlsm layout."""
    n_consumers = 60
    products = [f"Prod{i}" for i in range(1, 7)]
    attributes = [
        "Sweet", "Sour", "Bitter", "Salty", "Crunchy",
        "Soft", "Juicy", "Creamy", "Fruity", "Astringent",
    ]
    # Each product has a characteristic attribute-selection probability profile.
    product_profiles = {
        "Prod1": {"Sweet": .85, "Sour": .1, "Bitter": .05, "Salty": .1, "Crunchy": .7, "Soft": .2, "Juicy": .6, "Creamy": .2, "Fruity": .75, "Astringent": .05},
        "Prod2": {"Sweet": .2, "Sour": .15, "Bitter": .75, "Salty": .2, "Crunchy": .2, "Soft": .75, "Juicy": .15, "Creamy": .6, "Fruity": .1, "Astringent": .6},
        "Prod3": {"Sweet": .55, "Sour": .5, "Bitter": .2, "Salty": .15, "Crunchy": .5, "Soft": .4, "Juicy": .55, "Creamy": .3, "Fruity": .5, "Astringent": .2},
        "Prod4": {"Sweet": .9, "Sour": .05, "Bitter": .05, "Salty": .05, "Crunchy": .15, "Soft": .8, "Juicy": .3, "Creamy": .8, "Fruity": .2, "Astringent": .05},
        "Prod5": {"Sweet": .1, "Sour": .8, "Bitter": .3, "Salty": .1, "Crunchy": .6, "Soft": .2, "Juicy": .7, "Creamy": .1, "Fruity": .65, "Astringent": .35},
        "Prod6": {"Sweet": .3, "Sour": .2, "Bitter": .6, "Salty": .55, "Crunchy": .75, "Soft": .15, "Juicy": .1, "Creamy": .15, "Fruity": .1, "Astringent": .45},
    }
    liking_base = {"Prod1": 7.2, "Prod2": 4.6, "Prod3": 6.0, "Prod4": 7.8, "Prod5": 5.4, "Prod6": 4.2}

    rows = []
    for consumer in range(1, n_consumers + 1):
        consumer_bias = rng.normal(0, 0.8)
        for product in products:
            profile = product_profiles[product]
            liking = np.clip(liking_base[product] + consumer_bias + rng.normal(0, 1.0), 1, 9)
            row = {
                "Consumer": consumer,
                "Product": product,
                "Liking": int(round(liking)),
            }
            for attribute in attributes:
                row[attribute] = int(rng.random() < profile[attribute])
            rows.append(row)
    return pd.DataFrame(rows)


def make_jar(rng: np.random.Generator) -> pd.DataFrame:
    """Consumer x Product JAR data with liking penalties away from JAR=3."""
    n_consumers = 60
    products = ["Prod1", "Prod2", "Prod3", "Prod4"]
    jar_attributes = ["Sweetness_JAR", "Saltiness_JAR", "Texture_JAR"]
    # Each product's typical JAR center (3 = just about right) per attribute.
    product_jar_center = {
        "Prod1": {"Sweetness_JAR": 3.0, "Saltiness_JAR": 3.0, "Texture_JAR": 3.0},
        "Prod2": {"Sweetness_JAR": 4.0, "Saltiness_JAR": 2.5, "Texture_JAR": 3.5},
        "Prod3": {"Sweetness_JAR": 2.0, "Saltiness_JAR": 3.5, "Texture_JAR": 2.5},
        "Prod4": {"Sweetness_JAR": 3.5, "Saltiness_JAR": 3.0, "Texture_JAR": 4.0},
    }
    base_liking = {"Prod1": 7.0, "Prod2": 5.5, "Prod3": 5.8, "Prod4": 6.3}

    rows = []
    for consumer in range(1, n_consumers + 1):
        consumer_bias = rng.normal(0, 0.6)
        for product in products:
            row = {"Consumer": consumer, "Product": product}
            penalty = 0.0
            for attribute in jar_attributes:
                center = product_jar_center[product][attribute]
                jar_value = int(np.clip(round(rng.normal(center, 0.9)), 1, 5))
                row[attribute] = jar_value
                penalty += 0.9 * (jar_value - 3) ** 2
            liking = base_liking[product] + consumer_bias - penalty + rng.normal(0, 0.5)
            row["Overall_Liking"] = int(np.clip(round(liking), 1, 9))
            rows.append(row)
    return pd.DataFrame(rows)


def make_hedonic_consumer(rng: np.random.Generator) -> pd.DataFrame:
    """Consumer x Product liking data with 3 latent preference segments.

    Serves Hedonic Analysis (repeated mode), Consumer Segmentation, and the
    Internal Preference Map.
    """
    n_consumers = 30
    products = [f"Prod{i}" for i in range(1, 7)]
    segments = rng.choice([0, 1, 2], size=n_consumers, p=[0.4, 0.35, 0.25])
    # Segment-specific product preference (mean liking per product).
    segment_profile = {
        0: [7.5, 7.0, 6.0, 4.0, 3.5, 4.5],  # prefers Prod1-Prod3
        1: [3.5, 4.0, 4.5, 7.5, 7.8, 6.5],  # prefers Prod4-Prod6
        2: [5.5, 5.8, 5.5, 5.8, 5.5, 5.6],  # neutral across all products
    }
    # A universal quality shift layered on top of the segment preferences, so
    # the aggregate Hedonic/Friedman comparison is also significant, not just
    # the segment-level clustering.
    universal_bias = [1.0, 0.3, -0.5, 0.5, -1.0, -0.2]

    rows = []
    for consumer_idx in range(1, n_consumers + 1):
        segment = segments[consumer_idx - 1]
        profile = segment_profile[segment]
        for product_idx, product in enumerate(products):
            liking = np.clip(
                profile[product_idx] + universal_bias[product_idx] + rng.normal(0, 0.9),
                1,
                9,
            )
            rows.append(
                {
                    "Consumer": f"C{consumer_idx:03d}",
                    "Product": product,
                    "Liking": int(round(liking)),
                }
            )
    return pd.DataFrame(rows)


def make_external_preference(rng: np.random.Generator) -> pd.DataFrame:
    """Product-level liking + sensory descriptors for External Preference Map."""
    products = [f"Product_{i}" for i in range(1, 9)]
    sweetness = rng.uniform(1, 9, len(products))
    bitterness = rng.uniform(1, 9, len(products))
    crunchiness = rng.uniform(1, 9, len(products))
    aroma_intensity = rng.uniform(1, 9, len(products))

    # Liking increases with sweetness & aroma, decreases with bitterness.
    liking = (
        4.5
        + 0.35 * sweetness
        - 0.30 * bitterness
        + 0.15 * crunchiness
        + 0.20 * aroma_intensity
        + rng.normal(0, 0.4, len(products))
    )

    return pd.DataFrame(
        {
            "Product": products,
            "Liking": np.clip(liking, 1, 9).round(2),
            "Sweetness": sweetness.round(2),
            "Bitterness": bitterness.round(2),
            "Crunchiness": crunchiness.round(2),
            "Aroma_Intensity": aroma_intensity.round(2),
        }
    )


def make_pls(rng: np.random.Generator) -> pd.DataFrame:
    """Process parameters (X) predicting sensory responses (Y) for PLS Regression."""
    n = 40
    sweetener_level = rng.uniform(5, 20, n)       # grams per 100g
    fat_content = rng.uniform(0, 15, n)           # percent
    mixing_time = rng.uniform(5, 30, n)           # minutes
    process_temp = rng.uniform(60, 100, n)        # Celsius

    sweetness_perception = (
        2.0 + 0.35 * sweetener_level - 0.02 * process_temp + rng.normal(0, 0.6, n)
    )
    creaminess = 2.5 + 0.45 * fat_content + 0.03 * mixing_time + rng.normal(0, 0.6, n)
    overall_liking = (
        3.0
        + 0.18 * sweetener_level
        + 0.22 * fat_content
        - 0.015 * (process_temp - 80) ** 2 / 10
        + rng.normal(0, 0.5, n)
    )

    return pd.DataFrame(
        {
            "SweetenerLevel": sweetener_level.round(2),
            "FatContent": fat_content.round(2),
            "MixingTime": mixing_time.round(2),
            "ProcessTemp": process_temp.round(2),
            "SweetnessPerception": np.clip(sweetness_perception, 0, 10).round(2),
            "Creaminess": np.clip(creaminess, 0, 10).round(2),
            "OverallLiking": np.clip(overall_liking, 0, 10).round(2),
        }
    )


# ---------------------------------------------------------------------------


def main() -> None:
    OUT_GENERAL.mkdir(parents=True, exist_ok=True)
    OUT_SENSORY.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)

    make_student_performance(rng).to_csv(
        OUT_GENERAL / "student_performance_demo.csv", index=False
    )
    wide, long = make_repeated_measures(rng)
    wide.to_csv(OUT_GENERAL / "repeated_measures_wide_demo.csv", index=False)
    long.to_csv(OUT_GENERAL / "repeated_measures_long_demo.csv", index=False)

    make_discrimination(rng).to_csv(
        OUT_SENSORY / "sensory_discrimination_demo.csv", index=False
    )
    make_ranking(rng).to_csv(OUT_SENSORY / "sensory_ranking_demo.csv", index=False)
    make_panel_profile(rng).to_csv(
        OUT_SENSORY / "sensory_panel_profile_demo.csv", index=False
    )
    make_product_profile_wide(rng).to_csv(
        OUT_SENSORY / "sensory_product_profile_wide_demo.csv", index=False
    )
    make_temporal(rng).to_csv(OUT_SENSORY / "sensory_temporal_demo.csv", index=False)
    make_cata(rng).to_csv(OUT_SENSORY / "sensory_cata_demo.csv", index=False)
    make_jar(rng).to_csv(OUT_SENSORY / "sensory_jar_demo.csv", index=False)
    make_hedonic_consumer(rng).to_csv(
        OUT_SENSORY / "sensory_hedonic_consumer_demo.csv", index=False
    )
    make_external_preference(rng).to_csv(
        OUT_SENSORY / "sensory_external_preference_demo.csv", index=False
    )
    make_pls(rng).to_csv(OUT_SENSORY / "sensory_pls_demo.csv", index=False)

    print("Synthetic datasets written to:")
    for path in sorted((OUT_GENERAL, OUT_SENSORY)):
        for csv_file in sorted(path.glob("*.csv")):
            print(f"  {csv_file.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
