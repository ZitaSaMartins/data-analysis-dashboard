# Synthetic Test Data

Ready-to-upload CSV datasets for exercising every analysis page in the
dashboard. Each file has a real, intentional effect baked in (a group
difference, a correlation, a product preference, a penalty pattern, etc.), so
running the matching analysis produces a meaningful — usually statistically
significant — result instead of pure noise.

Regenerate all files (deterministic, seeded) with:

```
python scripts/generate_sample_data.py
```

## general_stats/

### `student_performance_demo.csv` (500 rows)
Student exam-performance data covering **Explore Data** (distributions,
correlations, comparisons, hierarchical, temporal) and most of
**Analyze Relationships**, plus **Advanced Analysis**.

| Page | Columns to pick |
|---|---|
| Distributions / Test Distribution / Test Variance | any numeric column (`ExamScore`, `StudyHours`, `PriorGPA`, `AttendanceRate`, `SatisfactionScore`) |
| Part-to-Whole & Hierarchical | hierarchy levels: `Region`, `Department` |
| Temporal | X = `Date`, Y = `ExamScore` |
| Independent t-test / Mann-Whitney | outcome `ExamScore`, group `StudyGroup` (In-Person vs Online) |
| One-Way ANOVA / Kruskal-Wallis / Mood's Median | outcome `ExamScore`, group `TeachingMethod` |
| Two-Way ANOVA | outcome `ExamScore`, factors `StudyGroup` × `TeachingMethod` |
| Pearson / Spearman / Kendall / Linear Regression | `StudyHours` or `PriorGPA` vs `ExamScore` |
| Chi-Square | `TeachingMethod` vs `PassFail` |
| Build Predictions (ML) | target `ExamScore` (regression) or `PassFail` (classification); any other columns as predictors |

### `repeated_measures_wide_demo.csv` (50 rows) and `repeated_measures_long_demo.csv` (200 rows)
Same simulated 12-week stress-reduction program (`Baseline`, `Week4`, `Week8`,
`Week12`), in wide and long form.

| Page | File | Columns |
|---|---|---|
| Paired t-test / Wilcoxon Signed-Rank | wide | any two of `Baseline`/`Week4`/`Week8`/`Week12` |
| Friedman | wide | all four time columns |
| Repeated Measures ANOVA | long | outcome `StressScore`, subject `ParticipantID`, condition `TimePoint` |

## sensory_analysis/

### `sensory_discrimination_demo.csv` (60 rows)
One `Response` column (`Correct`/`Incorrect`) above both the 1/3 and 1/2
chance baselines. Serves **Triangle**, **Paired Comparison**, **Duo-Trio**,
and **Tetrad** tests — pick `Response` and mark `Correct` as the correct value.

### `sensory_ranking_demo.csv` (60 rows)
`Assessor`, `Product`, `Rank` — 12 assessors ranking 5 products with a
consistent true preference order. Serves the **Ranking Test**.

### `sensory_panel_profile_demo.csv` (60 rows)
`Assessor`, `Product`, `Replicate`, plus `Sweetness`, `Bitterness`,
`Aroma_Intensity`, `Crunchiness`, `Overall_Liking`. One rich dataset serves:
**QDA / Sensory Profile**, **Product Characterization**, **Panel
Discrimination**, **Panel Repeatability**, **Panel Agreement** (pick one
attribute as Rating), **Panel Scale Use**, **GPA**, and **STATIS**.

### `sensory_product_profile_wide_demo.csv` (10 rows)
Product-level wide matrix with block-prefixed attributes (`taste_*`,
`aroma_*`, `texture_*`) forming two clearly separated product clusters.
Serves **PCA**, **Hierarchical Clustering**, and **MFA** (block delimiter `_`).

### `sensory_temporal_demo.csv` (132 rows)
`Product`, `Time`, `Replicate`, `Intensity` — four products with distinct
time-intensity curves (fast burst vs slow build-up). Serves **Temporal
Methods**.

### `sensory_cata_demo.csv` (360 rows)
`Consumer`, `Product`, `Liking`, plus 10 binary 0/1 CATA attributes, matching
the layout of `demo_CATA_EN.xlsm`. Serves **CATA Analysis**.

### `sensory_jar_demo.csv` (240 rows)
`Consumer`, `Product`, `Overall_Liking`, and three JAR attributes
(`Sweetness_JAR`, `Saltiness_JAR`, `Texture_JAR`, scale 1–5, midpoint 3) with
a real liking penalty away from "just about right". Serves **JAR & Penalty
Analysis**.

### `sensory_hedonic_consumer_demo.csv` (180 rows)
`Consumer`, `Product`, `Liking` — 30 consumers × 6 products with 3 latent
preference segments plus an overall product-quality effect. Serves **Hedonic
Analysis** (repeated mode), **Consumer Segmentation**, and **Internal
Preference Map**.

### `sensory_external_preference_demo.csv` (8 rows)
Product-level `Product`, `Liking`, and 4 sensory descriptors. Serves
**External Preference Map**.

### `sensory_pls_demo.csv` (40 rows)
Process parameters (`SweetenerLevel`, `FatContent`, `MixingTime`,
`ProcessTemp`) and sensory responses (`SweetnessPerception`, `Creaminess`,
`OverallLiking`) with real linear relationships. Serves **PLS Regression**.
