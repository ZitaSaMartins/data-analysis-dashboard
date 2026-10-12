import pandas as pd
import numpy as np

def recommend_models(analysis_type: str, df: pd.DataFrame, characteristics: list) -> list:
    """
    Recommend models based on analysis type and data characteristics.
    Uses an expert system approach (no API calls, completely free).
    """
    
    # Analyze dataset
    n_samples = len(df)
    n_features = len(df.columns)
    numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
    
    recommendations = []
    
    if analysis_type == "EDA":
        recommendations = recommend_eda(df, characteristics)
    
    elif analysis_type == "Hypothesis Testing":
        recommendations = recommend_hypothesis_tests(df, characteristics)
    
    elif analysis_type == "Classification":
        recommendations = recommend_classification(df, n_samples, n_features, characteristics)
    
    elif analysis_type == "Regression":
        recommendations = recommend_regression(df, n_samples, n_features, characteristics)
    
    elif analysis_type == "Sensory Analysis":
        recommendations = recommend_sensory(df, characteristics)
    
    return recommendations


def recommend_eda(df: pd.DataFrame, characteristics: list) -> list:
    """Recommend EDA approaches."""
    
    return [
        {
            'model': 'Descriptive Statistics',
            'suitability': 10,
            'reason': 'Always start with summary statistics, means, and medians to understand your data.',
            'pros': 'Simple, interpretable, quick to compute.',
            'cons': 'May miss complex patterns.',
            'when_to_use': 'Always - foundational step in any analysis.'
        },
        {
            'model': 'Data Distributions (Histograms & Density Plots)',
            'suitability': 9,
            'reason': 'Visualize how variables are distributed to identify skewness, outliers, and modality.',
            'pros': 'Quick visual assessment, identify potential data issues.',
            'cons': 'Less useful with very high-dimensional data.',
            'when_to_use': 'For understanding individual variable shapes before modeling.'
        },
        {
            'model': 'Correlation Analysis',
            'suitability': 9,
            'reason': 'Identify relationships between numeric variables.',
            'pros': 'Reveals multicollinearity, informs feature selection.',
            'cons': 'Only captures linear relationships.',
            'when_to_use': 'When exploring relationships before predictive modeling.'
        },
        {
            'model': 'Scatter Plots & Joint Plots',
            'suitability': 8,
            'reason': 'Visualize bivariate relationships to identify patterns and outliers.',
            'pros': 'Can reveal non-linear patterns.',
            'cons': 'Becomes cluttered with many features.',
            'when_to_use': 'When exploring specific variable pairs.'
        },
        {
            'model': 'Box Plots by Groups',
            'suitability': 8,
            'reason': 'Compare distributions across categorical groups.',
            'pros': 'Easy to spot differences between groups.',
            'cons': 'Can be hard to interpret with many groups.',
            'when_to_use': 'When comparing groups or looking for group effects.'
        }
    ]


def recommend_hypothesis_tests(df: pd.DataFrame, characteristics: list) -> list:
    """Recommend hypothesis tests."""
    
    return [
        {
            'model': 't-test (parametric)',
            'suitability': 9,
            'reason': 'Compare means of two independent groups when data is normally distributed.',
            'pros': 'Powerful, widely used, assumes normal distribution.',
            'cons': 'Sensitive to outliers, requires normality assumption.',
            'when_to_use': 'Comparing 2 groups with ~normal data. Check normality first!'
        },
        {
            'model': 'ANOVA (parametric)',
            'suitability': 9,
            'reason': 'Compare means across 3+ groups when data is normally distributed.',
            'pros': 'Powerful for multiple groups.',
            'cons': 'Assumes normality and homogeneity of variance.',
            'when_to_use': 'Comparing 3+ groups; follow up with post-hoc tests.'
        },
        {
            'model': 'Mann-Whitney U (non-parametric)',
            'suitability': 8,
            'reason': 'Non-parametric alternative to t-test when normality assumption fails.',
            'pros': 'Robust to non-normality and outliers.',
            'cons': 'Less powerful than t-test if normality holds.',
            'when_to_use': 'Data is non-normal, skewed, or has outliers.'
        },
        {
            'model': 'Kruskal-Wallis (non-parametric)',
            'suitability': 8,
            'reason': 'Non-parametric alternative to ANOVA for 3+ groups.',
            'pros': 'No normality assumption required.',
            'cons': 'Less powerful than ANOVA if data is truly normal.',
            'when_to_use': 'Multiple groups with non-normal or ordinal data.'
        },
        {
            'model': 'Chi-Square Test',
            'suitability': 9,
            'reason': 'Test association between two categorical variables.',
            'pros': 'Designed for categorical data.',
            'cons': 'Requires sufficient sample size and cell counts.',
            'when_to_use': 'Testing independence of categorical variables.'
        },
        {
            'model': 'Correlation & Regression',
            'suitability': 8,
            'reason': 'Test linear relationship between numeric variables.',
            'pros': 'Tests both magnitude and direction of relationship.',
            'cons': 'May not capture non-linear relationships.',
            'when_to_use': 'Examining linear associations between continuous variables.'
        }
    ]


def recommend_classification(df: pd.DataFrame, n_samples: int, n_features: int, characteristics: list) -> list:
    """Recommend classification models."""
    
    recommendations = []
    
    # Logistic Regression
    recommendations.append({
        'model': 'Logistic Regression',
        'suitability': 9 if n_samples > 50 else 8,
        'reason': 'Simple, interpretable linear classifier. Great for learning and baseline.',
        'pros': 'Fast, interpretable coefficients, works well with linear boundaries.',
        'cons': 'Assumes linear boundary, poor with complex patterns.',
        'when_to_use': 'Interpretability matters, baseline model, small samples.'
    })
    
    # Random Forest
    suitability = 9
    if n_samples < 50:
        suitability = 6
    if "Imbalanced classes" in characteristics:
        suitability -= 1
    
    recommendations.append({
        'model': 'Random Forest',
        'suitability': suitability,
        'reason': 'Powerful ensemble method, handles non-linear patterns, robust to outliers.',
        'pros': 'Non-linear, feature selection built-in, handles missing data well.',
        'cons': 'Slower to train, less interpretable, can overfit on small samples.',
        'when_to_use': 'Many features, complex patterns, moderate-to-large datasets.'
    })
    
    # SVM recommendation
    recommendations.append({
        'model': 'Support Vector Machine (SVM)',
        'suitability': 7 if n_samples > 100 else 5,
        'reason': 'Powerful for high-dimensional data and complex boundaries.',
        'pros': 'Works well in high dimensions, memory efficient.',
        'cons': 'Sensitive to feature scaling, slower on large datasets.',
        'when_to_use': 'High-dimensional data (feature scaling required).'
    })
    
    # Naive Bayes
    recommendations.append({
        'model': 'Naive Bayes',
        'suitability': 8,
        'reason': 'Fast probabilistic classifier, good for text and categorical features.',
        'pros': 'Very fast, works with small samples, provides probabilities.',
        'cons': 'Assumes feature independence (often not true).',
        'when_to_use': 'Fast predictions needed, text classification, categorical features.'
    })
    
    return sorted(recommendations, key=lambda x: x['suitability'], reverse=True)


def recommend_regression(df: pd.DataFrame, n_samples: int, n_features: int, characteristics: list) -> list:
    """Recommend regression models."""
    
    recommendations = []
    
    # Linear Regression
    recommendations.append({
        'model': 'Linear Regression',
        'suitability': 9 if n_features < 10 else 7,
        'reason': 'Simple, interpretable baseline for continuous prediction.',
        'pros': 'Fast, interpretable, works well for linear relationships.',
        'cons': 'Assumes linearity, sensitive to outliers.',
        'when_to_use': 'Interpretability required, linear relationship expected.'
    })
    
    # Ridge/Lasso
    recommendations.append({
        'model': 'Ridge/Lasso Regression',
        'suitability': 8 if n_features > 10 else 6,
        'reason': 'Handles multicollinearity and feature selection through regularization.',
        'pros': 'Reduces overfitting, handles correlated features.',
        'cons': 'Requires parameter tuning, adds complexity.',
        'when_to_use': 'Many correlated features, need to prevent overfitting.'
    })
    
    # Random Forest Regression
    suitability = 9
    if n_samples < 50:
        suitability = 6
    if "Non-linear relationships" in characteristics:
        suitability = 10
    
    recommendations.append({
        'model': 'Random Forest Regression',
        'suitability': suitability,
        'reason': 'Powerful non-linear model, handles feature interactions automatically.',
        'pros': 'Captures non-linearity, robust to outliers, feature importance.',
        'cons': 'Less interpretable, slower, can overfit small datasets.',
        'when_to_use': 'Non-linear patterns expected, moderate-to-large sample size.'
    })
    
    # Gradient Boosting
    recommendations.append({
        'model': 'Gradient Boosting (XGBoost)',
        'suitability': 8 if n_samples > 100 else 5,
        'reason': 'State-of-the-art ensemble, excellent for competition and accuracy.',
        'pros': 'Often best accuracy, handles missing data, captures interactions.',
        'cons': 'Slower to train, more tuning required, less interpretable.',
        'when_to_use': 'Maximum accuracy needed, larger datasets, you have time to tune.'
    })
    
    # Polynomial Regression
    recommendations.append({
        'model': 'Polynomial Regression',
        'suitability': 6 if "Non-linear relationships" in characteristics else 4,
        'reason': 'Captures non-linear relationships using polynomial features.',
        'pros': 'Relatively simple, good for known polynomial relationships.',
        'cons': 'Can overfit, requires careful degree selection.',
        'when_to_use': 'Clear non-linear pattern (U-shape, quadratic) visible in data.'
    })
    
    return sorted(recommendations, key=lambda x: x['suitability'], reverse=True)


def recommend_sensory(df: pd.DataFrame, characteristics: list) -> list:
    """Recommend sensory analysis methods."""
    
    return [
        {
            'model': 'Descriptive Sensory Analysis (Sensory Profile)',
            'suitability': 10,
            'reason': 'Gold standard for trained sensory panels - characterizes product properties.',
            'pros': 'Comprehensive, reproducible, identifies key attributes.',
            'cons': 'Requires trained panel, time-consuming.',
            'when_to_use': 'For detailed product characterization with trained evaluators.'
        },
        {
            'model': 'Hedonic/Affective Testing',
            'suitability': 10,
            'reason': 'Consumer liking scores - directly measures acceptability.',
            'pros': 'Consumer insights, predicts market success.',
            'cons': 'Large sample needed, emotional context affects ratings.',
            'when_to_use': 'Understanding consumer preferences and market potential.'
        },
        {
            'model': 'Paired Comparison',
            'suitability': 9,
            'reason': 'Simple, intuitive - consumers choose preferred product.',
            'pros': 'Easy for consumers, less scale confusion.',
            'cons': 'Only binary outcome.',
            'when_to_use': 'Comparing 2-3 products quickly.'
        },
        {
            'model': 'Ranking Test',
            'suitability': 8,
            'reason': 'Rank 3-5 samples from best to worst.',
            'pros': 'Simple non-parametric test.',
            'cons': 'Can only rank few samples.',
            'when_to_use': 'Simple preference study with 3-5 samples.'
        },
        {
            'model': 'Triangle Test',
            'suitability': 9,
            'reason': 'Discrimination test - can you find the odd sample?',
            'pros': 'Sensitivity to small differences.',
            'cons': 'Statistical power requires many panelists.',
            'when_to_use': 'Testing if differences are perceptible.'
        },
        {
            'model': 'Principal Component Analysis (PCA)',
            'suitability': 8,
            'reason': 'Reduce sensory data dimensionality while preserving patterns.',
            'pros': 'Visualizes product relationships, identifies main drivers.',
            'cons': 'Less interpretable than raw attributes.',
            'when_to_use': 'Many attributes, want to see product clusters.'
        }
    ]