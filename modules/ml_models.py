import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                            mean_squared_error, r2_score, confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns

def run_ml_pipeline(df: pd.DataFrame, dependent_vars: list, independent_vars: list):
    """Run machine learning pipeline."""
    
    if not dependent_vars or not independent_vars:
        st.error("Please specify dependent and independent variables.")
        return
    
    if len(dependent_vars) > 1:
        st.warning("Using first dependent variable for single-output prediction.")
        target_var = dependent_vars[0]
    else:
        target_var = dependent_vars[0]
    
    st.markdown(f"**Target Variable:** {target_var}")
    st.markdown(f"**Features:** {', '.join(independent_vars)}")
    
    # Prepare data
    X = df[independent_vars].copy()
    y = df[target_var].copy()
    
    # Handle missing values
    valid_idx = (~X.isnull().any(axis=1)) & (~y.isnull())
    X = X[valid_idx]
    y = y[valid_idx]
    
    if len(X) == 0:
        st.error("No valid data after removing missing values.")
        return
    
    st.info(f"Using {len(X)} observations after cleaning.")
    
    # Determine task type
    if pd.api.types.is_numeric_dtype(y):
        if y.nunique() <= 10:
            task_type = "Classification"
            st.markdown("**Task Type:** Classification (predicting categories)")
        else:
            task_type = "Regression"
            st.markdown("**Task Type:** Regression (predicting continuous values)")
    else:
        task_type = "Classification"
        st.markdown("**Task Type:** Classification (predicting categories)")
    
    # Convert categorical features to numeric
    X_encoded = X.copy()
    for col in X_encoded.columns:
        if not pd.api.types.is_numeric_dtype(X_encoded[col]):
            X_encoded[col] = pd.factorize(X_encoded[col])[0]
    
    # Train/test split
    col1, col2 = st.columns(2)
    
    with col1:
        test_size = st.slider("Test set size (%):", 10, 50, 20) / 100
    
    with col2:
        random_state = st.number_input("Random state (for reproducibility):", 0, 1000, 42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=test_size, random_state=random_state
    )
    
    # Model selection
    if task_type == "Classification":
        model_options = ["Logistic Regression", "Random Forest"]
    else:
        model_options = ["Linear Regression", "Random Forest"]
    
    selected_model = st.selectbox("Select model:", model_options)
    
    if st.button("Train Model", type="primary"):
        with st.spinner("Training model..."):
            # Fit model
            if selected_model == "Logistic Regression":
                model = LogisticRegression(max_iter=1000)
            elif selected_model == "Linear Regression":
                model = LinearRegression()
            elif selected_model == "Random Forest" and task_type == "Classification":
                model = RandomForestClassifier(n_estimators=100, random_state=random_state)
            elif selected_model == "Random Forest" and task_type == "Regression":
                model = RandomForestRegressor(n_estimators=100, random_state=random_state)
            
            # Train
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Evaluate
            st.markdown("### Model Performance")
            
            if task_type == "Classification":
                evaluate_classification(y_train, y_pred_train, y_test, y_pred_test, model)
            else:
                evaluate_regression(y_train, y_pred_train, y_test, y_pred_test, model)
            
            # Feature importance (for tree-based models)
            if hasattr(model, 'feature_importances_'):
                show_feature_importance(model, independent_vars)


def evaluate_classification(y_train, y_pred_train, y_test, y_pred_test, model):
    """Evaluate classification model."""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Training Set:**")
        st.metric("Accuracy", f"{accuracy_score(y_train, y_pred_train):.3f}")
    
    with col2:
        st.markdown("**Test Set:**")
        st.metric("Accuracy", f"{accuracy_score(y_test, y_pred_test):.3f}")
    
    # Detailed metrics
    st.markdown("#### Classification Metrics (Test Set)")
    
    st.markdown(f"**Precision:** {precision_score(y_test, y_pred_test, average='weighted', zero_division=0):.3f}")
    st.markdown(f"**Recall:** {recall_score(y_test, y_pred_test, average='weighted', zero_division=0):.3f}")
    st.markdown(f"**F1-Score:** {f1_score(y_test, y_pred_test, average='weighted', zero_division=0):.3f}")
    
    # Confusion matrix
    st.markdown("#### Confusion Matrix")
    
    cm = confusion_matrix(y_test, y_pred_test)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar_kws={'label': 'Count'})
    ax.set_xlabel('Predicted', fontweight='bold')
    ax.set_ylabel('Actual', fontweight='bold')
    ax.set_title('Confusion Matrix', fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)


def evaluate_regression(y_train, y_pred_train, y_test, y_pred_test, model):
    """Evaluate regression model."""
    
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Train R²", f"{train_r2:.3f}")
    with col2:
        st.metric("Test R²", f"{test_r2:.3f}")
    with col3:
        st.metric("Train RMSE", f"{train_rmse:.3f}")
    with col4:
        st.metric("Test RMSE", f"{test_rmse:.3f}")
    
    # Actual vs Predicted
    st.markdown("#### Actual vs Predicted Values")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter plot
    ax1.scatter(y_test, y_pred_test, alpha=0.6, edgecolors='k')
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax1.set_xlabel('Actual', fontweight='bold')
    ax1.set_ylabel('Predicted', fontweight='bold')
    ax1.set_title('Actual vs Predicted', fontweight='bold')
    ax1.grid(alpha=0.3)
    
    # Residuals
    residuals = y_test - y_pred_test
    ax2.scatter(y_pred_test, residuals, alpha=0.6, edgecolors='k')
    ax2.axhline(y=0, color='r', linestyle='--', lw=2)
    ax2.set_xlabel('Predicted', fontweight='bold')
    ax2.set_ylabel('Residuals', fontweight='bold')
    ax2.set_title('Residual Plot', fontweight='bold')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)


def show_feature_importance(model, feature_names: list):
    """Display feature importance for tree-based models."""
    
    st.markdown("#### Feature Importance")
    
    importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Table
    st.dataframe(importance, use_container_width=True)
    
    # Bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.barh(importance['Feature'], importance['Importance'], color='steelblue', edgecolor='black')
    ax.set_xlabel('Importance Score', fontweight='bold')
    ax.set_title('Feature Importance', fontweight='bold')
    ax.grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    st.pyplot(fig)