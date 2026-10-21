import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

def validate_dataset(df: pd.DataFrame) -> Dict:
    """
    Comprehensive dataset validation.
    Returns dict with validation status and issues found.
    """
    validation_issues = []
    
    # Check 1: Empty dataset
    if df.empty:
        validation_issues.append("❌ Dataset is empty")
        return {'is_valid': False, 'issues': validation_issues}
    
    # Check 2: Column names
    if df.columns.duplicated().any():
        validation_issues.append(f"❌ Duplicate column names found: {df.columns[df.columns.duplicated()].tolist()}")
    
    # Check 3: Invalid characters in column names
    invalid_chars = [col for col in df.columns if any(char in col for char in ['!', '@', '#', '$', '%', '^', '&', '*', '(', ')'])]
    if invalid_chars:
        validation_issues.append(f"⚠️ Column names with special characters: {invalid_chars}")
    
    # Check 4: Spaces at start/end of column names
    spaces_in_names = [col for col in df.columns if col != col.strip()]
    if spaces_in_names:
        validation_issues.append(f"⚠️ Column names with leading/trailing spaces: {spaces_in_names}")
    
    # Check 5: Data type detection
    type_issues = _check_column_types(df)
    validation_issues.extend(type_issues)
    
    # Check 6: Missing values threshold
    missing_pct = (df.isnull().sum() / len(df) * 100).max()
    if missing_pct > 50:
        validation_issues.append(f"⚠️ Some columns have >50% missing values")
    
    # Check 7: Single value columns
    single_value = [col for col in df.columns if df[col].nunique() == 1]
    if single_value:
        validation_issues.append(f"⚠️ Columns with only one unique value: {single_value}")
    
    is_valid = len(validation_issues) == 0
    
    return {
        'is_valid': is_valid,
        'issues': validation_issues
    }


def _check_column_types(df: pd.DataFrame) -> List[str]:
    """Check for data type issues."""
    issues = []
    
    for col in df.columns:
        data = df[col].dropna()
        
        if len(data) == 0:
            issues.append(f"⚠️ Column '{col}' has all missing values")
            continue
        
        # Try to infer if numeric column has text values
        if df[col].dtype == 'object':
            try:
                numeric_check = pd.to_numeric(data, errors='coerce').notna().sum()
                if numeric_check > 0 and numeric_check < len(data):
                    issues.append(f"⚠️ Column '{col}' has mixed numeric and text values")
            except:
                pass
    
    return issues


def generate_validation_report(df: pd.DataFrame) -> str:
    """Generate a human-readable validation report."""
    report = f"""
    **Dataset Overview:**
    - Total Rows: {df.shape[0]:,}
    - Total Columns: {df.shape[1]}
    - Total Cells: {df.shape[0] * df.shape[1]:,}
    - Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
    
    **Data Completeness:**
    - Total Missing Values: {df.isnull().sum().sum()}
    - Percentage Complete: {(1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100:.1f}%
    - Columns with Missing Data: {(df.isnull().sum() > 0).sum()}
    
    **Data Types Distribution:**
    """
    
    type_counts = df.dtypes.value_counts()
    for dtype, count in type_counts.items():
        report += f"\n    - {dtype}: {count} columns"
    
    return report


def analyze_variable_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Automatically classify variables as continuous, categorical, or binary.
    Helps students understand which variables are dependent vs independent.
    """
    continuous = []
    categorical = []
    binary = []
    
    for col in df.columns:
        data = df[col].dropna()
        
        # Binary: exactly 2 unique values
        if df[col].nunique() == 2:
            binary.append(col)
        
        # Continuous: numeric with many unique values
        elif pd.api.types.is_numeric_dtype(df[col]):
            if df[col].nunique() > 20:
                continuous.append(col)
            else:
                categorical.append(col)
        
        # Categorical: object type or few unique values
        else:
            categorical.append(col)
    
    return {
        'continuous': continuous,
        'categorical': categorical,
        'binary': binary
    }


def get_statistical_summary(df: pd.DataFrame, col: str) -> Dict:
    """Get detailed statistical summary for a column."""
    data = df[col].dropna()
    
    if pd.api.types.is_numeric_dtype(df[col]):
        return {
            'type': 'numeric',
            'count': len(data),
            'mean': data.mean(),
            'median': data.median(),
            'std': data.std(),
            'min': data.min(),
            'max': data.max(),
            'q25': data.quantile(0.25),
            'q75': data.quantile(0.75)
        }
    else:
        return {
            'type': 'categorical',
            'count': len(data),
            'unique': data.nunique(),
            'most_common': data.value_counts().index[0] if len(data) > 0 else None,
            'most_common_freq': data.value_counts().values[0] if len(data) > 0 else None
        }