"""
Utility functions for statistical analysis shared across modules.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Union, Optional


def safe_mean(values: List[Union[int, float]]) -> float:
    """Calculate mean with safe handling of empty lists.
    
    Args:
        values: List of numeric values
        
    Returns:
        float: Mean of values or 0 if list is empty
    """
    return np.mean(values) if values else 0


def safe_median(values: List[Union[int, float]]) -> float:
    """Calculate median with safe handling of empty lists.
    
    Args:
        values: List of numeric values
        
    Returns:
        float: Median of values or 0 if list is empty
    """
    return np.median(values) if values else 0


def safe_max(values: List[Union[int, float]]) -> float:
    """Calculate max with safe handling of empty lists.
    
    Args:
        values: List of numeric values
        
    Returns:
        float: Maximum of values or 0 if list is empty
    """
    return max(values) if values else 0


def safe_corr(series1: pd.Series, series2: pd.Series) -> float:
    """Calculate correlation with safe handling of errors.
    
    Args:
        series1: First data series
        series2: Second data series
        
    Returns:
        float: Correlation coefficient or 0 if calculation fails
    """
    try:
        correlation = series1.corr(series2)
        return correlation if pd.notna(correlation) else 0
    except Exception:
        return 0


def ensure_numeric(value: Any, default: float = 0.0) -> float:
    """Ensure a value is numeric, with fallback to default.
    
    Args:
        value: Value to convert to numeric
        default: Default value if conversion fails
        
    Returns:
        float: Numeric value or default if conversion fails
    """
    if value is None:
        return default
    
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def convert_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Safely convert DataFrame to dict records with handling for empty DataFrames.
    
    Args:
        df: DataFrame to convert
        
    Returns:
        List of records or empty list for empty DataFrame
    """
    return df.to_dict('records') if not df.empty else []


def filter_valid_rows(df: pd.DataFrame, required_columns: List[str], 
                      min_valid_columns: Optional[int] = None) -> pd.DataFrame:
    """Filter rows that have valid data in required columns.
    
    Args:
        df: DataFrame to filter
        required_columns: List of column names that must be valid
        min_valid_columns: Minimum number of valid columns required (default: all)
        
    Returns:
        DataFrame with only valid rows
    """
    if df.empty:
        return df
    
    # If min_valid_columns not specified, require all columns
    if min_valid_columns is None:
        min_valid_columns = len(required_columns)
        
    # Filter based on count of non-null values in required columns
    available_columns = [col for col in required_columns if col in df.columns]
    valid_mask = df[available_columns].notna().sum(axis=1) >= min_valid_columns
    
    return df[valid_mask]