"""
Data validation utilities for the price optimization project.

This module provides functions to validate data quality and structure
across different stages of the pipeline.
"""
import pandas as pd
from typing import List, Dict, Any, Optional


def validate_dataframe_structure(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validate that a DataFrame has the required columns.
    
    Args:
        df: The DataFrame to validate
        required_columns: List of column names that must be present
        
    Returns:
        True if all required columns are present, False otherwise
        
    Example:
        >>> df = pd.DataFrame({'price': [1, 2], 'sales': [10, 20]})
        >>> validate_dataframe_structure(df, ['price', 'sales'])
        True
    """
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        print(f"Missing required columns: {missing_columns}")
        return False
    return True


def validate_data_types(df: pd.DataFrame, expected_types: Dict[str, str]) -> bool:
    """
    Validate that DataFrame columns have the expected data types.
    
    Args:
        df: The DataFrame to validate
        expected_types: Dictionary mapping column names to expected dtypes
        
    Returns:
        True if all columns have expected types, False otherwise
        
    Example:
        >>> df = pd.DataFrame({'price': [1.0, 2.0], 'sales': [10, 20]})
        >>> validate_data_types(df, {'price': 'float64', 'sales': 'int64'})
        True
    """
    for column, expected_type in expected_types.items():
        if column in df.columns:
            actual_type = str(df[column].dtype)
            if actual_type != expected_type:
                print(f"Column '{column}' has type '{actual_type}', expected '{expected_type}'")
                return False
    return True


def validate_numeric_range(df: pd.DataFrame, column: str, min_val: float = None, max_val: float = None) -> bool:
    """
    Validate that a numeric column is within the specified range.
    
    Args:
        df: The DataFrame to validate
        column: The column name to validate
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
        
    Returns:
        True if all values are within range, False otherwise
    """
    if column not in df.columns:
        print(f"Column '{column}' not found in DataFrame")
        return False
    
    if min_val is not None:
        if (df[column] < min_val).any():
            print(f"Column '{column}' has values below minimum {min_val}")
            return False
    
    if max_val is not None:
        if (df[column] > max_val).any():
            print(f"Column '{column}' has values above maximum {max_val}")
            return False
    
    return True


def validate_no_missing_values(df: pd.DataFrame, columns: Optional[List[str]] = None) -> bool:
    """
    Validate that specified columns have no missing values.
    
    Args:
        df: The DataFrame to validate
        columns: List of columns to check. If None, checks all columns.
        
    Returns:
        True if no missing values found, False otherwise
    """
    if columns is None:
        columns = df.columns.tolist()
    
    for column in columns:
        if column in df.columns:
            missing_count = df[column].isnull().sum()
            if missing_count > 0:
                print(f"Column '{column}' has {missing_count} missing values")
                return False
    
    return True


def validate_ecommerce_data(df: pd.DataFrame) -> bool:
    """
    Validate that the e-commerce dataset meets expected quality standards.
    
    Args:
        df: The e-commerce DataFrame to validate
        
    Returns:
        True if data passes all validation checks, False otherwise
    """
    print("Validating e-commerce dataset...")
    
    # Check required columns
    required_columns = ['date', 'product_id', 'price', 'sales', 'marketing_spend', 'day_of_year']
    if not validate_dataframe_structure(df, required_columns):
        return False
    
    # Check data types
    expected_types = {
        'price': 'float64',
        'sales': 'int64',
        'marketing_spend': 'float64',
        'day_of_year': 'int64'
    }
    if not validate_data_types(df, expected_types):
        return False
    
    # Check for missing values
    if not validate_no_missing_values(df):
        return False
    
    # Check numeric ranges
    if not validate_numeric_range(df, 'price', min_val=0):
        return False
    
    if not validate_numeric_range(df, 'sales', min_val=0):
        return False
    
    if not validate_numeric_range(df, 'marketing_spend', min_val=0):
        return False
    
    if not validate_numeric_range(df, 'day_of_year', min_val=1, max_val=366):
        return False
    
    print("✅ All validation checks passed!")
    return True
