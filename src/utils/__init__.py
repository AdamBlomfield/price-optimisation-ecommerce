"""
Utility functions and modules for the price optimization project.

This package contains reusable code that can be shared across different
modules in the project, such as logging setup, data validation, and
other common functionality.
"""

from .logging import setup_logging, get_logger
from .validation import (
    validate_dataframe_structure,
    validate_data_types,
    validate_numeric_range,
    validate_no_missing_values,
    validate_ecommerce_data
)
from .file_utils import (
    ensure_directory_exists,
    get_file_size,
    get_file_size_mb,
    backup_file,
    clean_directory,
    get_project_root,
    get_relative_path
)

__all__ = [
    # Logging utilities
    'setup_logging',
    'get_logger',
    # Validation utilities
    'validate_dataframe_structure',
    'validate_data_types',
    'validate_numeric_range',
    'validate_no_missing_values',
    'validate_ecommerce_data',
    # File utilities
    'ensure_directory_exists',
    'get_file_size',
    'get_file_size_mb',
    'backup_file',
    'clean_directory',
    'get_project_root',
    'get_relative_path'
]
