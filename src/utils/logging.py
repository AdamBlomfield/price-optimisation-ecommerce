"""
Logging utilities for the price optimization project.

This module provides standardized logging configuration that can be reused
across different scripts and modules in the project.
"""
import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler


def setup_logging() -> logging.Logger:
    """
    Setup logging configuration with both console and file output.
    
    This function creates a logger with:
    - Console output for immediate feedback
    - File output with rotation to prevent large log files
    - Standardized formatting across the project
    
    Returns:
        A configured logger instance.
        
    Example:
        >>> logger = setup_logging()
        >>> logger.info("This will appear in both console and log file")
    """
    # Import config here to avoid circular imports
    import config
    
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, config.LOGGING["level"]))
    
    # Clear any existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        config.LOGGING["format"],
        datefmt=config.LOGGING["date_format"]
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, config.LOGGING["level"]))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        config.LOGGING["log_file"],
        maxBytes=config.LOGGING["max_file_size"],
        backupCount=config.LOGGING["backup_count"]
    )
    file_handler.setLevel(getattr(logging, config.LOGGING["level"]))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger instance with the project's standard configuration.
    
    Args:
        name: The name for the logger. If None, uses the calling module's name.
        
    Returns:
        A configured logger instance.
        
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Logging from my module")
    """
    if name is None:
        # Get the name of the calling module
        import inspect
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'unknown')
    
    logger = logging.getLogger(name)
    
    # Only configure if not already configured
    if not logger.handlers:
        # Import config here to avoid circular imports
        import config
        
        logger.setLevel(getattr(logging, config.LOGGING["level"]))
        
        # Create formatter
        formatter = logging.Formatter(
            config.LOGGING["format"],
            datefmt=config.LOGGING["date_format"]
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, config.LOGGING["level"]))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler with rotation
        file_handler = RotatingFileHandler(
            config.LOGGING["log_file"],
            maxBytes=config.LOGGING["max_file_size"],
            backupCount=config.LOGGING["backup_count"]
        )
        file_handler.setLevel(getattr(logging, config.LOGGING["level"]))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger
