"""
File utilities for the price optimization project.

This module provides functions for common file operations and path handling.
"""
import os
import shutil
from pathlib import Path
from typing import Union, Optional


def ensure_directory_exists(directory: Union[str, Path]) -> Path:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory: Path to the directory
        
    Returns:
        Path object of the directory
        
    Example:
        >>> ensure_directory_exists("data/processed")
        PosixPath('data/processed')
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_file_size(file_path: Union[str, Path]) -> int:
    """
    Get the size of a file in bytes.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File size in bytes, or 0 if file doesn't exist
    """
    file_path = Path(file_path)
    if file_path.exists():
        return file_path.stat().st_size
    return 0


def get_file_size_mb(file_path: Union[str, Path]) -> float:
    """
    Get the size of a file in megabytes.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File size in MB, or 0.0 if file doesn't exist
    """
    size_bytes = get_file_size(file_path)
    return size_bytes / (1024 * 1024)


def backup_file(file_path: Union[str, Path], backup_suffix: str = ".backup") -> Optional[Path]:
    """
    Create a backup of a file by copying it with a suffix.
    
    Args:
        file_path: Path to the file to backup
        backup_suffix: Suffix to add to the backup filename
        
    Returns:
        Path to the backup file, or None if backup failed
    """
    file_path = Path(file_path)
    if not file_path.exists():
        return None
    
    backup_path = file_path.with_suffix(file_path.suffix + backup_suffix)
    
    try:
        shutil.copy2(file_path, backup_path)
        return backup_path
    except Exception as e:
        print(f"Failed to create backup: {e}")
        return None


def clean_directory(directory: Union[str, Path], pattern: str = "*", keep_files: Optional[list] = None) -> int:
    """
    Clean files from a directory matching a pattern.
    
    Args:
        directory: Directory to clean
        pattern: Glob pattern to match files (default: all files)
        keep_files: List of filenames to keep (optional)
        
    Returns:
        Number of files removed
    """
    directory = Path(directory)
    if not directory.exists():
        return 0
    
    keep_files = keep_files or []
    removed_count = 0
    
    for file_path in directory.glob(pattern):
        if file_path.is_file() and file_path.name not in keep_files:
            try:
                file_path.unlink()
                removed_count += 1
            except Exception as e:
                print(f"Failed to remove {file_path}: {e}")
    
    return removed_count


def get_project_root() -> Path:
    """
    Get the root directory of the project.
    
    Returns:
        Path to the project root directory
    """
    # This assumes the utils module is in src/utils/
    current_file = Path(__file__)
    return current_file.parent.parent.parent


def get_relative_path(file_path: Union[str, Path], from_dir: Optional[Union[str, Path]] = None) -> Path:
    """
    Get the relative path of a file from a given directory.
    
    Args:
        file_path: Path to the file
        from_dir: Directory to calculate relative path from (default: project root)
        
    Returns:
        Relative path
    """
    file_path = Path(file_path)
    from_dir = from_dir or get_project_root()
    from_dir = Path(from_dir)
    
    try:
        return file_path.relative_to(from_dir)
    except ValueError:
        # If file_path is not relative to from_dir, return the absolute path
        return file_path
