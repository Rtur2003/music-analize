"""Validation utilities for input data and configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from utils.constants import (
    MAX_AUDIO_DURATION_SEC,
    MAX_FILE_SIZE,
    MIN_AUDIO_DURATION_SEC,
    SUPPORTED_AUDIO_EXTENSIONS,
)
from utils.exceptions import ValidationError


def validate_audio_file(path: Path) -> None:
    """
    Validate that an audio file exists and is readable.
    
    Args:
        path: Path to audio file
        
    Raises:
        ValidationError: If file is invalid
    """
    if not path.exists():
        raise ValidationError(f"Audio file not found: {path}")
    
    if not path.is_file():
        raise ValidationError(f"Path is not a file: {path}")
    
    if path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
        raise ValidationError(
            f"Unsupported file format: {path.suffix}. "
            f"Supported formats: {', '.join(SUPPORTED_AUDIO_EXTENSIONS)}"
        )
    
    file_size = path.stat().st_size
    if file_size == 0:
        raise ValidationError(f"Audio file is empty: {path}")
    
    if file_size > MAX_FILE_SIZE:
        raise ValidationError(
            f"File size {file_size} exceeds maximum allowed size {MAX_FILE_SIZE} bytes"
        )


def validate_sample_rate(sample_rate: int) -> None:
    """
    Validate sample rate is within reasonable bounds.
    
    Args:
        sample_rate: Audio sample rate in Hz
        
    Raises:
        ValidationError: If sample rate is invalid
    """
    if sample_rate <= 0:
        raise ValidationError(f"Sample rate must be positive, got: {sample_rate}")
    
    if sample_rate < 8000:
        raise ValidationError(f"Sample rate too low: {sample_rate}. Minimum is 8000 Hz")
    
    if sample_rate > 192000:
        raise ValidationError(f"Sample rate too high: {sample_rate}. Maximum is 192000 Hz")


def validate_duration(duration: float) -> None:
    """
    Validate audio duration is within reasonable bounds.
    
    Args:
        duration: Duration in seconds
        
    Raises:
        ValidationError: If duration is invalid
    """
    if duration <= 0:
        raise ValidationError(f"Duration must be positive, got: {duration}")
    
    if duration < MIN_AUDIO_DURATION_SEC:
        raise ValidationError(
            f"Duration {duration}s too short. Minimum is {MIN_AUDIO_DURATION_SEC}s"
        )
    
    if duration > MAX_AUDIO_DURATION_SEC:
        raise ValidationError(
            f"Duration {duration}s too long. Maximum is {MAX_AUDIO_DURATION_SEC}s"
        )


def validate_array_not_empty(arr: np.ndarray, name: str = "array") -> None:
    """
    Validate that a numpy array is not empty.
    
    Args:
        arr: Numpy array to validate
        name: Name of the array for error messages
        
    Raises:
        ValidationError: If array is empty
    """
    if arr is None:
        raise ValidationError(f"{name} cannot be None")
    
    if arr.size == 0:
        raise ValidationError(f"{name} cannot be empty")


def validate_model_path(path: Path, model_name: str = "model") -> None:
    """
    Validate that a model file exists.
    
    Args:
        path: Path to model file
        model_name: Name of the model for error messages
        
    Raises:
        ValidationError: If model file is invalid
    """
    if not path.exists():
        raise ValidationError(f"{model_name} not found at: {path}")
    
    if not path.is_file():
        raise ValidationError(f"{model_name} path is not a file: {path}")


def validate_config_dict(config: Dict[str, Any], required_keys: List[str]) -> None:
    """
    Validate that a configuration dictionary contains required keys.
    
    Args:
        config: Configuration dictionary
        required_keys: List of required key names
        
    Raises:
        ValidationError: If required keys are missing
    """
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValidationError(f"Missing required configuration keys: {', '.join(missing_keys)}")


def validate_positive_number(value: float, name: str = "value") -> None:
    """
    Validate that a number is positive.
    
    Args:
        value: Number to validate
        name: Name of the value for error messages
        
    Raises:
        ValidationError: If value is not positive
    """
    if value <= 0:
        raise ValidationError(f"{name} must be positive, got: {value}")


def validate_probability(value: float, name: str = "probability") -> None:
    """
    Validate that a value is a valid probability (0-1).
    
    Args:
        value: Value to validate
        name: Name of the value for error messages
        
    Raises:
        ValidationError: If value is not in [0, 1]
    """
    if not 0 <= value <= 1:
        raise ValidationError(f"{name} must be between 0 and 1, got: {value}")
