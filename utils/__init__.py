"""Utility functions and classes for the music analysis system."""

from .constants import *
from .exceptions import (
    AudioLoadError,
    ConfigurationError,
    FeatureExtractionError,
    ModelError,
    ModelNotFoundError,
    ModelPredictionError,
    MusicAnalysisError,
    ValidationError,
)
from .logging_config import get_logger, setup_logging
from .validators import (
    validate_array_not_empty,
    validate_audio_file,
    validate_config_dict,
    validate_duration,
    validate_model_path,
    validate_positive_number,
    validate_probability,
    validate_sample_rate,
)

__all__ = [
    # Exceptions
    "MusicAnalysisError",
    "AudioLoadError",
    "FeatureExtractionError",
    "ModelError",
    "ModelNotFoundError",
    "ModelPredictionError",
    "ConfigurationError",
    "ValidationError",
    # Logging
    "setup_logging",
    "get_logger",
    # Validators
    "validate_audio_file",
    "validate_sample_rate",
    "validate_duration",
    "validate_array_not_empty",
    "validate_model_path",
    "validate_config_dict",
    "validate_positive_number",
    "validate_probability",
]
