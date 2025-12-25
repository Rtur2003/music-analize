"""Custom exception hierarchy for better error handling."""

from __future__ import annotations


class MusicAnalysisError(Exception):
    """Base exception for all music analysis errors."""
    pass


class AudioLoadError(MusicAnalysisError):
    """Error loading or processing audio files."""
    pass


class FeatureExtractionError(MusicAnalysisError):
    """Error during feature extraction."""
    pass


class ModelError(MusicAnalysisError):
    """Error related to model operations."""
    pass


class ModelNotFoundError(ModelError):
    """Model file not found."""
    pass


class ModelPredictionError(ModelError):
    """Error during model prediction."""
    pass


class ConfigurationError(MusicAnalysisError):
    """Configuration validation error."""
    pass


class ValidationError(MusicAnalysisError):
    """Input validation error."""
    pass
