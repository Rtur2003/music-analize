from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml

from utils.constants import (
    DEFAULT_DURATION_SEC,
    DEFAULT_FMAX,
    DEFAULT_FMIN,
    DEFAULT_HOP_LENGTH,
    DEFAULT_LUFS_TARGET,
    DEFAULT_N_FFT,
    DEFAULT_N_MFCC,
    DEFAULT_N_MELS,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_THRESHOLD,
    DEFAULT_TOP_K,
)
from utils.exceptions import ConfigurationError
from utils.validators import validate_positive_number, validate_probability, validate_sample_rate

logger = logging.getLogger(__name__)


@dataclass
class AudioConfig:
    """Audio processing configuration."""
    sample_rate: int = DEFAULT_SAMPLE_RATE
    mono: bool = True
    target_duration_sec: int = DEFAULT_DURATION_SEC
    normalize_lufs: float = DEFAULT_LUFS_TARGET
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        validate_sample_rate(self.sample_rate)
        validate_positive_number(self.target_duration_sec, "target_duration_sec")
        if self.normalize_lufs > 0:
            raise ConfigurationError(f"normalize_lufs should be negative, got: {self.normalize_lufs}")


@dataclass
class FeatureConfig:
    """Feature extraction configuration."""
    n_fft: int = DEFAULT_N_FFT
    hop_length: int = DEFAULT_HOP_LENGTH
    n_mels: int = DEFAULT_N_MELS
    n_mfcc: int = DEFAULT_N_MFCC
    fmin: int = DEFAULT_FMIN
    fmax: int = DEFAULT_FMAX
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        validate_positive_number(self.n_fft, "n_fft")
        validate_positive_number(self.hop_length, "hop_length")
        validate_positive_number(self.n_mels, "n_mels")
        validate_positive_number(self.n_mfcc, "n_mfcc")
        
        if self.fmin >= self.fmax:
            raise ConfigurationError(f"fmin ({self.fmin}) must be less than fmax ({self.fmax})")
        
        if self.hop_length >= self.n_fft:
            logger.warning(f"hop_length ({self.hop_length}) should typically be less than n_fft ({self.n_fft})")


@dataclass
class GenreModelConfig:
    """Genre model configuration."""
    embedding_model: str = "wav2vec2_base"
    top_k: int = DEFAULT_TOP_K
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.top_k <= 0:
            raise ConfigurationError(f"top_k must be positive, got: {self.top_k}")


@dataclass
class AuthenticityModelConfig:
    """Authenticity model configuration."""
    base_model: str = "lightgbm"
    threshold: float = DEFAULT_THRESHOLD
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        validate_probability(self.threshold, "threshold")
        
        valid_models = {"lightgbm", "logreg", "logistic"}
        if self.base_model not in valid_models:
            raise ConfigurationError(
                f"Invalid base_model '{self.base_model}'. Valid options: {valid_models}"
            )


@dataclass
class ReportingConfig:
    """Reporting configuration."""
    output_dir: str = "reports"
    include_pdf: bool = True
    include_html: bool = True
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not self.include_pdf and not self.include_html:
            logger.warning("Both PDF and HTML output are disabled")


@dataclass
class StorageConfig:
    """Storage paths configuration."""
    feature_store: str = "data/processed/features.parquet"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    
    def validate_paths_exist(self, check_dirs: bool = False) -> None:
        """
        Validate that configured paths exist.
        
        Args:
            check_dirs: If True, check that directories exist
        """
        if check_dirs:
            raw_path = Path(self.raw_dir)
            if not raw_path.exists():
                logger.warning(f"Raw data directory does not exist: {raw_path}")


@dataclass
class Settings:
    """Application settings container."""
    audio: AudioConfig = field(default_factory=AudioConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    genre_model: GenreModelConfig = field(default_factory=GenreModelConfig)
    authenticity_model: AuthenticityModelConfig = field(default_factory=AuthenticityModelConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)

    @classmethod
    def from_yaml(cls, path: Path | str = "config/settings.yaml") -> "Settings":
        """
        Load settings from a YAML configuration file.
        
        Args:
            path: Path to YAML configuration file
            
        Returns:
            Settings instance
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        settings_path = Path(path)
        
        if not settings_path.exists():
            logger.warning(f"Configuration file not found: {settings_path}, using defaults")
            return cls()
        
        try:
            with settings_path.open("r", encoding="utf-8") as f:
                data: Dict[str, Any] = yaml.safe_load(f) or {}
            
            logger.info(f"Loaded configuration from {settings_path}")
            
            return cls(
                audio=AudioConfig(**data.get("audio", {})),
                features=FeatureConfig(**data.get("features", {})),
                genre_model=GenreModelConfig(**data.get("models", {}).get("genre", {})),
                authenticity_model=AuthenticityModelConfig(
                    **data.get("models", {}).get("authenticity", {})
                ),
                reporting=ReportingConfig(**data.get("reporting", {})),
                storage=StorageConfig(**data.get("storage", {})),
            )
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse YAML configuration: {e}")
            raise ConfigurationError(f"Invalid YAML configuration: {e}") from e
        except TypeError as e:
            logger.error(f"Invalid configuration parameters: {e}")
            raise ConfigurationError(f"Invalid configuration parameters: {e}") from e


def get_settings(path: Path | str = "config/settings.yaml") -> Settings:
    """
    Get application settings from configuration file.
    
    Args:
        path: Path to configuration file
        
    Returns:
        Settings instance
    """
    return Settings.from_yaml(path)
