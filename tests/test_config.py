"""Tests for configuration module."""

import pytest
from pathlib import Path

from config.settings import (
    AudioConfig,
    FeatureConfig,
    GenreModelConfig,
    AuthenticityModelConfig,
    Settings,
)
from utils.exceptions import ConfigurationError


def test_audio_config_defaults():
    """Test AudioConfig with default values."""
    config = AudioConfig()
    assert config.sample_rate == 44100
    assert config.mono is True
    assert config.target_duration_sec == 30
    assert config.normalize_lufs == -23.0


def test_audio_config_validation():
    """Test AudioConfig validation."""
    with pytest.raises(ConfigurationError):
        AudioConfig(sample_rate=0)
    
    with pytest.raises(ConfigurationError):
        AudioConfig(target_duration_sec=-1)
    
    with pytest.raises(ConfigurationError):
        AudioConfig(normalize_lufs=10.0)


def test_feature_config_defaults():
    """Test FeatureConfig with default values."""
    config = FeatureConfig()
    assert config.n_fft == 2048
    assert config.hop_length == 512
    assert config.n_mels == 128
    assert config.n_mfcc == 20


def test_feature_config_validation():
    """Test FeatureConfig validation."""
    with pytest.raises(ConfigurationError):
        FeatureConfig(n_fft=0)
    
    with pytest.raises(ConfigurationError):
        FeatureConfig(fmin=1000, fmax=100)


def test_genre_model_config_validation():
    """Test GenreModelConfig validation."""
    config = GenreModelConfig()
    assert config.top_k == 5
    
    with pytest.raises(ConfigurationError):
        GenreModelConfig(top_k=0)


def test_authenticity_model_config_validation():
    """Test AuthenticityModelConfig validation."""
    config = AuthenticityModelConfig()
    assert config.threshold == 0.5
    
    with pytest.raises(ConfigurationError):
        AuthenticityModelConfig(threshold=-0.1)
    
    with pytest.raises(ConfigurationError):
        AuthenticityModelConfig(threshold=1.5)
    
    with pytest.raises(ConfigurationError):
        AuthenticityModelConfig(base_model="invalid")


def test_settings_defaults():
    """Test Settings with default values."""
    settings = Settings()
    assert settings.audio.sample_rate == 44100
    assert settings.features.n_fft == 2048
    assert settings.genre_model.top_k == 5
    assert settings.authenticity_model.threshold == 0.5
