"""Tests for utility functions and validators."""

import numpy as np
import pytest
from pathlib import Path

from utils.exceptions import ValidationError, ConfigurationError
from utils.validators import (
    validate_audio_file,
    validate_sample_rate,
    validate_duration,
    validate_array_not_empty,
    validate_positive_number,
    validate_probability,
)


def test_validate_sample_rate():
    """Test sample rate validation."""
    validate_sample_rate(44100)
    validate_sample_rate(48000)
    
    with pytest.raises(ValidationError):
        validate_sample_rate(0)
    
    with pytest.raises(ValidationError):
        validate_sample_rate(-1000)
    
    with pytest.raises(ValidationError):
        validate_sample_rate(7000)
    
    with pytest.raises(ValidationError):
        validate_sample_rate(200000)


def test_validate_duration():
    """Test duration validation."""
    validate_duration(1.0)
    validate_duration(30.0)
    validate_duration(300.0)
    
    with pytest.raises(ValidationError):
        validate_duration(0)
    
    with pytest.raises(ValidationError):
        validate_duration(-1)
    
    with pytest.raises(ValidationError):
        validate_duration(0.05)
    
    with pytest.raises(ValidationError):
        validate_duration(1000)


def test_validate_array_not_empty():
    """Test array validation."""
    validate_array_not_empty(np.array([1, 2, 3]))
    validate_array_not_empty(np.array([[1], [2]]))
    
    with pytest.raises(ValidationError):
        validate_array_not_empty(None)
    
    with pytest.raises(ValidationError):
        validate_array_not_empty(np.array([]))


def test_validate_positive_number():
    """Test positive number validation."""
    validate_positive_number(1.0)
    validate_positive_number(0.001)
    validate_positive_number(1000)
    
    with pytest.raises(ValidationError):
        validate_positive_number(0)
    
    with pytest.raises(ValidationError):
        validate_positive_number(-1)


def test_validate_probability():
    """Test probability validation."""
    validate_probability(0.0)
    validate_probability(0.5)
    validate_probability(1.0)
    
    with pytest.raises(ValidationError):
        validate_probability(-0.1)
    
    with pytest.raises(ValidationError):
        validate_probability(1.1)
