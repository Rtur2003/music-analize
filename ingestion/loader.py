from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

from utils.constants import DEFAULT_SAMPLE_RATE
from utils.exceptions import AudioLoadError
from utils.validators import validate_audio_file, validate_sample_rate

logger = logging.getLogger(__name__)


@dataclass
class AudioSample:
    """Container for audio waveform and associated metadata."""
    waveform: np.ndarray
    sample_rate: int
    
    def __post_init__(self) -> None:
        """Validate audio sample after initialization."""
        if self.waveform.size == 0:
            raise AudioLoadError("Audio waveform cannot be empty")
        validate_sample_rate(self.sample_rate)


def load_audio(
    path: Path | str,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    mono: bool = True,
) -> AudioSample:
    """
    Load an audio file into a waveform array with validation.
    
    Args:
        path: Path to audio file
        sample_rate: Target sample rate for resampling (default: 44100)
        mono: Convert to mono if True (default: True)
        
    Returns:
        AudioSample containing waveform and sample rate
        
    Raises:
        AudioLoadError: If file cannot be loaded
        ValidationError: If file is invalid
    """
    audio_path = Path(path)
    validate_audio_file(audio_path)
    validate_sample_rate(sample_rate)
    
    try:
        logger.debug(f"Loading audio file: {audio_path}")
        y, sr = librosa.load(str(audio_path), sr=sample_rate, mono=mono)
        
        if y is None or len(y) == 0:
            raise AudioLoadError(f"Failed to load audio data from: {audio_path}")
        
        logger.debug(f"Loaded audio: shape={y.shape}, sr={sr}")
        return AudioSample(waveform=y, sample_rate=sr)
        
    except Exception as e:
        if isinstance(e, AudioLoadError):
            raise
        logger.error(f"Error loading audio file {audio_path}: {e}")
        raise AudioLoadError(f"Failed to load audio file: {e}") from e


def pad_or_trim(audio: AudioSample, target_duration_sec: float) -> AudioSample:
    """
    Pad or trim audio to a fixed duration in seconds.
    
    Args:
        audio: Input audio sample
        target_duration_sec: Target duration in seconds
        
    Returns:
        AudioSample with adjusted duration
        
    Raises:
        AudioLoadError: If parameters are invalid
    """
    if target_duration_sec <= 0:
        raise AudioLoadError(f"Target duration must be positive, got: {target_duration_sec}")
    
    target_len = int(target_duration_sec * audio.sample_rate)
    
    if target_len <= 0:
        raise AudioLoadError(f"Invalid target length: {target_len}")
    
    current_len = audio.waveform.shape[-1]
    
    if current_len > target_len:
        logger.debug(f"Trimming audio from {current_len} to {target_len} samples")
        trimmed = audio.waveform[:target_len]
        return AudioSample(waveform=trimmed, sample_rate=audio.sample_rate)
    
    if current_len < target_len:
        pad_width = target_len - current_len
        logger.debug(f"Padding audio from {current_len} to {target_len} samples")
        padded = np.pad(audio.waveform, (0, pad_width), mode="constant")
        return AudioSample(waveform=padded, sample_rate=audio.sample_rate)
    
    return audio


def load_and_prepare(
    path: Path | str,
    sample_rate: int,
    mono: bool,
    target_duration_sec: float,
) -> AudioSample:
    """
    Load and prepare audio file with validation.
    
    Args:
        path: Path to audio file
        sample_rate: Target sample rate
        mono: Convert to mono if True
        target_duration_sec: Target duration in seconds
        
    Returns:
        Prepared AudioSample
        
    Raises:
        AudioLoadError: If loading or preparation fails
        ValidationError: If inputs are invalid
    """
    try:
        audio = load_audio(path, sample_rate=sample_rate, mono=mono)
        return pad_or_trim(audio, target_duration_sec=target_duration_sec)
    except Exception as e:
        logger.error(f"Failed to load and prepare audio from {path}: {e}")
        raise
