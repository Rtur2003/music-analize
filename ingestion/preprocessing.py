from __future__ import annotations

import logging
from typing import List

import numpy as np
import pyloudnorm as pyln

from ingestion.loader import AudioSample
from utils.constants import DEFAULT_LUFS_TARGET, EPSILON
from utils.exceptions import AudioLoadError

logger = logging.getLogger(__name__)


def normalize_loudness(
    audio: AudioSample,
    target_lufs: float = DEFAULT_LUFS_TARGET,
) -> AudioSample:
    """
    Loudness normalization to target LUFS using ITU-R BS.1770.
    
    Args:
        audio: Input audio sample
        target_lufs: Target loudness in LUFS (default: -23.0)
        
    Returns:
        Normalized AudioSample
        
    Raises:
        AudioLoadError: If normalization fails
    """
    try:
        if audio.waveform.size == 0:
            raise AudioLoadError("Cannot normalize empty audio")
        
        meter = pyln.Meter(audio.sample_rate)
        loudness = meter.integrated_loudness(audio.waveform)
        
        if not np.isfinite(loudness):
            logger.warning(f"Invalid loudness measurement: {loudness}, returning original audio")
            return audio
        
        normalized = pyln.normalize.loudness(audio.waveform, loudness, target_lufs)
        
        if not np.all(np.isfinite(normalized)):
            logger.warning("Normalization produced non-finite values, returning original audio")
            return audio
        
        logger.debug(f"Normalized audio from {loudness:.2f} LUFS to {target_lufs:.2f} LUFS")
        return AudioSample(waveform=normalized, sample_rate=audio.sample_rate)
        
    except Exception as e:
        logger.error(f"Error normalizing loudness: {e}")
        raise AudioLoadError(f"Failed to normalize loudness: {e}") from e


def segment_audio(
    audio: AudioSample,
    segment_duration_sec: float = 10.0,
    hop_duration_sec: float | None = None,
) -> List[AudioSample]:
    """
    Slice audio into overlapping segments for windowed inference.
    
    Args:
        audio: Input audio sample
        segment_duration_sec: Duration of each segment in seconds
        hop_duration_sec: Hop size between segments (default: same as segment_duration)
        
    Returns:
        List of audio segments
        
    Raises:
        AudioLoadError: If parameters are invalid
    """
    if segment_duration_sec <= 0:
        raise AudioLoadError(f"Segment duration must be positive, got: {segment_duration_sec}")
    
    hop = hop_duration_sec if hop_duration_sec is not None else segment_duration_sec
    
    if hop <= 0:
        raise AudioLoadError(f"Hop duration must be positive, got: {hop}")
    
    seg_len = int(segment_duration_sec * audio.sample_rate)
    hop_len = int(hop * audio.sample_rate)
    
    if seg_len <= 0:
        raise AudioLoadError(f"Invalid segment length: {seg_len}")
    
    segments: List[AudioSample] = []
    waveform_len = len(audio.waveform)
    
    if waveform_len < seg_len:
        logger.debug(f"Audio shorter than segment length, returning as single segment")
        return [audio]
    
    for start in range(0, max(waveform_len - seg_len + 1, 1), hop_len):
        end = start + seg_len
        if end > waveform_len:
            break
        chunk = audio.waveform[start:end]
        segments.append(AudioSample(waveform=chunk, sample_rate=audio.sample_rate))
    
    if not segments:
        logger.debug(f"No segments extracted, returning original audio")
        return [audio]
    
    logger.debug(f"Created {len(segments)} segments from audio")
    return segments


def peak_normalize(audio: AudioSample, peak_dbfs: float = -1.0) -> AudioSample:
    """
    Peak normalize to a target dBFS to avoid clipping after processing.
    
    Args:
        audio: Input audio sample
        peak_dbfs: Target peak level in dBFS (default: -1.0)
        
    Returns:
        Normalized AudioSample
        
    Raises:
        AudioLoadError: If normalization fails
    """
    try:
        peak = np.max(np.abs(audio.waveform))
        
        if peak < EPSILON:
            logger.warning("Audio peak is zero or near-zero, returning original audio")
            return audio
        
        target_peak = 10 ** (peak_dbfs / 20)
        scaled = audio.waveform / peak * target_peak
        
        if not np.all(np.isfinite(scaled)):
            logger.warning("Peak normalization produced non-finite values, returning original audio")
            return audio
        
        logger.debug(f"Peak normalized audio to {peak_dbfs:.1f} dBFS")
        return AudioSample(waveform=scaled, sample_rate=audio.sample_rate)
        
    except Exception as e:
        logger.error(f"Error in peak normalization: {e}")
        raise AudioLoadError(f"Failed to peak normalize: {e}") from e
