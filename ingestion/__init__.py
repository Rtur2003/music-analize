"""Ingestion package for audio loading and preprocessing."""

from .loader import AudioSample, load_and_prepare, load_audio, pad_or_trim
from .preprocessing import normalize_loudness, peak_normalize, segment_audio

__all__ = [
    "AudioSample",
    "load_audio",
    "load_and_prepare",
    "pad_or_trim",
    "normalize_loudness",
    "peak_normalize",
    "segment_audio",
]
