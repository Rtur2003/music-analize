from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pyloudnorm as pyln

from ingestion.loader import AudioSample


def normalize_loudness(audio: AudioSample, target_lufs: float = -23.0) -> AudioSample:
    """
    Loudness normalization to target LUFS using ITU-R BS.1770.
    """
    meter = pyln.Meter(audio.sample_rate)
    loudness = meter.integrated_loudness(audio.waveform)
    normalized = pyln.normalize.loudness(audio.waveform, loudness, target_lufs)
    return AudioSample(waveform=normalized, sample_rate=audio.sample_rate)


def segment_audio(audio: AudioSample, segment_duration_sec: float = 10.0, hop_duration_sec: float | None = None) -> List[AudioSample]:
    """
    Slice audio into overlapping segments for windowed inference.
    """
    hop = hop_duration_sec or segment_duration_sec
    seg_len = int(segment_duration_sec * audio.sample_rate)
    hop_len = int(hop * audio.sample_rate)
    segments: List[AudioSample] = []
    for start in range(0, max(len(audio.waveform) - seg_len + 1, 1), hop_len):
        end = start + seg_len
        if end > len(audio.waveform):
            break
        chunk = audio.waveform[start:end]
        segments.append(AudioSample(waveform=chunk, sample_rate=audio.sample_rate))
    return segments or [audio]


def peak_normalize(audio: AudioSample, peak_dbfs: float = -1.0) -> AudioSample:
    """
    Peak normalize to a target dBFS to avoid clipping after processing.
    """
    peak = np.max(np.abs(audio.waveform))
    if peak == 0:
        return audio
    target_peak = 10 ** (peak_dbfs / 20)
    scaled = audio.waveform / peak * target_peak
    return AudioSample(waveform=scaled, sample_rate=audio.sample_rate)
