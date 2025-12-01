from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import librosa
import numpy as np


@dataclass
class AudioSample:
    waveform: np.ndarray
    sample_rate: int


def load_audio(path: Path | str, sample_rate: int = 44100, mono: bool = True) -> AudioSample:
    """
    Load an audio file into a waveform array.
    """
    audio_path = Path(path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    y, sr = librosa.load(str(audio_path), sr=sample_rate, mono=mono)
    return AudioSample(waveform=y, sample_rate=sr)


def pad_or_trim(audio: AudioSample, target_duration_sec: float) -> AudioSample:
    """
    Pad or trim audio to a fixed duration in seconds.
    """
    target_len = int(target_duration_sec * audio.sample_rate)
    if audio.waveform.shape[-1] > target_len:
        trimmed = audio.waveform[:target_len]
        return AudioSample(waveform=trimmed, sample_rate=audio.sample_rate)
    if audio.waveform.shape[-1] < target_len:
        pad_width = target_len - audio.waveform.shape[-1]
        padded = np.pad(audio.waveform, (0, pad_width), mode="constant")
        return AudioSample(waveform=padded, sample_rate=audio.sample_rate)
    return audio


def load_and_prepare(path: Path | str, sample_rate: int, mono: bool, target_duration_sec: float) -> AudioSample:
    audio = load_audio(path, sample_rate=sample_rate, mono=mono)
    return pad_or_trim(audio, target_duration_sec=target_duration_sec)
