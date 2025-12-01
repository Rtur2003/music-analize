from __future__ import annotations

from typing import Dict, Tuple

import librosa
import numpy as np

from ingestion.loader import AudioSample


def compute_mel_spectrogram(
    audio: AudioSample,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    fmin: int = 20,
    fmax: int | None = 20000,
) -> np.ndarray:
    S = librosa.feature.melspectrogram(
        y=audio.waveform,
        sr=audio.sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0,
    )
    return librosa.power_to_db(S, ref=np.max)


def compute_mfcc_stats(
    audio: AudioSample, n_mfcc: int = 20, n_fft: int = 2048, hop_length: int = 512
) -> Dict[str, float]:
    mfcc = librosa.feature.mfcc(
        y=audio.waveform, sr=audio.sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
    )
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)
    feats = {f"mfcc_mean_{i}": float(v) for i, v in enumerate(mfcc_mean)}
    feats.update({f"mfcc_std_{i}": float(v) for i, v in enumerate(mfcc_std)})
    return feats


def harmonic_percussive_ratio(audio: AudioSample) -> Dict[str, float]:
    harmonic, percussive = librosa.effects.hpss(audio.waveform)
    harm_energy = float(np.mean(np.abs(harmonic)))
    perc_energy = float(np.mean(np.abs(percussive)))
    ratio = harm_energy / (perc_energy + 1e-9)
    return {"harmonic_percussive_ratio": ratio, "harmonic_energy": harm_energy, "percussive_energy": perc_energy}


def chroma_features(audio: AudioSample, n_fft: int = 2048, hop_length: int = 512) -> Dict[str, float]:
    chroma = librosa.feature.chroma_cqt(y=audio.waveform, sr=audio.sample_rate)
    chroma_mean = chroma.mean(axis=1)
    chroma_std = chroma.std(axis=1)
    feats = {f"chroma_mean_{i}": float(v) for i, v in enumerate(chroma_mean)}
    feats.update({f"chroma_std_{i}": float(v) for i, v in enumerate(chroma_std)})
    return feats


def spectral_centroid_series(audio: AudioSample, n_fft: int = 2048, hop_length: int = 512) -> np.ndarray:
    return librosa.feature.spectral_centroid(y=audio.waveform, sr=audio.sample_rate, n_fft=n_fft, hop_length=hop_length)
