from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from config.settings import Settings
from ingestion.loader import AudioSample
from .base_features import extract_basic_features
from .embeddings import extract_embedding
from .spectral import (
    chroma_features,
    compute_mfcc_stats,
    compute_mel_spectrogram,
    harmonic_percussive_ratio,
    spectral_centroid_series,
)


def extract_all(
    audio: AudioSample,
    settings: Settings,
    embed_model_name: Optional[str] = None,
) -> Tuple[Dict[str, float], Optional[np.ndarray], np.ndarray, np.ndarray]:
    """
    Aggregate basic, spectral, and embedding features for downstream tasks.
    """
    basic = extract_basic_features(audio)
    mfcc = compute_mfcc_stats(
        audio,
        n_mfcc=settings.features.n_mfcc,
        n_fft=settings.features.n_fft,
        hop_length=settings.features.hop_length,
    )
    hpr = harmonic_percussive_ratio(audio)
    chroma = chroma_features(
        audio,
        n_fft=settings.features.n_fft,
        hop_length=settings.features.hop_length,
    )
    mel = compute_mel_spectrogram(
        audio,
        n_fft=settings.features.n_fft,
        hop_length=settings.features.hop_length,
        n_mels=settings.features.n_mels,
        fmin=settings.features.fmin,
        fmax=settings.features.fmax,
    )
    centroid_series = spectral_centroid_series(
        audio,
        n_fft=settings.features.n_fft,
        hop_length=settings.features.hop_length,
    )

    features: Dict[str, float] = {**basic, **mfcc, **hpr, **chroma}
    embedding = None
    if embed_model_name:
        try:
            embedding = extract_embedding(audio, model_name=embed_model_name)
            features.update({f"embed_{i}": float(v) for i, v in enumerate(embedding)})
        except Exception as exc:
            features["embedding_error"] = str(exc)

    return features, embedding, mel, centroid_series
