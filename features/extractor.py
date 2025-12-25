from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np

from config.settings import Settings
from ingestion.loader import AudioSample
from utils.exceptions import FeatureExtractionError
from .base_features import extract_basic_features
from .embeddings import extract_embedding
from .spectral import (
    chroma_features,
    compute_mfcc_stats,
    compute_mel_spectrogram,
    harmonic_percussive_ratio,
    spectral_centroid_series,
)

logger = logging.getLogger(__name__)


def extract_all(
    audio: AudioSample,
    settings: Settings,
    embed_model_name: Optional[str] = None,
) -> Tuple[Dict[str, float], Optional[np.ndarray], np.ndarray, np.ndarray]:
    """
    Aggregate basic, spectral, and embedding features for downstream tasks.
    
    Args:
        audio: Input audio sample
        settings: Application settings
        embed_model_name: Name of embedding model (optional)
        
    Returns:
        Tuple of (features_dict, embedding, mel_spectrogram, spectral_centroid)
        
    Raises:
        FeatureExtractionError: If feature extraction fails
    """
    try:
        logger.debug("Starting feature extraction")
        
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
                logger.debug(f"Extracting embedding with model: {embed_model_name}")
                embedding = extract_embedding(audio, model_name=embed_model_name)
                if embedding is not None:
                    features.update({f"embed_{i}": float(v) for i, v in enumerate(embedding)})
                    logger.debug(f"Extracted embedding of dimension {len(embedding)}")
            except Exception as exc:
                logger.warning(f"Embedding extraction failed: {exc}")
                features["embedding_error"] = str(exc)
        
        logger.info(f"Successfully extracted {len(features)} features")
        return features, embedding, mel, centroid_series
        
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        raise FeatureExtractionError(f"Failed to extract features: {e}") from e
