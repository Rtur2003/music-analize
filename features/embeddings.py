from __future__ import annotations

import logging
from typing import Dict

import numpy as np

from ingestion.loader import AudioSample
from utils.exceptions import FeatureExtractionError

logger = logging.getLogger(__name__)


def _get_bundle(model_name: str):
    """
    Get the torch audio bundle for the specified model.
    
    Args:
        model_name: Name of the embedding model
        
    Returns:
        Torchaudio pipeline bundle
        
    Raises:
        FeatureExtractionError: If model is not available
    """
    try:
        import torchaudio
    except ImportError as exc:
        logger.error("torchaudio is required for embedding extraction")
        raise FeatureExtractionError("torchaudio is required for embedding extraction.") from exc

    bundles = {
        "wav2vec2_base": torchaudio.pipelines.WAV2VEC2_BASE,
        "hubert_base": torchaudio.pipelines.HUBERT_BASE,
    }
    if hasattr(torchaudio.pipelines, "MERT_V1"):
        bundles["mert_v1"] = torchaudio.pipelines.MERT_V1
    
    if model_name not in bundles:
        available = ', '.join(bundles.keys())
        raise FeatureExtractionError(
            f"Unknown embedding model '{model_name}'. Available models: {available}"
        )
    
    return bundles[model_name]


def extract_embedding(audio: AudioSample, model_name: str = "wav2vec2_base") -> np.ndarray:
    """
    Extract a single embedding vector by mean-pooling frame-level representations.
    
    Args:
        audio: Input audio sample
        model_name: Name of the embedding model to use
        
    Returns:
        Embedding vector as numpy array
        
    Raises:
        FeatureExtractionError: If embedding extraction fails
    """
    try:
        import torch
    except ImportError as exc:
        logger.error("torch is required for embedding extraction")
        raise FeatureExtractionError("torch is required for embedding extraction.") from exc

    try:
        logger.debug(f"Loading embedding model: {model_name}")
        bundle = _get_bundle(model_name)
        model = bundle.get_model().eval()
        
        waveform = torch.tensor(audio.waveform, dtype=torch.float32).unsqueeze(0)
        
        with torch.inference_mode():
            features, lengths = model(waveform)
        
        if isinstance(features, (list, tuple)):
            features = features[0]
        
        pooled = features.mean(dim=1).squeeze(0)
        embedding = pooled.detach().cpu().numpy()
        
        if not np.all(np.isfinite(embedding)):
            logger.warning("Embedding contains non-finite values")
            embedding = np.nan_to_num(embedding, nan=0.0, posinf=0.0, neginf=0.0)
        
        logger.debug(f"Extracted embedding of dimension {embedding.shape}")
        return embedding
        
    except Exception as e:
        logger.error(f"Error extracting embedding: {e}")
        raise FeatureExtractionError(f"Failed to extract embedding: {e}") from e


def embedding_feature_dict(audio: AudioSample, model_name: str = "wav2vec2_base") -> Dict[str, float]:
    """
    Extract embedding as a feature dictionary.
    
    Args:
        audio: Input audio sample
        model_name: Name of the embedding model to use
        
    Returns:
        Dictionary mapping feature names to values
        
    Raises:
        FeatureExtractionError: If embedding extraction fails
    """
    vec = extract_embedding(audio, model_name=model_name)
    return {f"embed_{i}": float(v) for i, v in enumerate(vec)}
