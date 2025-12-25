from __future__ import annotations

import logging
from typing import Dict

import librosa
import numpy as np
import pyloudnorm as pyln

from ingestion.loader import AudioSample
from utils.constants import EPSILON
from utils.exceptions import FeatureExtractionError

logger = logging.getLogger(__name__)


def extract_basic_features(audio: AudioSample) -> Dict[str, float]:
    """
    Compute core loudness, energy, and timbral statistics.
    
    Args:
        audio: Input audio sample
        
    Returns:
        Dictionary of basic audio features
        
    Raises:
        FeatureExtractionError: If feature extraction fails
    """
    try:
        y = audio.waveform
        sr = audio.sample_rate
        
        if y.size == 0:
            raise FeatureExtractionError("Cannot extract features from empty audio")
        
        meter = pyln.Meter(sr)
        lufs = meter.integrated_loudness(y)
        
        rms = float(librosa.feature.rms(y=y).mean())
        zcr = float(librosa.feature.zero_crossing_rate(y).mean())
        spec_cent = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
        spec_bw = float(librosa.feature.spectral_bandwidth(y=y, sr=sr).mean())
        flatness = float(librosa.feature.spectral_flatness(y=y).mean())
        
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_rate = float(librosa.beat.plp(onset_envelope=onset_env, sr=sr).mean())
        
        rms_val = np.sqrt(np.mean(np.square(y)))
        peak_val = np.max(np.abs(y))
        crest_factor = float(peak_val / (rms_val + EPSILON))
        
        feats = {
            "lufs": float(lufs) if np.isfinite(lufs) else -70.0,
            "rms": rms,
            "zcr": zcr,
            "spec_cent": spec_cent,
            "spec_bw": spec_bw,
            "flatness": flatness,
            "onset_rate": onset_rate,
            "crest_factor": crest_factor,
        }
        
        for key, value in feats.items():
            if not np.isfinite(value):
                logger.warning(f"Feature {key} has non-finite value: {value}, setting to 0")
                feats[key] = 0.0
        
        logger.debug(f"Extracted {len(feats)} basic features")
        return feats
        
    except Exception as e:
        logger.error(f"Error extracting basic features: {e}")
        raise FeatureExtractionError(f"Failed to extract basic features: {e}") from e
