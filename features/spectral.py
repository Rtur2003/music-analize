from __future__ import annotations

import logging
from typing import Dict

import librosa
import numpy as np

from ingestion.loader import AudioSample
from utils.constants import DEFAULT_HOP_LENGTH, DEFAULT_N_FFT, EPSILON
from utils.exceptions import FeatureExtractionError

logger = logging.getLogger(__name__)


def compute_mel_spectrogram(
    audio: AudioSample,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
    n_mels: int = 128,
    fmin: int = 20,
    fmax: int | None = 20000,
) -> np.ndarray:
    """
    Compute mel-spectrogram with validation.
    
    Args:
        audio: Input audio sample
        n_fft: FFT window size
        hop_length: Hop length between frames
        n_mels: Number of mel bands
        fmin: Minimum frequency
        fmax: Maximum frequency (None = Nyquist)
        
    Returns:
        Mel-spectrogram in dB scale
        
    Raises:
        FeatureExtractionError: If computation fails
    """
    try:
        if audio.waveform.size == 0:
            raise FeatureExtractionError("Cannot compute mel-spectrogram from empty audio")
        
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
        
        mel_db = librosa.power_to_db(S, ref=np.max)
        
        if not np.all(np.isfinite(mel_db)):
            logger.warning("Mel-spectrogram contains non-finite values")
            mel_db = np.nan_to_num(mel_db, nan=0.0, posinf=0.0, neginf=-80.0)
        
        logger.debug(f"Computed mel-spectrogram: shape={mel_db.shape}")
        return mel_db
        
    except Exception as e:
        logger.error(f"Error computing mel-spectrogram: {e}")
        raise FeatureExtractionError(f"Failed to compute mel-spectrogram: {e}") from e


def compute_mfcc_stats(
    audio: AudioSample,
    n_mfcc: int = 20,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
) -> Dict[str, float]:
    """
    Compute MFCC statistics (mean and standard deviation).
    
    Args:
        audio: Input audio sample
        n_mfcc: Number of MFCC coefficients
        n_fft: FFT window size
        hop_length: Hop length between frames
        
    Returns:
        Dictionary of MFCC statistics
        
    Raises:
        FeatureExtractionError: If computation fails
    """
    try:
        mfcc = librosa.feature.mfcc(
            y=audio.waveform,
            sr=audio.sample_rate,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length,
        )
        
        mfcc_mean = mfcc.mean(axis=1)
        mfcc_std = mfcc.std(axis=1)
        
        feats = {f"mfcc_mean_{i}": float(v) for i, v in enumerate(mfcc_mean)}
        feats.update({f"mfcc_std_{i}": float(v) for i, v in enumerate(mfcc_std)})
        
        for key, value in feats.items():
            if not np.isfinite(value):
                logger.warning(f"MFCC feature {key} has non-finite value, setting to 0")
                feats[key] = 0.0
        
        logger.debug(f"Computed {len(feats)} MFCC features")
        return feats
        
    except Exception as e:
        logger.error(f"Error computing MFCC stats: {e}")
        raise FeatureExtractionError(f"Failed to compute MFCC stats: {e}") from e


def harmonic_percussive_ratio(audio: AudioSample) -> Dict[str, float]:
    """
    Compute harmonic-percussive separation ratio.
    
    Args:
        audio: Input audio sample
        
    Returns:
        Dictionary with harmonic/percussive energies and ratio
        
    Raises:
        FeatureExtractionError: If computation fails
    """
    try:
        harmonic, percussive = librosa.effects.hpss(audio.waveform)
        
        harm_energy = float(np.mean(np.abs(harmonic)))
        perc_energy = float(np.mean(np.abs(percussive)))
        ratio = harm_energy / (perc_energy + EPSILON)
        
        result = {
            "harmonic_percussive_ratio": ratio,
            "harmonic_energy": harm_energy,
            "percussive_energy": perc_energy,
        }
        
        for key, value in result.items():
            if not np.isfinite(value):
                logger.warning(f"HP feature {key} has non-finite value, setting to 0")
                result[key] = 0.0
        
        logger.debug("Computed harmonic-percussive features")
        return result
        
    except Exception as e:
        logger.error(f"Error computing harmonic-percussive ratio: {e}")
        raise FeatureExtractionError(f"Failed to compute harmonic-percussive ratio: {e}") from e


def chroma_features(
    audio: AudioSample,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
) -> Dict[str, float]:
    """
    Compute chroma feature statistics.
    
    Args:
        audio: Input audio sample
        n_fft: FFT window size
        hop_length: Hop length between frames
        
    Returns:
        Dictionary of chroma statistics
        
    Raises:
        FeatureExtractionError: If computation fails
    """
    try:
        chroma = librosa.feature.chroma_cqt(y=audio.waveform, sr=audio.sample_rate)
        
        chroma_mean = chroma.mean(axis=1)
        chroma_std = chroma.std(axis=1)
        
        feats = {f"chroma_mean_{i}": float(v) for i, v in enumerate(chroma_mean)}
        feats.update({f"chroma_std_{i}": float(v) for i, v in enumerate(chroma_std)})
        
        for key, value in feats.items():
            if not np.isfinite(value):
                logger.warning(f"Chroma feature {key} has non-finite value, setting to 0")
                feats[key] = 0.0
        
        logger.debug(f"Computed {len(feats)} chroma features")
        return feats
        
    except Exception as e:
        logger.error(f"Error computing chroma features: {e}")
        raise FeatureExtractionError(f"Failed to compute chroma features: {e}") from e


def spectral_centroid_series(
    audio: AudioSample,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
) -> np.ndarray:
    """
    Compute spectral centroid time series.
    
    Args:
        audio: Input audio sample
        n_fft: FFT window size
        hop_length: Hop length between frames
        
    Returns:
        Spectral centroid array
        
    Raises:
        FeatureExtractionError: If computation fails
    """
    try:
        centroid = librosa.feature.spectral_centroid(
            y=audio.waveform,
            sr=audio.sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
        )
        
        if not np.all(np.isfinite(centroid)):
            logger.warning("Spectral centroid contains non-finite values")
            centroid = np.nan_to_num(centroid, nan=0.0, posinf=0.0, neginf=0.0)
        
        logger.debug(f"Computed spectral centroid: shape={centroid.shape}")
        return centroid
        
    except Exception as e:
        logger.error(f"Error computing spectral centroid: {e}")
        raise FeatureExtractionError(f"Failed to compute spectral centroid: {e}") from e
