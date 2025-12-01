from __future__ import annotations

from typing import Dict

import librosa
import numpy as np
import pyloudnorm as pyln

from ingestion.loader import AudioSample


def extract_basic_features(audio: AudioSample) -> Dict[str, float]:
    """
    Compute core loudness, energy, and timbral statistics.
    """
    y = audio.waveform
    sr = audio.sample_rate
    meter = pyln.Meter(sr)
    lufs = meter.integrated_loudness(y)
    rms = float(librosa.feature.rms(y=y).mean())
    zcr = float(librosa.feature.zero_crossing_rate(y).mean())
    spec_cent = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
    spec_bw = float(librosa.feature.spectral_bandwidth(y=y, sr=sr).mean())
    flatness = float(librosa.feature.spectral_flatness(y=y).mean())
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_rate = float(librosa.beat.plp(onset_envelope=onset_env, sr=sr).mean())
    crest_factor = float(np.max(np.abs(y)) / (np.sqrt(np.mean(np.square(y))) + 1e-9))

    feats = {
        "lufs": float(lufs),
        "rms": rms,
        "zcr": zcr,
        "spec_cent": spec_cent,
        "spec_bw": spec_bw,
        "flatness": flatness,
        "onset_rate": onset_rate,
        "crest_factor": crest_factor,
    }
    return feats
