from pathlib import Path

import numpy as np

from config.settings import get_settings
from features.extractor import extract_all
from ingestion.loader import AudioSample


def test_extract_all_shapes():
    sr = 16000
    duration = 1
    t = np.linspace(0, duration, sr * duration, endpoint=False)
    tone = 0.1 * np.sin(2 * np.pi * 440 * t)
    audio = AudioSample(waveform=tone, sample_rate=sr)
    settings = get_settings()
    feats, emb, mel, centroid = extract_all(audio, settings=settings, embed_model_name=None)
    assert "lufs" in feats
    assert mel.ndim == 2
    assert emb is None
    assert centroid.ndim == 2
