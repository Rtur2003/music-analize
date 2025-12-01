from ingestion.loader import pad_or_trim, AudioSample
import numpy as np


def test_pad_or_trim():
    sr = 16000
    target_sec = 2
    short_audio = AudioSample(waveform=np.zeros(sr), sample_rate=sr)
    padded = pad_or_trim(short_audio, target_duration_sec=target_sec)
    assert len(padded.waveform) == sr * target_sec

    long_audio = AudioSample(waveform=np.zeros(sr * 5), sample_rate=sr)
    trimmed = pad_or_trim(long_audio, target_duration_sec=target_sec)
    assert len(trimmed.waveform) == sr * target_sec
