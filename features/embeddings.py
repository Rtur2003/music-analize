from __future__ import annotations

from typing import Dict

import numpy as np

from ingestion.loader import AudioSample


def _get_bundle(model_name: str):
    try:
        import torchaudio
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("torchaudio is required for embedding extraction.") from exc

    bundles = {
        "wav2vec2_base": torchaudio.pipelines.WAV2VEC2_BASE,
        "hubert_base": torchaudio.pipelines.HUBERT_BASE,
    }
    if hasattr(torchaudio.pipelines, "MERT_V1"):
        bundles["mert_v1"] = torchaudio.pipelines.MERT_V1  # type: ignore[attr-defined]
    if model_name not in bundles:
        raise ValueError(f"Unknown embedding model '{model_name}'. Available: {list(bundles)}")
    return bundles[model_name]


def extract_embedding(audio: AudioSample, model_name: str = "wav2vec2_base") -> np.ndarray:
    """
    Extract a single embedding vector by mean-pooling frame-level representations.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("torch is required for embedding extraction.") from exc

    bundle = _get_bundle(model_name)
    model = bundle.get_model().eval()
    waveform = torch.tensor(audio.waveform, dtype=torch.float32).unsqueeze(0)
    with torch.inference_mode():
        features, lengths = model(waveform)  # type: ignore[misc]
    if isinstance(features, (list, tuple)):
        features = features[0]
    pooled = features.mean(dim=1).squeeze(0)
    return pooled.detach().cpu().numpy()


def embedding_feature_dict(audio: AudioSample, model_name: str = "wav2vec2_base") -> Dict[str, float]:
    vec = extract_embedding(audio, model_name=model_name)
    return {f"embed_{i}": float(v) for i, v in enumerate(vec)}
