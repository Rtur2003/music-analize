from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

from config.settings import get_settings
from features.extractor import extract_all
from ingestion.loader import load_and_prepare
from ingestion.preprocessing import normalize_loudness
from models.trainer import load_model


def apply_augmentations(y: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
    return {
        "pitch_up": librosa.effects.pitch_shift(y, sr=sr, n_steps=2),
        "pitch_down": librosa.effects.pitch_shift(y, sr=sr, n_steps=-2),
        "stretch_fast": librosa.effects.time_stretch(y, rate=1.05),
        "stretch_slow": librosa.effects.time_stretch(y, rate=0.95),
        "with_noise": y + 0.002 * np.random.randn(len(y)),
    }


def robustness_eval(
    data_root: Path,
    model_dir: Path = Path("models/artifacts"),
    settings_path: Path | str = "config/settings.yaml",
) -> pd.DataFrame:
    settings = get_settings(settings_path)
    genre_clf = load_model(model_dir / "genre_classifier.joblib")
    auth_clf = load_model(model_dir / "auth_classifier.joblib")

    rows: List[Dict] = []
    for audio_path in tqdm(list(data_root.rglob("*.wav")), desc="Robustness sweep"):
        audio = load_and_prepare(
            audio_path,
            sample_rate=settings.audio.sample_rate,
            mono=settings.audio.mono,
            target_duration_sec=settings.audio.target_duration_sec,
        )
        audio = normalize_loudness(audio, target_lufs=settings.audio.normalize_lufs)

        original_feats, embed, _, _ = extract_all(audio, settings=settings, embed_model_name=settings.genre_model.embedding_model)
        for name, sig in {"original": audio.waveform, **apply_augmentations(audio.waveform, audio.sample_rate)}.items():
            aug_audio = audio.__class__(waveform=sig, sample_rate=audio.sample_rate)
            feats, embed_aug, _, _ = extract_all(aug_audio, settings=settings, embed_model_name=settings.genre_model.embedding_model)
            emb = embed_aug if embed_aug is not None else embed
            if emb is None:
                continue
            genre_pred = genre_clf.predict_top_k(np.array([emb]), k=1)[0][0] if genre_clf else ("unknown", 0.0)
            auth_score = float(auth_clf.predict_proba(np.array([emb]))[0]) if auth_clf else 0.0
            rows.append(
                {
                    "path": str(audio_path),
                    "augmentation": name,
                    "genre_pred": genre_pred[0],
                    "genre_conf": genre_pred[1],
                    "auth_score": auth_score,
                }
            )
    df = pd.DataFrame(rows)
    out = Path("reports/robustness.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


if __name__ == "__main__":
    robustness_eval(Path("data/raw"))
