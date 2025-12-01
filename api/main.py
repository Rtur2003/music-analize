from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile

from comparator.stats import compare_real_vs_ai
from config.settings import get_settings
from features.base_features import extract_basic_features
from features.embeddings import extract_embedding
from features.spectral import compute_mfcc_stats, compute_mel_spectrogram, harmonic_percussive_ratio
from ingestion.loader import load_and_prepare
from ingestion.preprocessing import normalize_loudness
from models.trainer import load_model
from reporting.plots import mel_spectrogram_fig
from reporting.report_builder import build_report

settings = get_settings()
app = FastAPI(title="Audio Authenticity & Genre Service", version="0.1.0")

# Attempt to load pretrained models if present
GENRE_MODEL_PATH = Path("models/artifacts/genre_classifier.joblib")
AUTH_MODEL_PATH = Path("models/artifacts/auth_classifier.joblib")
genre_model = load_model(GENRE_MODEL_PATH) if GENRE_MODEL_PATH.exists() else None
auth_model = load_model(AUTH_MODEL_PATH) if AUTH_MODEL_PATH.exists() else None


@app.get("/healthz")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)) -> Dict[str, object]:
    if file.content_type and not file.content_type.startswith("audio"):
        raise HTTPException(status_code=400, detail="Unsupported file type")

    with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp:
        content = await file.read()
        tmp.write(content)
        temp_path = Path(tmp.name)

    audio = load_and_prepare(
        temp_path,
        sample_rate=settings.audio.sample_rate,
        mono=settings.audio.mono,
        target_duration_sec=settings.audio.target_duration_sec,
    )
    audio = normalize_loudness(audio, target_lufs=settings.audio.normalize_lufs)

    basic = extract_basic_features(audio)
    mfcc = compute_mfcc_stats(
        audio,
        n_mfcc=settings.features.n_mfcc,
        n_fft=settings.features.n_fft,
        hop_length=settings.features.hop_length,
    )
    hpr = harmonic_percussive_ratio(audio)
    features = {**basic, **mfcc, **hpr}

    try:
        embedding = extract_embedding(audio, model_name=settings.genre_model.embedding_model)
        features.update({f"embed_{i}": float(v) for i, v in enumerate(embedding)})
    except Exception as exc:
        # Keep service responsive even if heavy models are absent
        embedding = None
        features["embedding_error"] = str(exc)

    genre_result: Optional[Dict[str, float]] = None
    if genre_model:
        probs = genre_model.predict_proba(np.array([embedding])) if embedding is not None else None
        if probs is not None:
            top_k = genre_model.predict_top_k(np.array([embedding]), k=settings.genre_model.top_k)[0]
            genre_result = {label: score for label, score in top_k}

    authenticity_score: Optional[float] = None
    if auth_model and embedding is not None:
        authenticity_score = float(auth_model.predict_proba(np.array([embedding]))[0])

    mel = compute_mel_spectrogram(
        audio,
        n_fft=settings.features.n_fft,
        hop_length=settings.features.hop_length,
        n_mels=settings.features.n_mels,
        fmin=settings.features.fmin,
        fmax=settings.features.fmax,
    )
    mel_fig = mel_spectrogram_fig(mel_db=mel, title="Mel-Spectrogram")

    report_path = Path(settings.reporting.output_dir) / f"{temp_path.stem}.html"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    top_genre_label = next(iter(genre_result.keys())) if genre_result else "unknown"
    top_genre_conf = genre_result[top_genre_label] if genre_result else 0.0
    build_report(
        sample_name=file.filename,
        top_genre=top_genre_label,
        top_genre_conf=top_genre_conf,
        authenticity_score=authenticity_score or 0.0,
        metrics={k: round(v, 3) for k, v in basic.items()},
        figures={"mel_spectrogram": mel_fig},
        html_path=report_path,
        pdf=settings.reporting.include_pdf,
    )

    return {
        "filename": file.filename,
        "genre": genre_result,
        "authenticity_score": authenticity_score,
        "features": {k: float(v) for k, v in basic.items()},
        "report_path": str(report_path),
    }
