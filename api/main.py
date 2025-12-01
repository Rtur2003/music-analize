from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from config.settings import get_settings
from features.extractor import extract_all
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

    try:
        features, embedding, mel = extract_all(
            audio,
            settings=settings,
            embed_model_name=settings.genre_model.embedding_model,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Feature extraction failed: {exc}")

    genre_result: Optional[Dict[str, float]] = None
    if genre_model:
        probs = genre_model.predict_proba(np.array([embedding])) if embedding is not None else None
        if probs is not None:
            top_k = genre_model.predict_top_k(np.array([embedding]), k=settings.genre_model.top_k)[0]
            genre_result = {label: score for label, score in top_k}

    authenticity_score: Optional[float] = None
    if auth_model and embedding is not None:
        authenticity_score = float(auth_model.predict_proba(np.array([embedding]))[0])

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
        metrics={
            k: round(v, 3)
            for k, v in features.items()
            if not k.startswith("mfcc") and not k.startswith("embed_") and k != "embedding_error"
        },
        figures={"mel_spectrogram": mel_fig},
        html_path=report_path,
        pdf=settings.reporting.include_pdf,
    )
    try:
        temp_path.unlink(missing_ok=True)
    except Exception:
        pass

    return {
        "filename": file.filename,
        "genre": genre_result,
        "authenticity_score": authenticity_score,
        "features": {k: float(v) for k, v in features.items()},
        "report_path": str(report_path),
    }
