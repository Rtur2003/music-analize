from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile

from .schemas import AnalysisResponse, GenreScore, HealthResponse
from config.settings import get_settings
from features.extractor import extract_all
from ingestion.loader import load_and_prepare
from ingestion.preprocessing import normalize_loudness
from models.trainer import load_model
from reporting.plots import (
    chroma_bar,
    envelope_fig,
    feature_bar,
    mel_spectrogram_fig,
    spectral_centroid_fig,
    waveform_fig,
)
from reporting.report_builder import build_report
from utils.constants import MAX_FILE_SIZE, SUPPORTED_AUDIO_EXTENSIONS
from utils.exceptions import AudioLoadError, FeatureExtractionError, ModelError
from utils.logging_config import setup_logging

setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()
app = FastAPI(title="Audio Authenticity & Genre Service", version="0.1.0")

GENRE_MODEL_PATH = Path("models/artifacts/genre_classifier.joblib")
AUTH_MODEL_PATH = Path("models/artifacts/auth_classifier.joblib")
genre_model = load_model(GENRE_MODEL_PATH) if GENRE_MODEL_PATH.exists() else None
auth_model = load_model(AUTH_MODEL_PATH) if AUTH_MODEL_PATH.exists() else None

if not genre_model:
    logger.warning("Genre model not found, genre prediction will be unavailable")
if not auth_model:
    logger.warning("Authenticity model not found, authenticity scoring will be unavailable")


@app.get("/healthz", response_model=HealthResponse)
async def health() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(file: UploadFile = File(...)) -> Dict[str, object]:
    """
    Analyze an audio file for genre and authenticity.
    
    Args:
        file: Uploaded audio file
        
    Returns:
        Analysis results including genre predictions and authenticity score
        
    Raises:
        HTTPException: If analysis fails
    """
    temp_path = None
    
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in SUPPORTED_AUDIO_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format: {file_ext}. "
                       f"Supported formats: {', '.join(SUPPORTED_AUDIO_EXTENSIONS)}"
            )
        
        logger.info(f"Processing file: {file.filename}")
        
        content = await file.read()
        
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File size exceeds maximum allowed size of {MAX_FILE_SIZE} bytes"
            )
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            tmp.write(content)
            temp_path = Path(tmp.name)
        
        audio = load_and_prepare(
            temp_path,
            sample_rate=settings.audio.sample_rate,
            mono=settings.audio.mono,
            target_duration_sec=settings.audio.target_duration_sec,
        )
        audio = normalize_loudness(audio, target_lufs=settings.audio.normalize_lufs)
        
        features, embedding, mel, centroid_series = extract_all(
            audio,
            settings=settings,
            embed_model_name=settings.genre_model.embedding_model,
        )
        
        genre_result: Optional[Dict[str, float]] = None
        if genre_model and embedding is not None:
            try:
                top_k = genre_model.predict_top_k(
                    np.array([embedding]),
                    k=settings.genre_model.top_k
                )[0]
                genre_result = {label: score for label, score in top_k}
                logger.debug(f"Genre predictions: {genre_result}")
            except Exception as e:
                logger.error(f"Genre prediction failed: {e}")
        
        authenticity_score: Optional[float] = None
        if auth_model and embedding is not None:
            try:
                authenticity_score = float(auth_model.predict_proba(np.array([embedding]))[0])
                logger.debug(f"Authenticity score: {authenticity_score}")
            except Exception as e:
                logger.error(f"Authenticity prediction failed: {e}")
        
        mel_fig = mel_spectrogram_fig(mel_db=mel, title="Mel-Spectrogram")
        wave_fig = waveform_fig(audio.waveform, audio.sample_rate, title="Waveform")
        env_fig = envelope_fig(audio.waveform, audio.sample_rate, title="Energy Envelope")
        cent_fig = spectral_centroid_fig(
            centroid_series,
            sr=audio.sample_rate,
            hop_length=settings.features.hop_length,
            title="Spectral Centroid",
        )
        chroma_mean = np.array([v for k, v in features.items() if k.startswith("chroma_mean_")])
        chroma_fig = chroma_bar(chroma_mean, title="Chroma Energy")
        metric_bar = feature_bar(
            {k: v for k, v in features.items() 
             if k in ("lufs", "rms", "flatness", "harmonic_percussive_ratio", "crest_factor")},
            title="Core Features",
        )
        
        report_path = Path(settings.reporting.output_dir) / f"{Path(file.filename).stem}.html"
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
            figures={
                "mel_spectrogram": mel_fig,
                "waveform": wave_fig,
                "energy_envelope": env_fig,
                "spectral_centroid": cent_fig,
                "chroma": chroma_fig,
                "core_features": metric_bar,
            },
            html_path=report_path,
            pdf=settings.reporting.include_pdf,
        )
        
        genre_response = (
            [GenreScore(label=label, confidence=score) for label, score in genre_result.items()]
            if genre_result
            else None
        )
        
        logger.info(f"Analysis completed successfully for {file.filename}")
        
        return {
            "filename": file.filename,
            "genre": genre_response,
            "authenticity_score": authenticity_score,
            "features": {k: float(v) for k, v in features.items()},
            "report_path": str(report_path),
            "message": "Processed without embedding" if embedding is None else None,
        }
        
    except AudioLoadError as e:
        logger.error(f"Audio loading error: {e}")
        raise HTTPException(status_code=400, detail=f"Audio loading failed: {str(e)}")
    except FeatureExtractionError as e:
        logger.error(f"Feature extraction error: {e}")
        raise HTTPException(status_code=500, detail=f"Feature extraction failed: {str(e)}")
    except ModelError as e:
        logger.error(f"Model error: {e}")
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error during analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    finally:
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
                logger.debug(f"Cleaned up temporary file: {temp_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file {temp_path}: {e}")
