from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import typer
from rich import print as rprint

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
from utils.exceptions import AudioLoadError, FeatureExtractionError, ModelError
from utils.logging_config import setup_logging

setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)

app = typer.Typer(help="CLI for audio genre + authenticity analysis.")


@app.command()
def analyze(
    path: Path = typer.Argument(..., exists=True, readable=True, help="Audio file path"),
    model_dir: Path = typer.Option(Path("models/artifacts"), help="Directory with trained models"),
    output_dir: Path = typer.Option(Path("reports"), help="Directory to store reports"),
    model_name: str = typer.Option("wav2vec2_base", help="Embedding model name"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
):
    """
    Analyze an audio file for genre and authenticity.
    
    Args:
        path: Path to audio file
        model_dir: Directory containing trained models
        output_dir: Directory to save reports
        model_name: Name of embedding model to use
        verbose: Enable verbose logging
    """
    if verbose:
        setup_logging(level=logging.DEBUG)
    
    try:
        logger.info(f"Starting analysis of {path}")
        settings = get_settings()
        
        genre_path = model_dir / "genre_classifier.joblib"
        auth_path = model_dir / "auth_classifier.joblib"
        
        genre_model = None
        auth_model = None
        
        if genre_path.exists():
            try:
                genre_model = load_model(genre_path)
                logger.info("Genre model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load genre model: {e}")
        else:
            logger.warning(f"Genre model not found at {genre_path}")
        
        if auth_path.exists():
            try:
                auth_model = load_model(auth_path)
                logger.info("Authenticity model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load authenticity model: {e}")
        else:
            logger.warning(f"Authenticity model not found at {auth_path}")
        
        audio = load_and_prepare(
            path,
            sample_rate=settings.audio.sample_rate,
            mono=settings.audio.mono,
            target_duration_sec=settings.audio.target_duration_sec,
        )
        audio = normalize_loudness(audio, target_lufs=settings.audio.normalize_lufs)
        
        features, embedding, mel, centroid_series = extract_all(
            audio,
            settings=settings,
            embed_model_name=model_name,
        )
        
        genre_result: Optional[dict] = None
        if genre_model and embedding is not None:
            try:
                top_k = genre_model.predict_top_k(
                    np.array([embedding]),
                    k=settings.genre_model.top_k
                )[0]
                genre_result = {label: score for label, score in top_k}
                logger.info(f"Genre predictions computed: {genre_result}")
            except Exception as e:
                logger.error(f"Genre prediction failed: {e}")
        
        authenticity_score: Optional[float] = None
        if auth_model and embedding is not None:
            try:
                authenticity_score = float(auth_model.predict_proba(np.array([embedding]))[0])
                logger.info(f"Authenticity score computed: {authenticity_score:.3f}")
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
        
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / f"{path.stem}.html"
        top_label = next(iter(genre_result.keys())) if genre_result else "unknown"
        top_conf = genre_result[top_label] if genre_result else 0.0
        
        build_report(
            sample_name=path.name,
            top_genre=top_label,
            top_genre_conf=top_conf,
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
        
        rprint("[green]✓ Analysis complete[/green]")
        rprint(f"[blue]Report:[/blue] {report_path}")
        if genre_result:
            rprint(f"[blue]Genre:[/blue] {genre_result}")
        if authenticity_score is not None:
            rprint(f"[blue]Authenticity score:[/blue] {authenticity_score:.3f}")
        
        logger.info("Analysis completed successfully")
        
    except AudioLoadError as e:
        logger.error(f"Audio loading failed: {e}")
        rprint(f"[red]Error:[/red] Failed to load audio file: {e}")
        raise typer.Exit(code=1)
    except FeatureExtractionError as e:
        logger.error(f"Feature extraction failed: {e}")
        rprint(f"[red]Error:[/red] Feature extraction failed: {e}")
        raise typer.Exit(code=1)
    except ModelError as e:
        logger.error(f"Model operation failed: {e}")
        rprint(f"[red]Error:[/red] Model operation failed: {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        rprint(f"[red]Error:[/red] Unexpected error: {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
