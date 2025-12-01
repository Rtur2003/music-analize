from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import typer
from rich import print

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

app = typer.Typer(help="CLI for audio genre + authenticity analysis.")


@app.command()
def analyze(
    path: Path = typer.Argument(..., exists=True, readable=True, help="Audio file path"),
    model_dir: Path = typer.Option(Path("models/artifacts"), help="Directory with trained models"),
    output_dir: Path = typer.Option(Path("reports"), help="Directory to store reports"),
    model_name: str = typer.Option("wav2vec2_base", help="Embedding model name"),
):
    settings = get_settings()
    genre_path = model_dir / "genre_classifier.joblib"
    auth_path = model_dir / "auth_classifier.joblib"
    genre_model = load_model(genre_path) if genre_path.exists() else None
    auth_model = load_model(auth_path) if auth_path.exists() else None

    audio = load_and_prepare(
        path,
        sample_rate=settings.audio.sample_rate,
        mono=settings.audio.mono,
        target_duration_sec=settings.audio.target_duration_sec,
    )
    audio = normalize_loudness(audio, target_lufs=settings.audio.normalize_lufs)

    try:
        features, embedding, mel, centroid_series = extract_all(audio, settings=settings, embed_model_name=model_name)
    except Exception as exc:
        raise typer.Exit(code=1, message=f"Feature extraction failed: {exc}")

    genre_result: Optional[dict] = None
    if genre_model and embedding is not None:
        top_k = genre_model.predict_top_k(np.array([embedding]), k=settings.genre_model.top_k)[0]
        genre_result = {label: score for label, score in top_k}

    authenticity_score: Optional[float] = None
    if auth_model and embedding is not None:
        authenticity_score = float(auth_model.predict_proba(np.array([embedding]))[0])

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
        {k: v for k, v in features.items() if k in ("lufs", "rms", "flatness", "harmonic_percussive_ratio", "crest_factor")},
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

    print("[green]Analysis complete[/green]")
    print(f"Report: {report_path}")
    if genre_result:
        print(f"Genre: {genre_result}")
    if authenticity_score is not None:
        print(f"Authenticity score: {authenticity_score:.3f}")


if __name__ == "__main__":
    app()
