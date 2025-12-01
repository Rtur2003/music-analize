from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import typer
from rich import print

from config.settings import get_settings
from features.base_features import extract_basic_features
from features.embeddings import extract_embedding
from features.spectral import compute_mfcc_stats, compute_mel_spectrogram, harmonic_percussive_ratio
from ingestion.loader import load_and_prepare
from ingestion.preprocessing import normalize_loudness
from models.trainer import load_model
from reporting.plots import mel_spectrogram_fig
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

    basic = extract_basic_features(audio)
    mfcc = compute_mfcc_stats(
        audio,
        n_mfcc=settings.features.n_mfcc,
        n_fft=settings.features.n_fft,
        hop_length=settings.features.hop_length,
    )
    hpr = harmonic_percussive_ratio(audio)
    features = {**basic, **mfcc, **hpr}

    embedding = None
    try:
        embedding = extract_embedding(audio, model_name=model_name)
        features.update({f"embed_{i}": float(v) for i, v in enumerate(embedding)})
    except Exception as exc:
        print(f"[yellow]Warning:[/yellow] embedding extraction failed: {exc}")

    genre_result: Optional[dict] = None
    if genre_model and embedding is not None:
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

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{path.stem}.html"
    top_label = next(iter(genre_result.keys())) if genre_result else "unknown"
    top_conf = genre_result[top_label] if genre_result else 0.0

    build_report(
        sample_name=path.name,
        top_genre=top_label,
        top_genre_conf=top_conf,
        authenticity_score=authenticity_score or 0.0,
        metrics={k: round(v, 3) for k, v in basic.items()},
        figures={"mel_spectrogram": mel_fig},
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
