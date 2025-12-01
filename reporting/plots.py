from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import librosa


def mel_spectrogram_fig(mel_db: np.ndarray, title: str = "Mel-Spectrogram") -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=mel_db,
            colorscale="Magma",
            colorbar=dict(title="dB"),
        )
    )
    fig.update_layout(title=title, xaxis_title="Frames", yaxis_title="Mel bins")
    return fig


def waveform_fig(y: np.ndarray, sr: int, title: str = "Waveform") -> go.Figure:
    times = np.linspace(0, len(y) / sr, num=len(y))
    fig = go.Figure(data=go.Scatter(x=times, y=y, mode="lines", line=dict(color="#22d3ee")))
    fig.update_layout(title=title, xaxis_title="Time (s)", yaxis_title="Amplitude")
    return fig


def envelope_fig(y: np.ndarray, sr: int, frame_ms: int = 50, title: str = "Energy Envelope") -> go.Figure:
    hop = int(sr * frame_ms / 1000)
    frames = [np.sqrt(np.mean(np.square(y[i : i + hop]))) for i in range(0, len(y), hop)]
    times = np.arange(len(frames)) * (hop / sr)
    fig = go.Figure(data=go.Scatter(x=times, y=frames, mode="lines", line=dict(color="#38bdf8")))
    fig.update_layout(title=title, xaxis_title="Time (s)", yaxis_title="RMS")
    return fig


def spectral_centroid_fig(centroid: np.ndarray, sr: int, hop_length: int, title: str = "Spectral Centroid") -> go.Figure:
    times = librosa.times_like(centroid, sr=sr, hop_length=hop_length)
    fig = go.Figure(data=go.Scatter(x=times, y=centroid.flatten(), mode="lines", line=dict(color="#a855f7")))
    fig.update_layout(title=title, xaxis_title="Time (s)", yaxis_title="Hz")
    return fig


def chroma_bar(chroma_mean: np.ndarray, title: str = "Chroma Energy") -> go.Figure:
    labels = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    fig = go.Figure(data=go.Bar(x=labels, y=chroma_mean, marker_color="#fbbf24"))
    fig.update_layout(title=title, xaxis_title="Pitch Class", yaxis_title="Mean Energy")
    return fig


def feature_bar(features: Dict[str, float], title: str = "Feature Summary") -> go.Figure:
    items = list(features.items())
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    fig = go.Figure(data=go.Bar(x=labels, y=values, marker_color="#22d3ee"))
    fig.update_layout(title=title, xaxis_tickangle=-45)
    return fig


def feature_histogram(
    df: pd.DataFrame,
    feature: str,
    label_col: str = "label",
    title: Optional[str] = None,
) -> px.histogram:
    ttl = title or f"Distribution of {feature}"
    fig = px.histogram(df, x=feature, color=label_col, nbins=40, opacity=0.6, barmode="overlay", title=ttl)
    fig.update_layout(xaxis_title=feature, yaxis_title="Count")
    return fig


def divergence_barplot(divergence_df: pd.DataFrame, metric: str = "js") -> px.bar:
    fig = px.bar(
        divergence_df,
        x="feature",
        y=metric,
        color="genre",
        title=f"{metric.upper()} divergence per feature and genre",
    )
    fig.update_layout(xaxis_tickangle=-45)
    return fig


def confusion_heatmap(cm: np.ndarray, labels: Iterable[str], title: str = "Confusion Matrix") -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=list(labels),
            y=list(labels),
            colorscale="Blues",
            colorbar=dict(title="Count"),
        )
    )
    fig.update_layout(title=title, xaxis_title="Predicted", yaxis_title="True")
    return fig


def calibration_plot(probs: Iterable[float], labels: Iterable[int]) -> px.line:
    df = pd.DataFrame({"prob": list(probs), "label": list(labels)})
    df["bin"] = pd.cut(df["prob"], bins=np.linspace(0, 1, 11), include_lowest=True)
    calib = df.groupby("bin").agg(mean_prob=("prob", "mean"), frac_pos=("label", "mean")).reset_index()
    fig = px.line(calib, x="mean_prob", y="frac_pos", title="Calibration curve")
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash"))
    fig.update_layout(xaxis_title="Predicted probability", yaxis_title="Observed frequency")
    return fig
