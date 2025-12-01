from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


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


def calibration_plot(probs: Iterable[float], labels: Iterable[int]) -> px.line:
    df = pd.DataFrame({"prob": list(probs), "label": list(labels)})
    df["bin"] = pd.cut(df["prob"], bins=np.linspace(0, 1, 11), include_lowest=True)
    calib = df.groupby("bin").agg(mean_prob=("prob", "mean"), frac_pos=("label", "mean")).reset_index()
    fig = px.line(calib, x="mean_prob", y="frac_pos", title="Calibration curve")
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash"))
    fig.update_layout(xaxis_title="Predicted probability", yaxis_title="Observed frequency")
    return fig
