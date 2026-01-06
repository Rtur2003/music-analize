"""Reporting and visualization utilities."""

from .plots import (
    calibration_plot,
    chroma_bar,
    confusion_heatmap,
    divergence_barplot,
    envelope_fig,
    feature_bar,
    feature_histogram,
    mel_spectrogram_fig,
    spectral_centroid_fig,
    waveform_fig,
)
from .report_builder import build_report

__all__ = [
    "build_report",
    "calibration_plot",
    "chroma_bar",
    "confusion_heatmap",
    "divergence_barplot",
    "envelope_fig",
    "feature_bar",
    "feature_histogram",
    "mel_spectrogram_fig",
    "spectral_centroid_fig",
    "waveform_fig",
]
