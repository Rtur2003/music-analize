from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd

from comparator.divergences import js_divergence, kl_divergence, wasserstein


def summarize_by_group(df: pd.DataFrame, group_cols: List[str], feature_cols: Iterable[str]) -> pd.DataFrame:
    stats = df.groupby(group_cols)[list(feature_cols)].agg(["mean", "std", "median"])
    stats.columns = ["_".join(col).strip() for col in stats.columns.values]
    return stats.reset_index()


def compare_real_vs_ai(
    df: pd.DataFrame,
    genre_col: str,
    label_col: str,
    feature_cols: Iterable[str],
    ai_label: int = 1,
) -> pd.DataFrame:
    """
    Compute per-genre divergence metrics between real and AI samples.
    """
    rows = []
    for genre, group in df.groupby(genre_col):
        real = group[group[label_col] != ai_label]
        ai = group[group[label_col] == ai_label]
        if real.empty or ai.empty:
            continue
        for feat in feature_cols:
            r_vals = real[feat].dropna().values
            a_vals = ai[feat].dropna().values
            if len(r_vals) < 2 or len(a_vals) < 2:
                continue
            # Normalize histograms for divergence; fallback to Wasserstein for continuous.
            hist_range = (min(r_vals.min(), a_vals.min()), max(r_vals.max(), a_vals.max()))
            r_hist, bin_edges = np.histogram(r_vals, bins=30, range=hist_range, density=True)
            a_hist, _ = np.histogram(a_vals, bins=30, range=hist_range, density=True)
            rows.append(
                {
                    "genre": genre,
                    "feature": feat,
                    "kl": kl_divergence(r_hist, a_hist),
                    "js": js_divergence(r_hist, a_hist),
                    "wasserstein": wasserstein(r_vals, a_vals),
                    "real_mean": float(np.mean(r_vals)),
                    "ai_mean": float(np.mean(a_vals)),
                }
            )
    return pd.DataFrame(rows)
