from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder

from comparator.stats import compare_real_vs_ai
from models.trainer import load_model
from reporting.plots import calibration_plot, divergence_barplot
from reporting.report_builder import build_report


def evaluate_models(
    feature_path: Path,
    model_dir: Path = Path("models/artifacts"),
    output_dir: Path = Path("reports/eval"),
    threshold: float = 0.5,
) -> Dict[str, object]:
    df = pd.read_parquet(feature_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    genre_clf = load_model(model_dir / "genre_classifier.joblib")
    auth_clf = load_model(model_dir / "auth_classifier.joblib")
    le = None
    if (model_dir / "genre_label_encoder.joblib").exists():
        le = load_model(model_dir / "genre_label_encoder.joblib")
    else:
        le = LabelEncoder().fit(df["genre"])

    embed_cols = [c for c in df.columns if c.startswith("embed_")]
    if not embed_cols:
        embed_cols = df.select_dtypes(include="number").columns.tolist()
        embed_cols = [c for c in embed_cols if c not in ("is_ai",)]

    X = df[embed_cols].values
    y_genre = le.transform(df["genre"])
    y_auth = df["is_ai"].values

    # Genre metrics
    genre_preds = genre_clf.pipeline.predict(X)
    genre_acc = accuracy_score(y_genre, genre_preds)
    genre_report = classification_report(y_genre, genre_preds, target_names=le.classes_, output_dict=True)
    cm = confusion_matrix(y_genre, genre_preds)

    # Authenticity metrics
    auth_scores = auth_clf.predict_proba(X)
    auth_preds = (auth_scores >= threshold).astype(int)
    auth_auroc = roc_auc_score(y_auth, auth_scores)
    auth_acc = accuracy_score(y_auth, auth_preds)
    prec, rec, _ = precision_recall_curve(y_auth, auth_scores)
    pr_auc = np.trapz(rec, prec)

    # Divergence real vs AI per genre
    divergence_df = compare_real_vs_ai(df, genre_col="genre", label_col="is_ai", feature_cols=embed_cols)

    # Plots
    div_plot = divergence_barplot(divergence_df, metric="js") if not divergence_df.empty else None
    calib_fig = calibration_plot(auth_scores, y_auth)

    # Save summary HTML
    figures = {}
    if div_plot:
        figures["divergence_js"] = div_plot
    figures["calibration"] = calib_fig

    html_path = output_dir / "evaluation.html"
    build_report(
        sample_name="Dataset Evaluation",
        top_genre="N/A",
        top_genre_conf=0.0,
        authenticity_score=float(auth_auroc),
        metrics={
            "genre_accuracy": f"{genre_acc:.3f}",
            "auth_auroc": f"{auth_auroc:.3f}",
            "auth_accuracy": f"{auth_acc:.3f}",
            "pr_auc": f"{pr_auc:.3f}",
        },
        figures=figures,
        html_path=html_path,
        pdf=False,
    )

    # Persist raw metrics
    summary = {
        "genre_accuracy": genre_acc,
        "genre_report": genre_report,
        "confusion_matrix": cm.tolist(),
        "auth_auroc": auth_auroc,
        "auth_accuracy": auth_acc,
        "pr_auc": pr_auc,
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not divergence_df.empty:
        divergence_df.to_csv(output_dir / "divergence.csv", index=False)

    return summary


if __name__ == "__main__":
    evaluate_models(Path("data/processed/features.parquet"))
