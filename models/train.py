from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

from config.settings import get_settings
from models.trainer import (
    load_model,
    save_model,
    train_authenticity_classifier,
    train_genre_classifier,
)


def load_feature_store(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def train_models(
    feature_path: Path,
    output_dir: Path = Path("models/artifacts"),
    settings_path: Path | str = "config/settings.yaml",
) -> Tuple[object, object]:
    settings = get_settings(settings_path)
    df = load_feature_store(feature_path)

    if "genre" not in df.columns or "is_ai" not in df.columns:
        raise ValueError("Feature store must include 'genre' and 'is_ai' columns.")

    # Genre model on embeddings only if present, else all numeric features
    embed_cols = [c for c in df.columns if c.startswith("embed_")]
    if not embed_cols:
        embed_cols = df.select_dtypes(include="number").columns.tolist()
        embed_cols = [c for c in embed_cols if c not in ("is_ai",)]
    X_genre = df[embed_cols].values
    y_genre = df["genre"].values

    # Authenticity model uses same feature subset
    X_auth = X_genre
    y_auth = df["is_ai"].values

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    (train_idx, val_idx) = next(sss.split(X_genre, y_genre))

    genre_clf = train_genre_classifier(X_genre[train_idx], y_genre[train_idx], top_k=settings.genre_model.top_k)
    auth_clf = train_authenticity_classifier(
        X_auth[train_idx],
        y_auth[train_idx],
        threshold=settings.authenticity_model.threshold,
        base_model=settings.authenticity_model.base_model,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    save_model(genre_clf, output_dir / "genre_classifier.joblib")
    save_model(auth_clf, output_dir / "auth_classifier.joblib")

    # Quick validation metrics
    genre_val_acc = (genre_clf.pipeline.predict(X_genre[val_idx]) == y_genre[val_idx]).mean()
    auth_val = auth_clf.predict_proba(X_auth[val_idx])
    auth_val_acc = ((auth_val >= settings.authenticity_model.threshold).astype(int) == y_auth[val_idx]).mean()

    print(f"Genre val acc: {genre_val_acc:.3f}")
    print(f"Auth val acc: {auth_val_acc:.3f}")

    return genre_clf, auth_clf


if __name__ == "__main__":
    train_models(Path("data/processed/features.parquet"))
