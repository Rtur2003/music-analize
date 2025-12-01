from __future__ import annotations

from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd

from models.authenticity_model import AuthenticityClassifier
from models.genre_model import GenreClassifier


def train_genre_classifier(X: pd.DataFrame, y: Sequence[str], top_k: int = 5) -> GenreClassifier:
    model = GenreClassifier(top_k=top_k)
    model.fit(X.values, y)
    return model


def train_authenticity_classifier(
    X: pd.DataFrame, y: Sequence[int], threshold: float = 0.5, base_model: str = "lightgbm"
) -> AuthenticityClassifier:
    model = AuthenticityClassifier(threshold=threshold, base_model=base_model)
    model.fit(X.values, y)
    return model


def save_model(model, path: Path | str) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)


def load_model(path: Path | str):
    return joblib.load(Path(path))
