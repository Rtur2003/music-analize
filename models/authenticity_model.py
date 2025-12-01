from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _build_classifier(base_model: str):
    if base_model == "lightgbm":
        try:
            from lightgbm import LGBMClassifier  # type: ignore
        except Exception:  # pragma: no cover - optional dependency
            base_model = "logreg"
        else:
            return LGBMClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=-1,
                num_leaves=64,
                subsample=0.8,
                colsample_bytree=0.8,
            )
    if base_model == "logreg" or base_model == "logistic":
        return LogisticRegression(max_iter=500, n_jobs=4)
    raise ValueError(f"Unsupported base model: {base_model}")


class AuthenticityClassifier:
    """
    Binary classifier: 1 = AI-generated, 0 = human/original.
    """

    def __init__(self, threshold: float = 0.5, base_model: str = "lightgbm"):
        self.threshold = threshold
        self.base_model_name = base_model
        self.model = _build_classifier(base_model)
        self.pipeline: Pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True)),
                ("clf", self.model),
            ]
        )

    def fit(self, X: np.ndarray, y: Sequence[int]) -> "AuthenticityClassifier":
        self.pipeline.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict_proba(X)[:, 1]

    def predict_label(self, X: np.ndarray, threshold: float | None = None) -> np.ndarray:
        thr = threshold if threshold is not None else self.threshold
        scores = self.predict_proba(X)
        return (scores >= thr).astype(int)
