from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class GenreClassifier:
    """
    Simple genre classifier that operates on precomputed embeddings or feature vectors.
    """

    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self.pipeline: Pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True)),
                ("clf", LogisticRegression(max_iter=500, n_jobs=4, multi_class="multinomial")),
            ]
        )

    def fit(self, X: np.ndarray, y: Sequence[str]) -> "GenreClassifier":
        self.pipeline.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict_proba(X)

    def predict_top_k(self, X: np.ndarray, k: int | None = None) -> List[List[Tuple[str, float]]]:
        k = k or self.top_k
        probs = self.predict_proba(X)
        classes = list(self.pipeline.named_steps["clf"].classes_)
        top_indices = np.argsort(probs, axis=1)[:, ::-1][:, :k]
        results: List[List[Tuple[str, float]]] = []
        for i, row in enumerate(top_indices):
            results.append([(classes[idx], float(probs[i, idx])) for idx in row])
        return results
