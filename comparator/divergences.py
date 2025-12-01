from __future__ import annotations

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy, wasserstein_distance


def kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-9) -> float:
    p_safe = np.clip(p, epsilon, 1.0)
    q_safe = np.clip(q, epsilon, 1.0)
    p_safe /= p_safe.sum()
    q_safe /= q_safe.sum()
    return float(entropy(p_safe, q_safe))


def js_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-9) -> float:
    p_safe = np.clip(p, epsilon, 1.0)
    q_safe = np.clip(q, epsilon, 1.0)
    p_safe /= p_safe.sum()
    q_safe /= q_safe.sum()
    return float(jensenshannon(p_safe, q_safe) ** 2)


def wasserstein(p: np.ndarray, q: np.ndarray) -> float:
    return float(wasserstein_distance(p, q))
