from __future__ import annotations

import logging

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy, wasserstein_distance

from utils.constants import EPSILON

logger = logging.getLogger(__name__)


def kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = EPSILON) -> float:
    """
    Compute Kullback-Leibler divergence between two distributions.
    
    Args:
        p: First probability distribution
        q: Second probability distribution
        epsilon: Small value to avoid division by zero
        
    Returns:
        KL divergence value
    """
    try:
        p_safe = np.clip(p, epsilon, 1.0)
        q_safe = np.clip(q, epsilon, 1.0)
        p_safe /= p_safe.sum()
        q_safe /= q_safe.sum()
        result = float(entropy(p_safe, q_safe))
        
        if not np.isfinite(result):
            logger.warning("KL divergence resulted in non-finite value, returning 0")
            return 0.0
        
        return result
    except Exception as e:
        logger.error(f"Error computing KL divergence: {e}")
        return 0.0


def js_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = EPSILON) -> float:
    """
    Compute Jensen-Shannon divergence between two distributions.
    
    Args:
        p: First probability distribution
        q: Second probability distribution
        epsilon: Small value to avoid division by zero
        
    Returns:
        JS divergence value
    """
    try:
        p_safe = np.clip(p, epsilon, 1.0)
        q_safe = np.clip(q, epsilon, 1.0)
        p_safe /= p_safe.sum()
        q_safe /= q_safe.sum()
        result = float(jensenshannon(p_safe, q_safe) ** 2)
        
        if not np.isfinite(result):
            logger.warning("JS divergence resulted in non-finite value, returning 0")
            return 0.0
        
        return result
    except Exception as e:
        logger.error(f"Error computing JS divergence: {e}")
        return 0.0


def wasserstein(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute Wasserstein distance between two distributions.
    
    Args:
        p: First distribution
        q: Second distribution
        
    Returns:
        Wasserstein distance
    """
    try:
        result = float(wasserstein_distance(p, q))
        
        if not np.isfinite(result):
            logger.warning("Wasserstein distance resulted in non-finite value, returning 0")
            return 0.0
        
        return result
    except Exception as e:
        logger.error(f"Error computing Wasserstein distance: {e}")
        return 0.0
