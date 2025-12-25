from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd

from models.authenticity_model import AuthenticityClassifier
from models.genre_model import GenreClassifier
from utils.constants import DEFAULT_THRESHOLD, DEFAULT_TOP_K
from utils.exceptions import ModelError
from utils.validators import validate_model_path

logger = logging.getLogger(__name__)


def train_genre_classifier(X: pd.DataFrame, y: Sequence[str], top_k: int = DEFAULT_TOP_K) -> GenreClassifier:
    """
    Train a genre classification model.
    
    Args:
        X: Feature matrix
        y: Genre labels
        top_k: Number of top predictions to return
        
    Returns:
        Trained GenreClassifier
        
    Raises:
        ModelError: If training fails
    """
    try:
        logger.info(f"Training genre classifier with {len(X)} samples and {len(set(y))} classes")
        model = GenreClassifier(top_k=top_k)
        model.fit(X.values, y)
        logger.info("Genre classifier training completed")
        return model
    except Exception as e:
        logger.error(f"Failed to train genre classifier: {e}")
        raise ModelError(f"Failed to train genre classifier: {e}") from e


def train_authenticity_classifier(
    X: pd.DataFrame,
    y: Sequence[int],
    threshold: float = DEFAULT_THRESHOLD,
    base_model: str = "lightgbm",
) -> AuthenticityClassifier:
    """
    Train an authenticity classification model.
    
    Args:
        X: Feature matrix
        y: Binary labels (0=real, 1=AI)
        threshold: Decision threshold
        base_model: Base model type ("lightgbm" or "logreg")
        
    Returns:
        Trained AuthenticityClassifier
        
    Raises:
        ModelError: If training fails
    """
    try:
        logger.info(f"Training authenticity classifier with {len(X)} samples")
        model = AuthenticityClassifier(threshold=threshold, base_model=base_model)
        model.fit(X.values, y)
        logger.info("Authenticity classifier training completed")
        return model
    except Exception as e:
        logger.error(f"Failed to train authenticity classifier: {e}")
        raise ModelError(f"Failed to train authenticity classifier: {e}") from e


def save_model(model, path: Path | str) -> None:
    """
    Save a trained model to disk.
    
    Args:
        model: Model object to save
        path: Output file path
        
    Raises:
        ModelError: If saving fails
    """
    try:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving model to {out_path}")
        joblib.dump(model, out_path)
        logger.info("Model saved successfully")
    except Exception as e:
        logger.error(f"Failed to save model to {path}: {e}")
        raise ModelError(f"Failed to save model: {e}") from e


def load_model(path: Path | str):
    """
    Load a trained model from disk.
    
    Args:
        path: Path to model file
        
    Returns:
        Loaded model object
        
    Raises:
        ModelError: If loading fails
    """
    try:
        model_path = Path(path)
        validate_model_path(model_path)
        logger.info(f"Loading model from {model_path}")
        model = joblib.load(model_path)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load model from {path}: {e}")
        raise ModelError(f"Failed to load model: {e}") from e
