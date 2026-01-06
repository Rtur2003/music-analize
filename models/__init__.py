"""Model definitions for genre and authenticity detection."""

from .authenticity_model import AuthenticityClassifier
from .genre_model import GenreClassifier
from .trainer import (
    load_model,
    save_model,
    train_authenticity_classifier,
    train_genre_classifier,
)

__all__ = [
    "AuthenticityClassifier",
    "GenreClassifier",
    "load_model",
    "save_model",
    "train_authenticity_classifier",
    "train_genre_classifier",
]
