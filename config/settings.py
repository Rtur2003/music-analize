from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class AudioConfig:
    sample_rate: int = 44100
    mono: bool = True
    target_duration_sec: int = 30
    normalize_lufs: float = -23.0


@dataclass
class FeatureConfig:
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    n_mfcc: int = 20
    fmin: int = 20
    fmax: int = 20000


@dataclass
class GenreModelConfig:
    embedding_model: str = "musicnn"
    top_k: int = 5


@dataclass
class AuthenticityModelConfig:
    base_model: str = "lightgbm"
    threshold: float = 0.5


@dataclass
class ReportingConfig:
    output_dir: str = "reports"
    include_pdf: bool = True
    include_html: bool = True


@dataclass
class StorageConfig:
    feature_store: str = "data/processed/features.parquet"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"


@dataclass
class Settings:
    audio: AudioConfig = AudioConfig()
    features: FeatureConfig = FeatureConfig()
    genre_model: GenreModelConfig = GenreModelConfig()
    authenticity_model: AuthenticityModelConfig = AuthenticityModelConfig()
    reporting: ReportingConfig = ReportingConfig()
    storage: StorageConfig = StorageConfig()

    @classmethod
    def from_yaml(cls, path: Path | str = "config/settings.yaml") -> "Settings":
        settings_path = Path(path)
        if not settings_path.exists():
            return cls()
        with settings_path.open("r", encoding="utf-8") as f:
            data: Dict[str, Any] = yaml.safe_load(f) or {}
        return cls(
            audio=AudioConfig(**data.get("audio", {})),
            features=FeatureConfig(**data.get("features", {})),
            genre_model=GenreModelConfig(**data.get("models", {}).get("genre", {})),
            authenticity_model=AuthenticityModelConfig(
                **data.get("models", {}).get("authenticity", {})
            ),
            reporting=ReportingConfig(**data.get("reporting", {})),
            storage=StorageConfig(**data.get("storage", {})),
        )


def get_settings(path: Path | str = "config/settings.yaml") -> Settings:
    return Settings.from_yaml(path)
