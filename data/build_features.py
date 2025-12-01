from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import duckdb
import pandas as pd
from tqdm import tqdm

from config.settings import get_settings
from features.extractor import extract_all
from ingestion.loader import load_and_prepare
from ingestion.preprocessing import normalize_loudness


def iter_audio_files(root: Path, exts: List[str] | None = None):
    exts = exts or [".wav", ".flac", ".mp3", ".ogg"]
    for ext in exts:
        for path in root.rglob(f"*{ext}"):
            yield path


def build_feature_store(
    data_root: Path,
    output_parquet: Path,
    settings_path: Path | str = "config/settings.yaml",
    embed_model_name: str | None = None,
) -> pd.DataFrame:
    settings = get_settings(settings_path)
    rows: List[Dict] = []
    for path in tqdm(list(iter_audio_files(data_root)), desc="Extracting features"):
        # Expecting path like data/raw/<genre>/<ai|real>/file.wav
        parts = path.relative_to(data_root).parts
        if len(parts) < 3:
            print(f"Skipping {path}, expected data/raw/<genre>/<ai|real>/file")
            continue
        genre, label_dir = parts[0], parts[1]
        is_ai = 1 if label_dir.lower() == "ai" else 0
        audio = load_and_prepare(
            path,
            sample_rate=settings.audio.sample_rate,
            mono=settings.audio.mono,
            target_duration_sec=settings.audio.target_duration_sec,
        )
        audio = normalize_loudness(audio, target_lufs=settings.audio.normalize_lufs)
        features, embedding, _ = extract_all(audio, settings=settings, embed_model_name=embed_model_name)
        row = {"path": str(path), "genre": genre, "is_ai": is_ai, **features}
        rows.append(row)

    df = pd.DataFrame(rows)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_parquet, index=False)

    # Optional: mirror into duckdb for queries
    duck_path = output_parquet.with_suffix(".duckdb")
    conn = duckdb.connect(str(duck_path))
    conn.execute("CREATE TABLE IF NOT EXISTS features AS SELECT * FROM df")
    conn.close()
    return df


if __name__ == "__main__":
    base = Path("data/raw")
    out = Path("data/processed/features.parquet")
    build_feature_store(base, out, embed_model_name=get_settings().genre_model.embedding_model)
