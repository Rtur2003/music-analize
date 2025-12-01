from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Tuple


def validate_structure(root: Path) -> Dict[str, Dict[str, int]]:
    """
    Validate dataset tree data/raw/<genre>/{ai,real}/*.wav
    Returns counts per genre/label.
    """
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"ai": 0, "real": 0})
    for genre_dir in root.iterdir():
        if not genre_dir.is_dir():
            continue
        for label in ("ai", "real"):
            label_dir = genre_dir / label
            if not label_dir.exists():
                continue
            counts[genre_dir.name][label] += len(list(label_dir.glob("*.wav")))
    return counts


def print_summary(counts: Dict[str, Dict[str, int]]) -> None:
    total_ai = sum(v["ai"] for v in counts.values())
    total_real = sum(v["real"] for v in counts.values())
    print(f"Total genres: {len(counts)} | AI: {total_ai} | Real: {total_real}")
    for genre, vals in counts.items():
        print(f"{genre:<15} AI: {vals['ai']:<5} Real: {vals['real']:<5}")


if __name__ == "__main__":
    root = Path("data/raw")
    if not root.exists():
        raise SystemExit("data/raw not found")
    summary = validate_structure(root)
    print_summary(summary)
