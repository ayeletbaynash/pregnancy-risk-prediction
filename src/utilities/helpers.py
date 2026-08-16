"""Shared helper utilities for the pregnancy-risk project."""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def ensure_dirs(*paths):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def save_json(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def load_source_splits(split_dir: Path):
    out = {}
    for source in ["sga", "gdm", "pe", "meir"]:
        for split in ["train", "test"]:
            p = split_dir / f"{source}_{split}.csv"
            if not p.exists():
                raise FileNotFoundError(f"Missing {p}. Run src/02_create_targets.py first.")
            out[f"{source}_{split}"] = pd.read_csv(p)
    return out


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


__all__ = [
    "ensure_dirs",
    "save_json",
    "load_source_splits",
    "set_seed",
]
