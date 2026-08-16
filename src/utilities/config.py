"""Configuration loading and project path helpers."""
from __future__ import annotations
import os
from pathlib import Path
import yaml


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path.resolve())
    cfg["_project_root"] = str(path.resolve().parent.parent)
    return cfg


def project_root(cfg: dict) -> Path:
    return Path(cfg["_project_root"])


def resolve_path(cfg: dict, value: str) -> Path:
    value = os.path.expandvars(str(value))
    p = Path(value).expanduser()
    return p if p.is_absolute() else project_root(cfg) / p


def data_dirs(cfg: dict) -> dict[str, Path]:
    return {k: resolve_path(cfg, v) for k, v in cfg["data"].items()}
