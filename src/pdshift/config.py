"""Configuration loading: a base YAML file plus an optional experiment override."""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(base_path: str | Path, override_path: str | Path | None = None) -> dict[str, Any]:
    """Load the base config and apply an experiment override if given.

    Args:
        base_path: Path to ``configs/base.yaml``.
        override_path: Optional path to an experiment YAML.

    Returns:
        The merged configuration dictionary.

    Raises:
        FileNotFoundError: If a config file does not exist.
        ValueError: If a config file is empty or not a mapping.
    """
    cfg = _read_yaml(base_path)
    if override_path is not None:
        cfg = _deep_merge(cfg, _read_yaml(override_path))
        logger.info("Applied override %s", override_path)
    return cfg


def _read_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must contain a YAML mapping")
    return data


def load_class_map(path: str | Path) -> list[dict[str, str]]:
    """Load the shared PlantVillage/PlantDoc label space.

    Returns:
        A list of ``{"name", "plantvillage", "plantdoc"}`` entries; list order defines label ids.
    """
    data = _read_yaml(path)
    classes = data.get("classes")
    if not classes:
        raise ValueError(f"No 'classes' entry in {path}")
    names = [c["name"] for c in classes]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate class names in class map")
    return classes
