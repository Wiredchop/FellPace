"""Persistence helpers for manual predicted-time overrides.

Overrides are stored in DB/predicted_time_overrides.json as a flat dict keyed by:

    "{racer_name_lower}|{year}"

Value is predicted time in seconds as a float.
"""

import json
from pathlib import Path

from loguru import logger

from fellpace.config import DB_DIR

PREDICTION_TIME_OVERRIDE_PATH = DB_DIR / "predicted_time_overrides.json"


def _normalise_name(name: str) -> str:
    return str(name).lower().strip()


def _build_key(racer_name: str, year: int) -> str:
    return f"{_normalise_name(racer_name)}|{int(year)}"


def load_prediction_time_overrides(path: Path = None) -> dict:
    """Load predicted-time override map from disk."""
    if path is None:
        path = PREDICTION_TIME_OVERRIDE_PATH
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception as exc:
        logger.warning(f"Could not load predicted-time overrides at {path}: {exc}")
    return {}


def _save_prediction_time_overrides(overrides: dict, path: Path = None) -> None:
    """Persist predicted-time override map to disk."""
    if path is None:
        path = PREDICTION_TIME_OVERRIDE_PATH
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(overrides, f, indent=2, sort_keys=True)
    except Exception as exc:
        logger.warning(f"Could not save predicted-time overrides at {path}: {exc}")


def set_prediction_time_override(racer_name: str, year: int, predicted_seconds: float) -> None:
    """Set or replace a manual predicted-time override for one racer/year."""
    key = _build_key(racer_name, year)
    overrides = load_prediction_time_overrides()
    overrides[key] = float(predicted_seconds)
    _save_prediction_time_overrides(overrides)
    logger.info(f"Set predicted-time override: {key} -> {float(predicted_seconds):.2f}s")


def clear_prediction_time_override(racer_name: str, year: int) -> None:
    """Remove a manual predicted-time override if present."""
    key = _build_key(racer_name, year)
    overrides = load_prediction_time_overrides()
    if key in overrides:
        overrides.pop(key)
        _save_prediction_time_overrides(overrides)
        logger.info(f"Cleared predicted-time override: {key}")
