"""Persistence helpers for manual race include/exclude overrides.

Overrides are stored in DB/race_include_overrides.json as a flat dict keyed by a
composite string of the form:

    "{racer_id}|{season}|{race_name_lower}"

where racer_id is an integer Racer_ID when available, otherwise a normalised
lowercase Racer_Name, season is an integer year, and race_name_lower is the
Race_Name lowercased and stripped. A True value forces inclusion; False forces
exclusion, overriding the automatic filter logic in filter_race_results.
"""

import json
import math
from pathlib import Path

from loguru import logger

from fellpace.config import DB_DIR

RACE_OVERRIDE_PATH = DB_DIR / "race_include_overrides.json"


def _is_missing(val) -> bool:
    """Return True if val is None or a floating-point NaN."""
    if val is None:
        return True
    try:
        return math.isnan(float(val))
    except (TypeError, ValueError):
        return False


def _load_overrides(path: Path = None) -> dict:
    """Load the override map from disk. Returns an empty dict on any failure."""
    if path is None:
        path = RACE_OVERRIDE_PATH
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception as exc:
        logger.warning(f"Could not load race override map at {path}: {exc}")
    return {}


def _save_overrides(overrides: dict, path: Path = None) -> None:
    """Persist the override map to disk."""
    if path is None:
        path = RACE_OVERRIDE_PATH
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(overrides, f, indent=2, sort_keys=True)
    except Exception as exc:
        logger.warning(f"Could not save race override map at {path}: {exc}")


def build_override_key(row) -> str | None:
    """Build a composite override key from a DataFrame row or dict-like object.

    Key format: "{racer_id_or_name}|{season}|{race_name_lower}"

    Returns None if any required field (Race_Name, Season, and at least one of
    Racer_ID / Racer_Name) is missing or null, so callers can safely skip rows
    that cannot be keyed.

    Args:
        row: A pandas Series, a plain dict, or any object that supports .get().

    Returns:
        The composite key string, or None if required fields are absent.
    """

    def _get(key):
        val = row.get(key) if hasattr(row, "get") else None
        return None if _is_missing(val) else val

    race_name = _get("Race_Name")
    season = _get("Season")
    racer_id = _get("Racer_ID")
    racer_name = _get("Racer_Name")

    if race_name is None or season is None:
        return None

    if racer_id is not None:
        identity = str(int(racer_id))
    elif racer_name is not None:
        identity = str(racer_name).lower().strip()
    else:
        return None

    return f"{identity}|{int(season)}|{str(race_name).lower().strip()}"


def set_race_override(racer_id_or_name, season: int, race_name: str, include: bool) -> None:
    """Set a manual include/exclude override for a specific race result.

    Args:
        racer_id_or_name: Integer Racer_ID or racer name string (normalised
            internally).
        season: Season year as an integer.
        race_name: Race name string (normalised internally).
        include: True to force the result to be included in the prediction;
            False to force exclusion.
    """
    if not _is_missing(racer_id_or_name) and isinstance(racer_id_or_name, (int, float)):
        identity = str(int(racer_id_or_name))
    else:
        identity = str(racer_id_or_name).lower().strip()

    key = f"{identity}|{int(season)}|{str(race_name).lower().strip()}"
    overrides = _load_overrides()
    overrides[key] = bool(include)
    _save_overrides(overrides)
    logger.info(f"Set race override: {key} -> {include}")
