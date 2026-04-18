"""Scrape yearly-best 5k/10k performances from Power of 10 athlete pages."""

from __future__ import annotations

from typing import Any
import re

import pandas as pd
import requests
from bs4 import BeautifulSoup


def _extract_year(date_text: str) -> int | None:
    if not isinstance(date_text, str) or not date_text.strip():
        return None
    parsed = pd.to_datetime(date_text, dayfirst=True, errors="coerce")
    if pd.isna(parsed):
        return None
    return int(parsed.year)


def _centiseconds_to_time(centiseconds: int) -> str:
    total_seconds = int(centiseconds / 100)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    if hours > 0:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


def _distance_for_event(event_name: str) -> str | None:
    event_lower = event_name.lower()
    if "parkrun" in event_lower or "5k" in event_lower or "5000" in event_lower or "5 k" in event_lower:
        return "5km"
    if "10k" in event_lower or "10000" in event_lower or "10 k" in event_lower:
        return "10km"
    return None


def scrape_athlete_yearly_best(athlete_url: str, min_year: int = 2016) -> dict[str, Any]:
    """Return yearly best 5k/10k performances for a Power of 10 athlete URL."""
    response = requests.get(athlete_url, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, "html.parser")

    athlete_name = "Unknown"
    for tag in soup.find_all(["h2", "h3", "strong"]):
        text = tag.get_text(strip=True)
        if text and len(text) > 3 and not any(x in text.lower() for x in ["club", "lead", "coach", "event", "performance"]):
            athlete_name = text
            break

    performances: list[dict[str, Any]] = []
    scripts = soup.find_all("script")

    for script in scripts:
        if not script.string:
            continue

        script_text = script.string
        event_indices = re.findall(r"var dataEventName(\d+) = ", script_text)

        for idx in event_indices:
            event_match = re.search(f"var dataEventName{idx} = '([^']+)'", script_text)
            if not event_match:
                continue

            event_name = event_match.group(1)
            distance = _distance_for_event(event_name)
            if distance not in {"5km", "10km"}:
                continue

            values_match = re.search(f"var dataRpValues{idx} = \\[([\\d,\\s]+)\\]", script_text)
            if not values_match:
                continue

            values = [int(value.strip()) for value in values_match.group(1).split(",") if value.strip()]

            dates_match = re.search(f"var dataRpMeetDates{idx} = \\[(.*?)\\];", script_text, re.DOTALL)
            venues_match = re.search(f"var dataRpLocations{idx} = \\[(.*?)\\];", script_text, re.DOTALL)

            dates = re.findall(r"'([^']*)'", dates_match.group(1)) if dates_match else []
            venues = re.findall(r"'([^']*)'", venues_match.group(1)) if venues_match else []

            for i, centiseconds in enumerate(values):
                date_value = dates[i] if i < len(dates) else ""
                year = _extract_year(date_value)
                if year is None or year < min_year:
                    continue

                performances.append(
                    {
                        "distance": distance,
                        "year": year,
                        "time": _centiseconds_to_time(centiseconds),
                        "time_seconds": float(centiseconds) / 100,
                        "date": date_value,
                        "venue": venues[i] if i < len(venues) else "",
                    }
                )

    if not performances:
        return {
            "athlete_name": athlete_name,
            "athlete_url": athlete_url,
            "yearly_best": [],
        }

    perf_df = pd.DataFrame(performances)
    perf_df = perf_df.sort_values(["distance", "year", "time_seconds", "date"], ascending=[True, True, True, True])
    yearly_best = perf_df.groupby(["distance", "year"], as_index=False).first()

    return {
        "athlete_name": athlete_name,
        "athlete_url": athlete_url,
        "yearly_best": yearly_best.rename(
            columns={
                "time": "best_time",
                "time_seconds": "best_time_seconds",
                "date": "best_date",
                "venue": "best_venue",
            }
        ).to_dict(orient="records"),
    }
