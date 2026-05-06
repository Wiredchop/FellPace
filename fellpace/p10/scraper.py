"""Scrape yearly-best 5k/10k performances from Power of 10 athlete pages."""

from __future__ import annotations

from typing import Any
import re
import Levenshtein
import loguru as logger
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
        return "p10_5k"
    if "10k" in event_lower or "10000" in event_lower or "10 k" in event_lower:
        return "p10_10k"
    return None


def _normalize_person_name(name: str) -> str:
    parts = [part for part in str(name).strip().split() if part]
    if not parts:
        return "Unknown"
    normalized = []
    for part in parts:
        if part.isupper() and len(part) > 1:
            normalized.append(part.title())
        else:
            normalized.append(part)
    return " ".join(normalized)


def _extract_athlete_name(soup: BeautifulSoup) -> str:
    # Athlete pages usually show name immediately above "CLUB".
    page_text = [line.strip() for line in soup.get_text("\n", strip=True).splitlines()]
    invalid_labels = {
        "age group",
        "athletes",
        "performance",
        "performances",
        "event rankings",
        "show road running",
        "show bio",
        "bio",
        "sex",
        "men",
        "women",
        "county",
        "club",
        "lead coach",
        "coach",
        "unattached",
    }

    for idx, line in enumerate(page_text):
        if line.upper().startswith("CLUB") and idx > 0:
            # Gather contiguous name-like tokens above CLUB. Some pages split
            # first and surname across separate lines.
            tokens: list[str] = []
            back_idx = idx - 1
            while back_idx >= 0 and len(tokens) < 4:
                token = page_text[back_idx].strip()
                token_lower = token.lower()
                if not token:
                    back_idx -= 1
                    continue
                if token_lower in invalid_labels:
                    break
                if any(char.isdigit() for char in token):
                    break
                if re.match(r"^[A-Za-z][A-Za-z'\-\.]*$", token):
                    tokens.append(token)
                    back_idx -= 1
                    continue
                # Also allow a full name on one line.
                if re.match(r"^[A-Za-z][A-Za-z'\-\.\s]+[A-Za-z]$", token):
                    return _normalize_person_name(token)
                break

            if len(tokens) >= 2:
                return _normalize_person_name(" ".join(reversed(tokens)))

    # Last fallback: parse page title metadata if CLUB-adjacent extraction fails.
    title_candidates: list[str] = []
    og_title = soup.find("meta", attrs={"property": "og:title"})
    if og_title and og_title.get("content"):
        title_candidates.append(str(og_title.get("content")))
    if soup.title and soup.title.string:
        title_candidates.append(str(soup.title.string))

    for title in title_candidates:
        match = re.search(
            r"athlete[^\-:|]*[-:|]\s*([A-Za-z][A-Za-z'\-\.\s]+)",
            title,
            flags=re.IGNORECASE,
        )
        if match:
            candidate = _normalize_person_name(match.group(1))
            if candidate.lower() not in invalid_labels:
                return candidate

    return "Unknown"


def scrape_athlete_yearly_best(
    athlete_url: str,
    min_year: int = 2016,
    max_year: int | None = None,
) -> dict[str, Any]:
    """Return yearly best 5k/10k performances for a Power of 10 athlete URL.

    Args:
        athlete_url: Power of 10 athlete page URL.
        min_year: Lower inclusive bound for performances.
        max_year: Upper inclusive bound for performances. Ignored when None.
    """
    response = requests.get(athlete_url, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, "html.parser")

    athlete_name = _extract_athlete_name(soup)

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
            if distance not in {"p10_5k", "p10_10k"}:
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
                if max_year is not None and year > max_year:
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
    
def get_racer_p10_results_for_prediction(racer_id: int, racer_name: str, athlete_url: str, prediction_year: int = None) -> pd.DataFrame | None:
    """Get a racer's Power of 10 yearly best results for use in prediction.

    Args:
        athlete_url: Power of 10 athlete page URL.
        prediction_year: If provided, only include seasons before this year.
    """
    if prediction_year is None:
        max_year = pd.Timestamp.now().year
    else:
        max_year = prediction_year
        
    
    data = scrape_athlete_yearly_best(athlete_url, max_year=max_year)
    
    scraped_name = data.get("athlete_name", "Unknown")
    
    name_distance = Levenshtein.distance(scraped_name.lower(), racer_name.lower())
    if name_distance > 2:
        logger.warning(f"Scraped name '{scraped_name}' is too different from racer name '{racer_name}' (distance {name_distance}). Skipping P10 results.")
        return None
    
    if len(data.get("yearly_best", [])) == 0:
        return None
    results = (
        pd.DataFrame(data.get("yearly_best"))
        .rename(
            columns={
            "distance": "Race_Name",
            "best_time_seconds": "Time",
            "year": "Season",
            }
        )
        .assign(
            Racer_ID=racer_id,
            Zpred_mu = None
    )
    )[["Racer_ID", "Race_Name", "Time", "Zpred_mu", "Season"]]

    return results

if __name__ == "__main__":
    # Example usage:
    athlete_url = "https://www.powerof10.uk/Home/Athlete/1ee315ac-0a39-4d4a-82a8-beabecaf8cd9"
    result = scrape_athlete_yearly_best(athlete_url)
    print(result)