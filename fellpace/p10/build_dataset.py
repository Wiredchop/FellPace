"""Build one long-format CSV with yearly-best 5k/10k performances from DB/po10.csv."""

from __future__ import annotations

from pathlib import Path
import re

import pandas as pd

from .scraper import scrape_athlete_yearly_best


INPUT_CSV = Path("DB/po10.csv")
OUTPUT_CSV = Path("DB/po10_best_times.csv")


def _extract_athlete_url(url: str) -> str | None:
    match = re.search(r"/Home/Athlete/([a-f0-9\-]+)", str(url), flags=re.IGNORECASE)
    if not match:
        return None
    return f"https://www.powerof10.uk/Home/Athlete/{match.group(1)}"


def build_best_times_dataset(input_csv: Path = INPUT_CSV, output_csv: Path = OUTPUT_CSV, min_year: int = 2016) -> pd.DataFrame:
    po10_df = pd.read_csv(input_csv)
    po10_df.columns = [column.strip().lower() for column in po10_df.columns]

    records: list[dict] = []

    for row in po10_df.itertuples(index=False):
        racer = str(row.racer).strip()
        athlete_url = _extract_athlete_url(row.url)
        if athlete_url is None:
            continue

        try:
            data = scrape_athlete_yearly_best(athlete_url, min_year=min_year)
        except Exception:
            continue

        for best in data.get("yearly_best", []):
            records.append(
                {
                    "racer": racer,
                    "athlete_name": data.get("athlete_name", ""),
                    "athlete_url": athlete_url,
                    "distance": best.get("distance", ""),
                    "year": int(best.get("year")),
                    "best_time": best.get("best_time", ""),
                    "best_time_seconds": float(best.get("best_time_seconds")),
                    "best_date": best.get("best_date", ""),
                    "best_venue": best.get("best_venue", ""),
                }
            )

    dataset = pd.DataFrame.from_records(records)
    if dataset.empty:
        dataset.to_csv(output_csv, index=False, encoding="utf-8")
        return dataset

    dataset = dataset.sort_values(["racer", "distance", "year", "best_time_seconds"], ascending=[True, True, True, True])
    dataset = dataset.drop_duplicates(subset=["racer", "distance", "year"], keep="first").reset_index(drop=True)
    dataset.to_csv(output_csv, index=False, encoding="utf-8")
    return dataset


def main() -> None:
    dataset = build_best_times_dataset()
    print(f"Saved: {OUTPUT_CSV}")
    print(f"Rows: {len(dataset)}")
    print(f"Unique racers: {dataset['racer'].nunique() if not dataset.empty else 0}")
    print(f"Year range: {dataset['year'].min()} - {dataset['year'].max()}" if not dataset.empty else "Year range: n/a")


if __name__ == "__main__":
    main()
