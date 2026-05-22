"""Combine Power of 10 best road times with Hallam Chase results for model training."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from fellpace.config import DB_DIR, DB_PATH
from fellpace.db.db_setup import setup_db
from fellpace.convert_tools import seconds_to_time_string
from fellpace.extract.racers import find_racer_ID, find_similar_name


PO10_CSV = DB_DIR / "po10_best_times.csv"
OUTPUT_CSV = DB_DIR / "po10_chase_combined.csv"


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _all_racers_in_db(con) -> pd.DataFrame:
    return pd.read_sql("SELECT Racer_ID, Racer_Name FROM Racers", con)


def _match_name_to_db(con, po10_name: str, db_racers: pd.DataFrame) -> tuple[int | None, str | None]:
    """
    Match a PO10 name to DB racer logic:
    1) exact lookup via find_racer_ID
    2) fallback to find_similar_name (closest by existing distance metric)

    Returns (racer_id, matched_db_name) or (None, None).
    """
    normalized_name = po10_name.lower().strip()

    # First, exact match (preferred)
    racer_id = find_racer_ID(con, normalized_name)
    if racer_id is not None:
        matched = db_racers.loc[db_racers["Racer_ID"] == int(racer_id), "Racer_Name"]
        matched_name = str(matched.iloc[0]) if not matched.empty else po10_name
        return int(racer_id), matched_name

    # Fallback to existing similar-name logic
    similar = find_similar_name(con, normalized_name)
    if similar.empty:
        logger.warning(f"Name mismatch: '{po10_name}' has no exact match and no similar-name candidates.")
        return None, None

    # pick closest suggestion deterministically
    similar = similar.sort_values(["distance", "Racer_ID"], ascending=[True, True]).reset_index(drop=True)
    best_distance = int(similar.loc[0, "distance"])
    best_name = str(similar.loc[0, "Racer_Name"])
    best_id = int(similar.loc[0, "Racer_ID"])

    # Only auto-accept very close matches; otherwise skip and log.
    if best_distance <= 1:
        logger.warning(
            f"Name mismatch resolved by fallback: '{po10_name}' -> '{best_name}' "
            f"(Racer_ID={best_id}, distance={best_distance})"
        )
        return best_id, best_name

    logger.warning(
        f"Name mismatch unresolved: '{po10_name}' best candidate '{best_name}' "
        f"(Racer_ID={best_id}, distance={best_distance}) exceeds auto-match threshold. Skipping."
    )
    return None, None


def _batch_chase_results(con, racer_ids: list[int]) -> pd.DataFrame:
    """Return all Hallam Chase results for a list of racer IDs."""
    placeholders = ",".join("?" * len(racer_ids))
    sql = f"""
        SELECT
            C.Racer_ID,
            C.Time        AS chase_time_seconds,
            CAST(strftime('%Y', CH.Chase_Date) AS INTEGER) AS year
        FROM Results_Chase AS C
        JOIN Chases AS CH ON C.Chase_ID = CH.Chase_ID
        WHERE C.Racer_ID IN ({placeholders})
          AND C.Time IS NOT NULL
        ORDER BY C.Racer_ID, year
    """
    return pd.read_sql(sql, con, params=racer_ids)


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_combined_dataset(
    po10_csv: Path = PO10_CSV,
    output_csv: Path = OUTPUT_CSV,
) -> pd.DataFrame:
    """
    Combine Power of 10 yearly-best road times with Hallam Chase results.

    For each racer in po10_best_times.csv:
    - Looks up their chase history in the fellpace DB.
    - Produces one row per racer-year with columns for:
        racer, year, chase_time_seconds,
        5km_best_seconds, 10km_best_seconds
    - Only includes years where a chase result exists.

    Returns:
        pd.DataFrame: Combined dataset, also saved to output_csv.
    """
    po10_csv = Path(po10_csv)
    output_csv = Path(output_csv)

    po10 = pd.read_csv(po10_csv)
    po10["year"] = pd.to_numeric(po10["year"], errors="coerce")
    po10 = po10.dropna(subset=["year", "best_time_seconds"])

    con = setup_db(DB_PATH)
    db_racers = _all_racers_in_db(con)

    # --- resolve racer names to DB IDs ---
    name_map: dict[str, tuple[int, str]] = {}  # po10_name -> (racer_id, db_name)
    for po10_name in po10["racer"].unique():
        racer_id, db_name = _match_name_to_db(con, po10_name, db_racers)
        if racer_id is None:
            logger.warning(f"No DB match found for '{po10_name}' — skipping.")
        else:
            if db_name.lower() != po10_name.lower():
                logger.info(f"Name resolved: '{po10_name}' → '{db_name}' (id={racer_id})")
            name_map[po10_name] = (racer_id, db_name)

    if not name_map:
        logger.error("No racers could be matched to the database.")
        con.close()
        return pd.DataFrame()

    # --- batch fetch chase results ---
    racer_ids = [racer_id for racer_id, _ in name_map.values()]
    chase_df = _batch_chase_results(con, racer_ids)
    con.close()

    # Add po10_name column so we can merge on racer label
    id_to_po10_name = {racer_id: po10_name for po10_name, (racer_id, _) in name_map.items()}
    chase_df["racer"] = chase_df["Racer_ID"].map(id_to_po10_name)
    chase_df["year"] = pd.to_numeric(chase_df["year"], errors="coerce")

    # --- pivot po10 road times to wide: one col per distance ---
    road_wide = (
        po10[["racer", "year", "distance", "best_time_seconds"]]
        .pivot_table(index=["racer", "year"], columns="distance", values="best_time_seconds", aggfunc="min")
        .reset_index()
    )
    road_wide.columns.name = None
    rename_map = {
        "p10_5k": "5km_best_seconds",
        "p10_10k": "10km_best_seconds",
    }
    road_wide = road_wide.rename(columns=rename_map)

    # --- merge chase results with road times ---
    combined = chase_df[["racer", "year", "chase_time_seconds"]].merge(
        road_wide, on=["racer", "year"], how="left"
    )

    # Add human-readable time columns
    combined["chase_time"] = combined["chase_time_seconds"].apply(
        lambda s: seconds_to_time_string(int(s)) if pd.notna(s) else ""
    )
    if "5km_best_seconds" in combined.columns:
        combined["5km_best_time"] = combined["5km_best_seconds"].apply(
            lambda s: seconds_to_time_string(int(s)) if pd.notna(s) else ""
        )
    if "10km_best_seconds" in combined.columns:
        combined["10km_best_time"] = combined["10km_best_seconds"].apply(
            lambda s: seconds_to_time_string(int(s)) if pd.notna(s) else ""
        )

    combined = combined.sort_values(["racer", "year"]).reset_index(drop=True)

    # Reorder columns sensibly
    col_order = (
        ["racer", "year", "chase_time_seconds", "chase_time"]
        + (["5km_best_seconds", "5km_best_time"] if "5km_best_seconds" in combined.columns else [])
        + (["10km_best_seconds", "10km_best_time"] if "10km_best_seconds" in combined.columns else [])
    )
    combined = combined[[c for c in col_order if c in combined.columns]]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_csv, index=False, encoding="utf-8")

    logger.info(f"Saved combined dataset: {output_csv}")
    logger.info(f"Rows: {len(combined)} | Unique racers: {combined['racer'].nunique()}")

    return combined


def extract_road_chase_training_data(input_csv: Path = OUTPUT_CSV) -> pd.DataFrame:
    """Reshape the combined PO10/Chase CSV into the training format expected by
    train_time_models().

    Each row pairs a road best time (5k or 10k) with the Chase time from the
    same year for the same racer. Rows with missing road or Chase times are
    dropped before returning.

    The resulting Race_Name values are:
        'p10_5k'   — 5 km road best time predicting Chase time
        'p10_10k'  — 10 km road best time predicting Chase time

    Args:
        input_csv: Path to the combined CSV produced by build_combined_dataset().
                   Defaults to OUTPUT_CSV (DB/po10_chase_combined.csv).

    Returns:
        pd.DataFrame with columns:
            Race_Name  — 'p10_5k' or 'p10_10k'
            PrevTime   — road best time in seconds
            HCTime     — Chase finishing time in seconds
    """
    input_csv = Path(input_csv)

    if not input_csv.exists():
        logger.warning(f"Road/chase combined CSV missing at {input_csv}. Attempting to build it now.")
        try:
            combined_built = build_combined_dataset(output_csv=input_csv)
        except FileNotFoundError as exc:
            logger.warning(
                f"Could not build road/chase dataset because a source file is missing: {exc}. "
                "Skipping road time models."
            )
            return pd.DataFrame(columns=["Race_Name", "PrevTime", "HCTime"])

        if combined_built.empty:
            logger.warning("Built road/chase dataset is empty. Skipping road time models.")
            return pd.DataFrame(columns=["Race_Name", "PrevTime", "HCTime"])

        combined = combined_built
    else:
        combined = pd.read_csv(input_csv)

    frames = []
    for dist, col in [("p10_5k", "5km_best_seconds"), ("p10_10k", "10km_best_seconds")]:
        if col not in combined.columns:
            continue
        sub = combined[[col, "chase_time_seconds"]].dropna()
        sub = sub.rename(columns={col: "PrevTime", "chase_time_seconds": "HCTime"})
        sub["Race_Name"] = dist
        frames.append(sub[["Race_Name", "PrevTime", "HCTime"]])

    if not frames:
        logger.warning("No road time columns found in combined CSV.")
        return pd.DataFrame(columns=["Race_Name", "PrevTime", "HCTime"])

    return pd.concat(frames, ignore_index=True)


def main() -> None:
    combined = build_combined_dataset()
    if combined.empty:
        return

    print(f"Saved: {OUTPUT_CSV}")
    print(f"Rows: {len(combined)}")
    print(f"Unique racers: {combined['racer'].nunique()}")
    print(f"Year range: {combined['year'].min()} - {combined['year'].max()}")
    print("\nSample:")
    print(combined.head(10).to_string(index=False))

    mask_5k_missing = (
        combined["5km_best_seconds"].isna()
        if "5km_best_seconds" in combined.columns
        else pd.Series(True, index=combined.index)
    )
    mask_10k_missing = (
        combined["10km_best_seconds"].isna()
        if "10km_best_seconds" in combined.columns
        else pd.Series(True, index=combined.index)
    )
    missing_road = combined[mask_5k_missing & mask_10k_missing]
    if len(missing_road) > 0:
        print(f"\nWarning: {len(missing_road)} chase rows have no road time for that year "
              f"(road data exists from 2016+).")


if __name__ == "__main__":
    main()
