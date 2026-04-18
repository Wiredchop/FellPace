import pandas as pd
import Levenshtein
import json
from pathlib import Path
from loguru import logger
from fellpace.config import DB_DIR


RACER_NAME_ALIAS_PATH = DB_DIR / "racer_name_aliases.json"


def _load_racer_name_aliases(path: Path = RACER_NAME_ALIAS_PATH) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception as exc:
        logger.warning(f"Could not load racer alias map at {path}: {exc}")
    return {}


def _save_racer_name_aliases(alias_map: dict, path: Path = RACER_NAME_ALIAS_PATH) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(alias_map, f, indent=2, sort_keys=True)
    except Exception as exc:
        logger.warning(f"Could not save racer alias map at {path}: {exc}")


def _resolve_alias_racer_id(con, requested_name: str) -> tuple:
    alias_map = _load_racer_name_aliases()
    mapped_name = alias_map.get(requested_name)
    if not mapped_name:
        return None, None

    mapped_id = find_racer_ID(con, name=mapped_name)
    if mapped_id is not None:
        logger.info(f"Using saved name alias: '{requested_name}' -> '{mapped_name}'")
        return mapped_id, mapped_name

    logger.warning(
        f"Saved alias for '{requested_name}' points to missing DB racer '{mapped_name}'."
    )
    return None, None


def _add_racer_alias(requested_name: str, resolved_name: str) -> None:
    alias_map = _load_racer_name_aliases()
    alias_map[requested_name] = resolved_name
    _save_racer_name_aliases(alias_map)
    logger.info(f"Saved name alias: '{requested_name}' -> '{resolved_name}'")


def find_racer_ID(con, name):
    racer_ID_query = """
    SELECT * FROM Racers
    WHERE lower(Racer_Name) = ?
    """
    name = name.lower()
    racer_match = pd.read_sql(racer_ID_query, con, params=(name,))
    if racer_match.empty:
        return None
    return racer_match["Racer_ID"].values[0]

def find_similar_name(con, name:str):
    #Get a list of the racers and racer_ids from the database
    assert name.lower() == name, "Lower case names only"
    
    Get_racer_query = """
    SELECT Racer_ID, Racer_Name
    FROM Racers
    """
    racers = pd.read_sql_query(Get_racer_query,con)
    racers['distance'] = racers.apply(lambda r: Levenshtein.distance(name, r["Racer_Name"].lower()) ,axis = 1)
    return racers[racers['distance'] <= 2].reset_index(drop=True)


def get_racers_results(con, racer_ID, season: int = -1) -> pd.DataFrame:
    season_filter = "" if season == -1 else "AND Season = ?"
    query = f"""
    WITH Racers_Results AS
    (
        SELECT Racer_ID, ZScore_log, Race_ID, Time
        FROM Results
    ),
    Races_Rename AS
    (
        SELECT CASE
        WHEN Race_Name LIKE "Parkrun_endcliffe%" THEN "PR_Endcliffe"
        WHEN Race_Name LIKE "Parkrun_hillsborough%" THEN "PR_Hillsborough"
        ELSE Race_Name
        END  AS Race_Name,
        CASE
            WHEN CAST(strftime("%m",Race_Date) AS INTEGER) > 5
            THEN CAST(strftime("%Y",Race_Date) AS INTEGER)
            ELSE CAST(strftime("%Y",Race_Date) AS INTEGER) -1
        END AS Season,
        Race_ID
        FROM Races
    ),
    Results_joined AS
    (
        SELECT R.Racer_ID, R.Racer_Name, C.Race_Name, C.Season, XPercentile(ZScore_log) as ZScore, C.Race_ID
        FROM Racers_Results
        LEFT JOIN Races_Rename as C
        ON C.Race_ID = Racers_Results.Race_ID
        JOIN Racers AS R
        ON R.Racer_ID = Racers_Results.Racer_ID
        WHERE Time IS NOT NULL
        GROUP BY R.Racer_ID, C.Race_Name, C.Season
    )
    SELECT R.Racer_ID,R.Racer_Name, R.Race_Name, R.Season, R.ZScore
    FROM Results_joined as R
    WHERE R.Racer_ID = ?
    {season_filter}
    """
    query_params = (str(racer_ID),) if season == -1 else (str(racer_ID), season)
    racer_results = pd.read_sql(query, con, params=query_params)
    return racer_results

if __name__ == "__main__":
    from fellpace.db.db_setup import setup_db
    from fellpace.config import DB_PATH
    con = setup_db(DB_PATH)
    print(get_racers_results(con, 353, 2022))
    con.close()


def secure_racer_id(con, racer_name: str):
    """Securely retrieve the racer ID from the database, prompting the user if necessary.
    
    If the name is not found, it will search for similar names and allow the user to select one.
    The function returns the original or updated name if necessary.

    Args:
        con (Connection): Database connection object.
        racer_name (str): Name of the racer.

    Returns:
        Tuple[int, str]: A tuple containing the racer ID and name.
    """

    requested_name = racer_name.lower().strip()
    racer_id = find_racer_ID(con, name=requested_name)
    resolved_name = requested_name

    if racer_id is None:
        racer_id, resolved_name = _resolve_alias_racer_id(con, requested_name)
        if racer_id is not None:
            return racer_id, resolved_name

        logger.warning(f"Racer {requested_name} not found in database.")
        logger.info("Looking for similar names...")
        names = find_similar_name(con, name=requested_name)
        if names.empty:
            logger.info("No similar names found.")
            return None, None
        i = -1
        for i, row in names.iterrows():
            logger.info(f"{i}: {row['Racer_Name']}")
        logger.info(f"{i+1}: None of these are correct")
        selected_index = int(input("Select the number of the name you want to use: "))
        if selected_index == i + 1:
            logger.info("No name selected, exiting.")
            return None, None

        if selected_index < 0 or selected_index >= len(names):
            logger.info("Invalid selection, exiting.")
            return None, None

        resolved_name = names.iloc[selected_index]['Racer_Name']
        racer_id = names.iloc[selected_index]['Racer_ID']

        # Persist alias for future lookups.
        _add_racer_alias(requested_name, resolved_name)

    return racer_id, resolved_name
