import pandas as pd
import Levenshtein
from loguru import logger


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
    racer_id = find_racer_ID(con, name = racer_name)
    if racer_id is None:
        logger.warning(f"Racer {racer_name} not found in database.")
        logger.info("Looking for similar names...")
        names = find_similar_name(con, name = racer_name)
        if names.empty:
            logger.info("No similar names found.")
            return
        for i, row in names.iterrows():
            logger.info(f"{i}: {row['Racer_Name']}")
        logger.info(f"{i+1}: None of these are correct")
        selected_index = int(input("Select the number of the name you want to use: "))
        if selected_index == i+1:
            logger.info("No name selected, exiting.")
            return
        racer_id = names.iloc[selected_index]['Racer_ID']
    return racer_id
