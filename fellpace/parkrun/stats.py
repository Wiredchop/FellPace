"Get some simple statistics from the database."
import pandas as pd
import numpy as np
def parkrun_mean_std(con, season: int = -1) -> tuple:
    """
    Get the mean and standard deviation of parkrun times for a given racer and season.
    
    Args:
        con (Connection): SQLite connection object.
        racer_id (int): ID of the racer.
        season (int): Season to filter results by. Default is -1 (all seasons).
        
    Returns:
        tuple: Mean and standard deviation of parkrun times.
    """
    
    def ln(t):
        return np.log(t)
    con.create_function("ln", 1, ln)
    
    
    season_filter = "" if season == -1 else "AND Season = ?"
    query = f""" -- sql
    WITH Racers_Results AS
    (
        SELECT Racer_ID, ZScore_log, Race_ID, ln(Time) as Time
        FROM Results
        WHERE Time IS NOT NULL
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
    )
    
        SELECT AVG(R.Time) as Mean, STDDEV(R.Time) as StdDev
        FROM Racers_Results R
        LEFT JOIN Races_Rename as C
        ON C.Race_ID = R.Race_ID
        WHERE Time IS NOT NULL
        AND C.Race_Name = "PR_Endcliffe"
        {season_filter}
        GROUP BY C.Race_Name
    """
    query_params = None if season == -1 else (season,)
    PR_stats = pd.read_sql(query, con, params=query_params)
    return PR_stats