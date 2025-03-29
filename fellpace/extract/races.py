import sqlite3
import pandas as pd

def get_race_series_summary(con: sqlite3.Connection):
    query = """
    SELECT Series_Name, MIN(R.Race_Date) AS first_race, MAX(R.Race_Date) AS latest_race, COUNT(DISTINCT R.Race_ID) AS number_of_races, COUNT(DISTINCT R2.Racer_ID) AS number_of_runners
    FROM Race_Series RS
    JOIN Races R
    ON RS.Series_ID = R.Series_ID
    JOIN Results R2
    ON R.Race_ID = R2.Race_ID
    GROUP BY Series_Name
    """
    return pd.read_sql(query, con)

def get_chase_summary(con: sqlite3.Connection):
    query = """
    SELECT Chase_Date AS Date_of_Chase, COUNT(DISTINCT ChaseR_ID) AS Number_of_Runners
    FROM Chases C
    JOIN Results_Chase RC
    ON C.Chase_ID = RC.Chase_ID
    GROUP BY Chase_Date
    ORDER BY Chase_Date
    """
    return pd.read_sql(query, con)