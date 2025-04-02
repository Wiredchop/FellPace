import pandas as pd


def extract_all_zscore_data(con):
    sql_extract_zscore = '''WITH Chase_Yrs AS
    (
        SELECT Racer_ID, 
               CAST(strftime("%Y",CD.Chase_Date) AS INTEGER) - 1 as Season,
               Time,
               ZScore_log
        FROM Results_Chase
        JOIN Chases as CD
        ON CD.Chase_ID = Results_Chase.Chase_ID
    ),
    Racers_Results AS
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
        END AS Race_Name,
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
        SELECT Racer_ID, C.Race_Name, C.Season, XPercentile(ZScore_log) as ZScore, C.Race_ID
        FROM Racers_Results
        LEFT JOIN Races_Rename as C
        ON C.Race_ID = Racers_Results.Race_ID
        WHERE Time IS NOT NULL
        GROUP BY Racer_ID, C.Race_Name, C.Season
    )
    SELECT R.Racer_ID,R.Race_Name, R.Season, R.ZScore, HC.Time as HCTime, HC.ZScore_log as HCScore
    FROM Results_joined as R
    LEFT JOIN Chase_Yrs as HC
    ON HC.Racer_ID = R.Racer_ID
    AND HC.Season = R.Season
    WHERE R.Racer_ID IN (SELECT Racer_ID FROM Results_Chase)
    AND HCScore IS NOT NULL'''

    data_Zs = pd.read_sql(sql_extract_zscore, con)
    data_Zs.sort_values('Race_Name', inplace=True)
    return data_Zs