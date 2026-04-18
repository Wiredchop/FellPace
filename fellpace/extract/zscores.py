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


def extract_previous_year_chase_data(con):
    """
    Build a training dataset where a racer's Hallam Chase z-score from the
    immediately preceding year is used to predict their result in the current year.

    Only strict year-on-year pairs are included (Season - 1 → Season).
    If a racer did not run in the immediately preceding year, no row is produced
    for that target year — use extract_older_chase_data() to capture those cases.

    Examples (racer participation in seasons 1, 2, 3):
        Years 1, 2, 3  →  (1→2), (2→3)
        Years 1, 3     →  (no rows)      year 2 missing, strict pair required
        Year  1 only   →  (no rows)      nothing to predict from

    Race_Name is set to 'previous_chase' so the output can be concatenated with
    extract_all_zscore_data() for unified model training.

    Args:
        con: Active SQLite connection (configured via db_setup.setup_db).

    Returns:
        pd.DataFrame with columns:
            Racer_ID  — racer identifier
            Race_Name — always 'previous_chase'
            Season    — the year being predicted
            ZScore    — racer's z-score in the immediately preceding Chase
            HCTime    — racer's actual finishing time in the predicted Chase
            HCScore   — racer's actual z-score in the predicted Chase
    """
    sql = '''WITH Chase_Yrs AS
    (
        SELECT Racer_ID,
               CAST(strftime("%Y", CD.Chase_Date) AS INTEGER) - 1 AS Season,
               Time,
               ZScore_log
        FROM Results_Chase
        JOIN Chases AS CD
        ON CD.Chase_ID = Results_Chase.Chase_ID
    )
    SELECT
        curr.Racer_ID,
        "previous_chase" AS Race_Name,
        curr.Season        AS Season,
        prev.ZScore_log    AS ZScore,
        curr.Time          AS HCTime,
        curr.ZScore_log    AS HCScore
    FROM Chase_Yrs AS curr
    JOIN Chase_Yrs AS prev
        ON  prev.Racer_ID = curr.Racer_ID
        AND prev.Season   = curr.Season - 1
    ORDER BY curr.Racer_ID, curr.Season'''

    return pd.read_sql(sql, con)


def extract_older_chase_data(con):
    """
    Build a training dataset where the most recent available prior Hallam Chase
    z-score is used to predict the current year, but ONLY for years where the
    racer has no result in the immediately preceding year.

    This is the complement of extract_previous_year_chase_data(): it covers gaps
    in participation where a direct year-on-year comparison is not possible.
    The predictor is always the closest available prior result.

    Examples (racer participation in seasons 1, 2, 3):
        Years 1, 2, 3  →  (no rows)      year-on-year pairs exist, not needed here
        Years 1, 3     →  (1→3)          gap: year 2 missing, fall back to year 1
        Year  2 only   →  (no rows)      nothing prior to predict from

    Race_Name is set to 'older_chase' so the output can be distinguished from
    extract_previous_year_chase_data() when concatenated for model training.

    Args:
        con: Active SQLite connection (configured via db_setup.setup_db).

    Returns:
        pd.DataFrame with columns:
            Racer_ID  — racer identifier
            Race_Name — always 'older_chase'
            Season    — the year being predicted
            ZScore    — racer's z-score in their most recent available prior Chase
            HCTime    — racer's actual finishing time in the predicted Chase
            HCScore   — racer's actual z-score in the predicted Chase
    """
    sql = '''WITH Chase_Yrs AS
    (
        SELECT Racer_ID,
               CAST(strftime("%Y", CD.Chase_Date) AS INTEGER) - 1 AS Season,
               Time,
               ZScore_log
        FROM Results_Chase
        JOIN Chases AS CD
        ON CD.Chase_ID = Results_Chase.Chase_ID
    )
    SELECT
        curr.Racer_ID,
        "older_chase" AS Race_Name,
        curr.Season        AS Season,
        prev.ZScore_log    AS ZScore,
        curr.Time          AS HCTime,
        curr.ZScore_log    AS HCScore
    FROM Chase_Yrs AS curr
    JOIN Chase_Yrs AS prev
        ON  prev.Racer_ID = curr.Racer_ID
        AND prev.Season = (
            SELECT MAX(p.Season)
            FROM Chase_Yrs AS p
            WHERE p.Racer_ID = curr.Racer_ID
              AND p.Season < curr.Season
        )
    WHERE NOT EXISTS (
        SELECT 1 FROM Chase_Yrs AS py
        WHERE py.Racer_ID = curr.Racer_ID
          AND py.Season   = curr.Season - 1
    )
    ORDER BY curr.Racer_ID, curr.Season'''

    return pd.read_sql(sql, con)


def extract_older_chase_times(con):
    """
    Same pairing and gap logic as extract_older_chase_data(), but uses raw
    finishing time (seconds) as both predictor and target instead of z-scores.

    Only produced for years where the racer has no result in the immediately
    preceding year (complement of extract_previous_year_chase_times).

    Args:
        con: Active SQLite connection (configured via db_setup.setup_db).

    Returns:
        pd.DataFrame with columns:
            Racer_ID  — racer identifier
            Race_Name — always 'older_chase'
            Season    — the year being predicted
            PrevTime  — racer's finishing time (seconds) in their most recent prior Chase
            HCTime    — racer's actual finishing time (seconds) in the predicted Chase
            HCScore   — racer's actual z-score in the predicted Chase
    """
    sql = '''WITH Chase_Yrs AS
    (
        SELECT Racer_ID,
               CAST(strftime("%Y", CD.Chase_Date) AS INTEGER) - 1 AS Season,
               Time,
               ZScore_log
        FROM Results_Chase
        JOIN Chases AS CD
        ON CD.Chase_ID = Results_Chase.Chase_ID
    )
    SELECT
        curr.Racer_ID,
        "older_chase" AS Race_Name,
        curr.Season        AS Season,
        prev.Time          AS PrevTime,
        curr.Time          AS HCTime,
        curr.ZScore_log    AS HCScore
    FROM Chase_Yrs AS curr
    JOIN Chase_Yrs AS prev
        ON  prev.Racer_ID = curr.Racer_ID
        AND prev.Season = (
            SELECT MAX(p.Season)
            FROM Chase_Yrs AS p
            WHERE p.Racer_ID = curr.Racer_ID
              AND p.Season < curr.Season
        )
    WHERE NOT EXISTS (
        SELECT 1 FROM Chase_Yrs AS py
        WHERE py.Racer_ID = curr.Racer_ID
          AND py.Season   = curr.Season - 1
    )
    ORDER BY curr.Racer_ID, curr.Season'''

    return pd.read_sql(sql, con)


def extract_previous_year_chase_times(con):
    """
    Same pairing logic as extract_previous_year_chase_data(), but uses raw
    finishing time (seconds) rather than z-score as both predictor and target.

    Intended for a side-by-side comparison with the z-score version to determine
    which representation is a stronger predictor of next-year Chase performance.

    Args:
        con: Active SQLite connection (configured via db_setup.setup_db).

    Returns:
        pd.DataFrame with columns:
            Racer_ID  — racer identifier
            Race_Name — always 'previous_chase'
            Season    — the year being predicted
            PrevTime  — racer's finishing time (seconds) in the preceding Chase
            HCTime    — racer's actual finishing time (seconds) in the predicted Chase
            HCScore   — racer's actual z-score in the predicted Chase
    """
    sql = '''WITH Chase_Yrs AS
    (
        SELECT Racer_ID,
               CAST(strftime("%Y", CD.Chase_Date) AS INTEGER) - 1 AS Season,
               Time,
               ZScore_log
        FROM Results_Chase
        JOIN Chases AS CD
        ON CD.Chase_ID = Results_Chase.Chase_ID
    )
    SELECT
        curr.Racer_ID,
        "previous_chase" AS Race_Name,
        curr.Season        AS Season,
        prev.Time          AS PrevTime,
        curr.Time          AS HCTime,
        curr.ZScore_log    AS HCScore
    FROM Chase_Yrs AS curr
    JOIN Chase_Yrs AS prev
        ON  prev.Racer_ID = curr.Racer_ID
        AND prev.Season   = curr.Season - 1
    ORDER BY curr.Racer_ID, curr.Season'''

    return pd.read_sql(sql, con)



if __name__ == "__main__":

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    from fellpace.db.db_setup import setup_db
    from fellpace.config import DB_PATH

    con = setup_db(DB_PATH)

    zdf = extract_previous_year_chase_data(con).dropna(subset=["ZScore", "HCScore"])
    tdf = extract_previous_year_chase_times(con).dropna(subset=["PrevTime", "HCTime"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Previous Chase year vs Current Chase year — predictor comparison", fontsize=13)

    panels = [
        (
            axes[0],
            zdf["ZScore"].values,
            zdf["HCScore"].values,
            "Previous year z-score",
            "Current year z-score",
            "Standard z-score",
            "crimson",
        ),
        (
            axes[1],
            tdf["PrevTime"].values,
            tdf["HCTime"].values,
            "Previous year time (s)",
            "Current year time (s)",
            "Raw time",
            "steelblue",
        ),
    ]

    for ax, x, y, xlabel, ylabel, title, colour in panels:
        slope, intercept, r, *_ = stats.linregress(x, y)
        x_line = np.linspace(x.min(), x.max(), 200)
        ax.scatter(x, y, alpha=0.5, s=30, color=colour)
        ax.plot(x_line, slope * x_line + intercept, color="black", linewidth=1.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}  |  R² = {r**2:.3f}")
        ax.annotate(
            f"y = {slope:.3f}x + {intercept:.3f}",
            xy=(0.05, 0.92),
            xycoords="axes fraction",
            fontsize=9,
            color="black",
        )

    plt.tight_layout()
    plt.show()