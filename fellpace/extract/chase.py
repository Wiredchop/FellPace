"Tools to extract chase related data from the database."
import pandas as pd
from sqlite3 import Connection

from fellpace.convert_tools import seconds_to_time_string
from fellpace.extract.racers import secure_racer_id

def get_previous_chase_results(con: Connection, racer_id: int = None, racer_name: str = None) -> pd.DataFrame:
    """
    Get previous chase results for a given racer.
    
    Provide either racer_id or racer_name, not both.
    
    Args:
        con (Connection): SQLite connection object.
        racer_id (int): ID of the racer.
        racer_name (str): Name of the racer.
        
    Returns:
        pd.DataFrame: DataFrame containing previous chase results.
    """
    sql = """
    SELECT C.Time, strftime('%Y', CH.Chase_Date) AS Season
    FROM Results_Chase AS C
    JOIN Chases AS CH ON C.Chase_ID = CH.Chase_ID
    WHERE C.Racer_ID = ?
    ORDER BY CH.Chase_Date DESC
    """
    assert (racer_id is not None) ^ (racer_name is not None), "Provide either racer_id or racer_name, not both."
    if racer_name:
        racer_id = secure_racer_id(con, racer_name.lower().strip())
    if racer_id is None:
        logger.warning(f"Racer {racer_name} not found in database.")
        return pd.DataFrame(columns=['Time', 'Season'])
    racer_id = int(racer_id)  # Ensure racer_id is an integer
    return pd.read_sql(sql, con, params=(racer_id,))

def extract_result_for_year(results: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Extract results for a specific year.
    
    Args:
        results (pd.DataFrame): Data
        """
    
    chase_time = results.loc[results['Season'] == str(year), "Time"].squeeze()
    
    if type(chase_time) == pd.Series:
        chase_time = "N/A"
    else:
        chase_time = seconds_to_time_string(chase_time)
    assert type(chase_time) in [float, int, str], f"Last year time should be a number or string, got {type(chase_time)}"
    return chase_time