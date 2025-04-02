from scipy.stats import zscore, percentileofscore
from typing import Tuple
import math
import pandas as pd
import sqlite3
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from typing import Tuple
from sklearn.linear_model import LinearRegression

def calculate_position_stats(time: npt.ArrayLike) -> Tuple[npt.ArrayLike, npt.ArrayLike,npt.ArrayLike]:
    """Calculate position stats, specifically the percentile AND zscore for position data TIME!!!
    It is useful to have both, the zscore assumes normality but has additional power at the tails of the distribution
    as it calculates how far ahead/behind someone may be from the average rather than just relative position within a group
    As race times are typically log-normal. We will also calculate the ZScore of the log-transformed times.

    Args:
        time (np.ndarray): A 1D np array of time values in seconds, returned from a pandas dataframe

    Returns:
        Tuple[np.ndarray,np.ndarray, np.ndarray]: A tuple containing the (zscore,zscore_log, percentile)
    """
    #Ensure the time is not a string object
    time = time.astype(np.int32)
    zscores = zscore(time,nan_policy = 'omit')
    time_log = np.log(time)
    zscores_log = zscore(time_log,nan_policy='omit')
    
    percentiles = percentileofscore(time,time,nan_policy = 'omit')
    #Percentile rounding errors can occur, so enforcing 0 - 100
    percentiles[percentiles < 0] = 0
    percentiles[percentiles > 100] = 100
    
    return (zscores,zscores_log,percentiles)


def get_linear_models(data:pd.DataFrame, g: str,x: str,y: str):
    # Get linear models for each race, present coefficients in a table so can correct Zscore between races
    data_for_models = data[[g,x,y]]
    
    def calculate_linear_model(group):
        x_values = group[x].values.reshape(-1,1)
        y_values = group[y].values
        model = LinearRegression().fit(x_values,y_values)
        group['ModelZ'] = model.predict(group[x].values.reshape(-1,1))
        return model
    Z_models = data_for_models.groupby(g).apply(calculate_linear_model)
    return Z_models.to_dict()

def remove_outliers(data: pd.DataFrame,x: str, y: str,g:str, thresh: float = 2.5)-> Tuple[pd.DataFrame,pd.Series,np.ndarray]:
    """Remove outliers from an x:y plot by assuming an x=y relationship and transforming data along this line
    this is a particularly simple transformation, could transform along regression line for more sophisticated method.
    After transformation, calculate z_score of distances from lines and remove outliers

    Args:
        data (pd.DataFrame): a dataframe to be cleaned
        x (str): the name of the column to use as x data
        y (str): the name of the column to use as y data
        g (str): the name of the column to group by
        thresh(float): what threshold to use as a z_score default as 2.5

    Returns:
       pd.DataFrame: the cleaned dataframe it will contain a new column ZoCZ giving the zscore of the transformed data points
       pd.Series:   The ZocZ values calculated by the algorithm    
       pd.Series:    A series of the same length as the original dataset containing "Cleaned" or "Included" based on the results of the cleaning algorithm
       """
    x_dat = data[x]
    y_dat = data[y]
    a = math.sqrt(2)/2
    data["ZoC"] = x_dat*a - y_dat * a

    def get_zscore(group):
        group["ZoCZ"] = zscore(group["ZoC"])
        return group

    data = data.groupby(g,as_index=False, group_keys=True).apply(get_zscore)
    data.drop(columns=["ZoC"])
    #Strip out records where Z is above 2.5
    return data.loc[abs(data["ZoCZ"]) <=thresh],data["ZoCZ"] ,np.where(abs(data["ZoCZ"])<=thresh,"Included","Cleaned")

def convert_Chase_ZScore_logs(con: sqlite3.Connection,Zscore_logs: pd.Series, year: int):
    
    # Add logarithm to the sqllite connection

    def ln(t):
        return np.log(t)
    con.create_function("ln", 1, ln)

    """This function extracts raw stats from the original Chase data in order to convert back Zscore log data.
    

    Args:
        con (sqlite3.Connection): A connection to the fellpace database. MUST HAVE sttdev ADDED!!
        Zscore_logs (pd.Series): A series of Zscore_log values to be converted into expected times
        year (int): We also need the year of the chase for which the times need converting
    """
    
    SQL_get_log_chase_stats = '''
        WITH Timel AS
        (
            SELECT *, ln(Time) Timel
            FROM Results_Chase
            WHERE Time IS NOT NULL
        ),

        sds AS
        (
            SELECT Chase_ID, stddev(Timel) AS sd, avg(Timel) AS mn
            FROM Timel
            GROUP BY Chase_ID
        )
        

        SELECT cast(strftime("%Y",C.Chase_Date) as integer) as Year, R.sd, R.mn FROM sds as R   
                JOIN Chases as C ON C.Chase_ID = R.Chase_ID
                WHERE Year == ?   
    '''
    year = int(year)
    # Ensure the year is an integer, np int type doesn't work with sqlite3
    Chase_stats = pd.read_sql(SQL_get_log_chase_stats,con,params=(year,))
    pred_logs = Chase_stats['mn'].values + Chase_stats['sd'].values * Zscore_logs
    return np.exp(pred_logs)
    
def convert_Chase_ZScore_logs_avg(con: sqlite3.Connection,Zscore_logs: pd.Series):
    """This does the same as the previous function but does not require a year input. When predicting
    times, we will not have that year's HC results and so an average of all available data must be taken

    Args:
        con (sqlite3.Connection): A connection to a database -- A standard deviation function 'stddev'
        must be included
        Zscore_logs (pd.Series): A series of ZScores

    Returns:
        _type_: Expected times based on ZScore values
    """
    # Add logarithm to the sqllite connection

    def ln(t):
        return np.log(t)
    con.create_function("ln", 1, ln)

    """This function extracts raw stats from the original Chase data in order to convert back Zscore log data.
    

    Args:
        con (sqlite3.Connection): A connection to the fellpace database. MUST HAVE sttdev ADDED!!
        Zscore_logs (pd.Series): A series of Zscore_log values to be converted into expected times
        year (int): We also need the year of the chase for which the times need converting
    """
    
    SQL_get_log_chase_stats = '''
        WITH Timel AS
        (
            SELECT *, ln(Time) Timel
            FROM Results_Chase
            WHERE Time IS NOT NULL
        ),

        sds AS
        (
            SELECT Chase_ID, stddev(Timel) AS sd, avg(Timel) AS mn
            FROM Timel
            GROUP BY Chase_ID
        )
        

        SELECT  avg(R.sd) as sd, avg(R.mn) as mn FROM sds as R
    '''
    
    Chase_stats = pd.read_sql(SQL_get_log_chase_stats,con)
    pred_logs = Chase_stats['mn'].values + Chase_stats['sd'].values * Zscore_logs
    return np.exp(pred_logs)