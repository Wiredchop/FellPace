from fellpace.analysis_tools import convert_Chase_ZScore_logs_avg
from fellpace.extract.racers import get_racer_id, get_racers_results


import numpy as np
import pandas as pd


def get_predicted_time(con, coeffs: pd.DataFrame, racer_ID: int, season: int = -1) -> pd.DataFrame:

    racer_results = get_racers_results(con, racer_ID, season)
    if racer_results.empty:
        return pd.DataFrame()
    racer_results['PredZ'] = racer_results.apply(lambda x: np.polyval(coeffs[x['Race_Name']], x['ZScore']), axis=1)
    racer_results['Predicted Time'] = convert_Chase_ZScore_logs_avg(con, racer_results['PredZ'])
    
    return racer_results[['Racer_Name', 'Race_Name', 'Season', 'ZScore', 'Predicted Time']]