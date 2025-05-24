import numpy as np
import pandas as pd
from fellpace.analysis_tools import convert_Chase_ZScore_logs
from fellpace.db.db_setup import setup_db
from fellpace.config import DB_PATH, COEFFS_FILE_PATH, COVAR_FILE_PATH

def load_models():
    if COEFFS_FILE_PATH.exists() and COVAR_FILE_PATH.exists():
        coeffs = pd.read_json(COEFFS_FILE_PATH, orient='index', typ='series')
        covar = pd.read_json(COVAR_FILE_PATH, orient='index', typ='series')
    else:
        print("No model files found. Please train the models first.")
    return coeffs, covar
        

def train_models(data_Zs, use_inliers=True):
    if use_inliers:
        data_Zs = data_Zs.loc[data_Zs['inlier'] == True]
    
    fit_results = data_Zs.groupby('Race_Name').apply(lambda x: np.polyfit(x['ZScore'], x['HCScore'], 1, full=False, cov=True))
    coeffs = fit_results.apply(lambda c: c[0])
    covar = fit_results.apply(lambda c: c[1])
    return coeffs, covar

def get_rmse_in_seconds(data_Zs: pd.DataFrame, coeffs, evaluate_inliers_only = True):
    if evaluate_inliers_only:
        data_Zs = data_Zs.loc[data_Zs['inlier'] == True]
    data_Zs.set_index(['Race_Name','Season', 'Racer_ID'], inplace=True)
    data_Zs['predicted_Z'] = (
        data_Zs.groupby(['Race_Name','Season', 'Racer_ID'], sort=False)
        .apply(
            lambda x: 
                np.polyval(coeffs[x.name[0]], x['ZScore'])[0])
        )
    con = setup_db(DB_PATH)

    predicted_times = (data_Zs.groupby(['Race_Name','Season'], sort=False, group_keys=False)
    .apply(
        lambda x: convert_Chase_ZScore_logs(con, x['predicted_Z'], year = x.name[1] + 1)
        )
        
    )
    
    data_Zs['residuals'] = predicted_times - data_Zs['HCTime']
    RMSE = data_Zs.groupby('Race_Name').apply(lambda x: np.sqrt(np.mean(x['residuals']**2)))
    return RMSE