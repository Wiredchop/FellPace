import numpy as np
import pandas as pd
from fellpace.analysis_tools import convert_Chase_ZScore_logs
from fellpace.db.db_setup import setup_db
from fellpace.config import DB_PATH, COEFFS_FILE_PATH, COVAR_FILE_PATH, RESID_STD_FILE_PATH, TIME_COEFFS_FILE_PATH, ROAD_TIME_COEFFS_FILE_PATH

def load_models(include_residuals: bool = True):
    if COEFFS_FILE_PATH.exists() and COVAR_FILE_PATH.exists():
        coeffs = pd.read_json(COEFFS_FILE_PATH, orient='index', typ='series')
        covar = pd.read_json(COVAR_FILE_PATH, orient='index', typ='series')
    else:
        print("No model files found. Please train the models first.")

    if include_residuals:
        if RESID_STD_FILE_PATH.exists():
            resid_stds = pd.read_json(RESID_STD_FILE_PATH, typ='series')
        else:
            # Backward compatibility with older model artifacts.
            resid_stds = pd.Series(dtype=float)
        return coeffs, covar, resid_stds

    return coeffs, covar


def train_time_models(data: pd.DataFrame) -> pd.Series:
    """Train time-domain linear models mapping previous Chase time → current Chase time.

    For each Race_Name group, fits a linear regression on raw seconds and stores
    the slope, intercept, and residual standard deviation. These three values
    are used at prediction time to convert a known previous Chase time into a
    z-score prediction with realistic uncertainty via the delta method:

        t_pred  = slope * t_prev + intercept
        sigma_z = sigma_resid / (t_pred * std_log_chase)

    Unlike the z-score models, RANSAC is not applied here — the time relationship
    is already known to be clean (slope ≈ 1) and the small dataset makes RANSAC
    unreliable.

    Args:
        data: DataFrame with columns Race_Name, PrevTime, HCTime.
              Typically the concatenation of extract_previous_year_chase_times()
              and extract_older_chase_times().

    Returns:
        pd.Series indexed by Race_Name, each value a dict with keys:
            slope, intercept, sigma_resid
    """
    def _fit_group(group):
        x = group['PrevTime'].values
        y = group['HCTime'].values
        slope, intercept = np.polyfit(x, y, 1)
        y_pred = slope * x + intercept
        sigma_resid = float(np.std(y - y_pred, ddof=2))
        return {'slope': float(slope), 'intercept': float(intercept), 'sigma_resid': sigma_resid}

    return data.groupby('Race_Name').apply(_fit_group)


def load_time_models() -> pd.Series:
    """Load previously trained time-domain chase models from disk.

    Returns:
        pd.Series indexed by Race_Name with dict values (slope, intercept, sigma_resid),
        or None if no file exists.
    """
    if TIME_COEFFS_FILE_PATH.exists():
        return pd.read_json(TIME_COEFFS_FILE_PATH, orient='index', typ='series')
    print("No time model file found. Please train the models first.")
    return None


def load_road_time_models() -> pd.Series:
    """Load previously trained road time models (5k/10k PO10 -> Chase) from disk.

    Returns:
        pd.Series indexed by Race_Name ('p10_5k', 'p10_10k') with dict values
        (slope, intercept, sigma_resid), or None if no file exists.
    """
    if ROAD_TIME_COEFFS_FILE_PATH.exists():
        return pd.read_json(ROAD_TIME_COEFFS_FILE_PATH, orient='index', typ='series')
    print("No road time model file found. Please train the models first.")
    return None
        

def train_models(data_Zs, use_inliers=True):
    if use_inliers:
        data_Zs = data_Zs.loc[data_Zs['inlier'] == True]
    
    fit_results = data_Zs.groupby('Race_Name').apply(lambda x: np.polyfit(x['ZScore'], x['HCScore'], 1, full=False, cov=True))
    coeffs = fit_results.apply(lambda c: c[0])
    covar = fit_results.apply(lambda c: c[1])

    # Residual scatter in z-space (irreducible noise) is part of predictive
    # uncertainty and must be added on top of coefficient covariance.
    resid_stds = data_Zs.groupby('Race_Name').apply(
        lambda x: np.std(x['HCScore'] - np.polyval(coeffs[x.name], x['ZScore']), ddof=2)
    )

    return coeffs, covar, resid_stds

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