from scipy.stats import norm
from fellpace.analysis_tools import convert_Chase_ZScore_logs_avg, get_chase_log_stats
from fellpace.extract.racers import get_racers_results
from fellpace.modelling.bayesian import calculate_initial_weights, calculate_recency_weights, recency_weighted_bayesian
from fellpace.parkrun.stats import parkrun_mean_std

from typing import Dict, Tuple
from datetime import date

import numpy as np
import pandas as pd


def get_predicted_times(con, coeffs: pd.DataFrame, racer_ID: int, season: int = -1) -> pd.DataFrame:

    racer_results = get_racers_results(con, racer_ID, season)
    if racer_results.empty:
        return pd.DataFrame()
    racer_results['PredZ'] = racer_results.apply(lambda x: np.polyval(coeffs[x['Race_Name']], x['ZScore']), axis=1)
    racer_results['Predicted Time'] = convert_Chase_ZScore_logs_avg(con, racer_results['PredZ'])
    
    return racer_results[['Racer_Name', 'Race_Name', 'Season', 'ZScore', 'PredZ', 'Predicted Time']].sort_values(['Season','Race_Name'])

def get_prediction_with_uncertainty_many(coeffs, cov_matrices, racer_results, residual_stds: pd.Series = None):
    def calculate_uncertainty(row):
        race = row['Race_Name']
        ZScore = row['ZScore']
        residual_std = 0.0
        if residual_stds is not None and race in residual_stds.index:
            residual_std = float(residual_stds[race])
        row['Zpred_mu'], row['Zpred_sig'] = get_prediction_with_uncertainty(
            coeffs[race],
            cov_matrices[race],
            ZScore,
            residual_std=residual_std,
        )
        return row

    racer_results_modified = racer_results.apply(calculate_uncertainty, axis=1)
    return racer_results_modified
        

def get_prediction_with_uncertainty(coeffs, cov_matrix, x, residual_std: float = 0.0):
    """
    Calculates the predicted value and its uncertainty (standard deviation) for a given x value.

    Args:
        coeffs: Coefficients of the linear regression (output of np.polyfit).
        cov_matrix: Covariance matrix of the regression coefficients.
        x: The x value for which to make the prediction.
        residual_std: Residual standard deviation for the race model in z-space.
            This captures irreducible model scatter and is added to coefficient
            covariance to get predictive uncertainty.

    Returns:
        A tuple containing the predicted value and its standard deviation.
    """
    # Calculate the predicted mean and variance
    mean_prediction = np.polyval(coeffs, x)
    x_vector = np.array([x, 1])  # For linear regression: [x, 1] corresponds to [slope, intercept]
    variance = np.dot(x_vector, np.dot(cov_matrix, x_vector.T)) + (residual_std ** 2)
    std_dev = np.sqrt(variance)

    return mean_prediction, std_dev

def make_chase_prediction(racer_result_with_predictions, prediction_year: int = None, verbose: bool = False) -> Tuple[float, float]:
    """
    Make a single prediction for the chase based on a series of individual predictions.
    
    This function prepares each result by calculating initial weights based on the results we have.
    It then adjusts these weights based on the recency of the results.
    
    Uses recency_weighted_bayesian as the core engine, including an explicit
    small-n uncertainty term so sparse data does not look overconfident.
    
    Args:
        racer_result_with_predictions (pd.DataFrame): DataFrame containing the results with predictions.
        prediction_year (int): The year for which the prediction is made. Defaults to current year.
        verbose (bool): If True, prints additional information about the prediction process.
    
    Returns:
        Tuple[float, float]: The predicted mean and standard deviation of the chase ZScore.
    """
    if prediction_year is None:
        prediction_year = date.today().year
    assert (racer_result_with_predictions['Season'] < prediction_year).all(), "All results must be from seasons BEFORE the prediction year"
    
    #  All values based on z distribution
    prior_mu = 0
    prior_sigma = 1.96
    
    # Calculate initial weights for each race
    racer_result_with_predictions['Initial_Weight'] = calculate_initial_weights(racer_result_with_predictions)
    
    # Update the weights based on recency
    initial_weights = racer_result_with_predictions['Initial_Weight'].values
    season = racer_result_with_predictions['Season'].values
    racer_result_with_predictions['Recency_Weight'] = calculate_recency_weights(
        prediction_year,
        season,
        initial_weights
    )
    
    # Extract values from the DataFrame
    weights = racer_result_with_predictions['Recency_Weight'].values
    mu_values = racer_result_with_predictions['Zpred_mu'].values
    sigma_values = racer_result_with_predictions['Zpred_sig'].values
    
    if verbose:
        race_names = (racer_result_with_predictions['Race_Name'] + ' ' + racer_result_with_predictions['Season'].astype(str)).values
    else:
        race_names = None
    
    predicted_mu, predicted_sigma = recency_weighted_bayesian(
        prior_mu,
        prior_sigma,
        mu_values,
        sigma_values,
        weights,
        race_names=race_names,
        small_n_tau=0.6,
        small_n_offset=1.0,
    )
    
    return predicted_mu, predicted_sigma
    

def get_chase_z_from_time(
    t_prev: float,
    race_name: str,
    time_coeffs: pd.Series,
    con,
) -> tuple:
    """Convert a previous Chase finishing time into a z-score prediction with uncertainty.

    Uses a pre-trained time-domain linear model (slope ≈ 1) to predict the
    expected finishing time, then converts to Chase z-score space via the
    delta method so the result is compatible with recency_weighted_bayesian.

    Maths:
        t_pred  = slope * t_prev + intercept
        z_pred  = (ln(t_pred) - mean_log) / std_log
        sigma_z = sigma_resid / (t_pred * std_log)     ← first-order propagation

    The sigma_resid is the empirical standard deviation of residuals from the
    time regression, giving uncertainty grounded in historical scatter rather
    than field-normalisation noise.

    Args:
        t_prev:      Previous Chase finishing time in seconds.
        race_name:   'previous_chase' or 'older_chase' — selects the correct model.
        time_coeffs: Loaded time models from load_time_models().
        con:         Active SQLite connection (for Chase log stats).

    Returns:
        Tuple[float, float]: (z_pred, sigma_z)
    """
    model = time_coeffs[race_name]
    slope         = model['slope']
    intercept     = model['intercept']
    sigma_resid   = model['sigma_resid']

    t_pred  = slope * t_prev + intercept
    mean_log, std_log = get_chase_log_stats(con)

    z_pred  = (np.log(t_pred) - mean_log) / std_log
    sigma_z = sigma_resid / (t_pred * std_log)

    return float(z_pred), float(sigma_z)


def get_probability_distribution(mean, std_dev, a = -3, b = 3, step=0.01):
    """
    Calculates the probability distribution of predictions within bounds [a, b],
    where each prediction is treated as a discrete second.

    Args:
        mean: Mean of the normal distribution.
        std_dev: Standard deviation of the normal distribution.
        a: Lower bound of the range (inclusive).
        b: Upper bound of the range (inclusive).
        step: Step size for the range (default is 1 second).

    Returns:
        A dictionary where keys are seconds in the range [a, b] and values are probabilities.
    """

    # Create a probability distribution for each second in the range [a, b]
    probabilities = {}
    current = a
    while current <= b:
        # Calculate the probability of the prediction being within this range
        prob = (
                norm.cdf(current + step / 2, loc=mean, scale=std_dev) -
                norm.cdf(current - step / 2, loc=mean, scale=std_dev)
               )
        probabilities[current] = prob
        current += step

    return pd.Series(probabilities)


def get_prediction_from_parkrun_time(
    con,
    parkrun_time: str,
    coeffs: pd.DataFrame,
    cov_matrices: Dict[str, np.ndarray],
    residual_std: float = 0.0,
) -> pd.DataFrame:
    """Get the predicted Zscore with uncertainty from a given parkrun time. This function does NOT convert to seconds.

    Args:
        con (dbconnection): Database connection.
        parkrun_time (str): The parkrun time in HH:MM:SS format.
        coeffs (pd.DataFrame): Coefficients for the prediction model.
        cov_matrices (Dict[str, np.ndarray]): Covariance matrices for the prediction model.

    Returns:
        pd.DataFrame: DataFrame with predicted Zscore and uncertainty.
    """

    # Convert parkrun time to seconds
    parkrun_seconds = sum(int(x) * 60 ** i for i, x in enumerate(reversed(parkrun_time.split(':'))))

    log_seconds = np.log(parkrun_seconds)

    stats = parkrun_mean_std(con, season = (date.today().year)-1)

    z_score = ((log_seconds - stats['Mean']) / stats['StdDev']).squeeze()

    mean, std = get_prediction_with_uncertainty(coeffs, cov_matrices, z_score, residual_std=residual_std)

    return mean, std, z_score

def get_prediction_time_from_parkrun_time(
    con,
    parkrun_time: str,
    prediction_year: int,
    coeffs: pd.DataFrame,
    cov_matrices: Dict[str, np.ndarray],
    residual_std: float = 0.0,
) -> pd.DataFrame:
    """
    Get the predicted times based on a parkrun time.
    
    Args:
        con: Database connection.
        parkrun_time (str): The parkrun time in HH:MM:SS format.
        prediction_year (int): The year for which the prediction is made.
        coeffs (pd.DataFrame): Coefficients for the prediction model.
        cov_matrices (Dict[str, np.ndarray]): Covariance matrices for the prediction model.
        
    Returns:
        pd.DataFrame: DataFrame with predicted times.
    """
    # Convert parkrun time to seconds
    parkrun_seconds = sum(int(x) * 60 ** i for i, x in enumerate(reversed(parkrun_time.split(':'))))
    
    log_seconds = np.log(parkrun_seconds)
    
    stats = parkrun_mean_std(con, season = prediction_year-1)
    
    z_score = ((log_seconds - stats['Mean']) / stats['StdDev']).squeeze()
    
    mean, std = get_prediction_with_uncertainty(coeffs, cov_matrices, z_score, residual_std=residual_std)
    
    pr_prediction = mean - (1.96 * std) 
    
    pr_prediction_t = convert_Chase_ZScore_logs_avg(con, pr_prediction)[0]
    
    return pr_prediction_t
    
if __name__ == "__main__":
    # This is just a placeholder to prevent execution when imported
    from fellpace.db.db_setup import setup_db
    from fellpace.config import DB_PATH
    from fellpace.modelling.training import load_models
    con = setup_db(DB_PATH)
    coeffs, covars = load_models()
    get_prediction_time_from_parkrun_time(con, "23:30", coeffs, covars)