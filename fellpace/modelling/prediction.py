from scipy.stats import norm
from fellpace.analysis_tools import convert_Chase_ZScore_logs_avg
from fellpace.extract.racers import get_racers_results
from fellpace.modelling.bayesian import calculate_initial_weights, calculate_recency_weights ,recency_weighted_bayesian
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

def get_prediction_with_uncertainty_many(coeffs, cov_matrices, racer_results):
    def calculate_uncertainty(row):
        race = row['Race_Name']
        ZScore = row['ZScore']
        row['Zpred_mu'], row['Zpred_sig'] = get_prediction_with_uncertainty(coeffs[race], cov_matrices[race], ZScore)
        return row

    racer_results_modified = racer_results.apply(calculate_uncertainty, axis=1)
    return racer_results_modified
        

def get_prediction_with_uncertainty(coeffs, cov_matrix, x):
    """
    Calculates the predicted value and its uncertainty (standard deviation) for a given x value.

    Args:
        coeffs: Coefficients of the linear regression (output of np.polyfit).
        cov_matrix: Covariance matrix of the regression coefficients.
        x: The x value for which to make the prediction.

    Returns:
        A tuple containing the predicted value and its standard deviation.
    """
    # Calculate the predicted mean and variance
    mean_prediction = np.polyval(coeffs, x)
    x_vector = np.array([x, 1])  # For linear regression: [x, 1] corresponds to [slope, intercept]
    variance = np.dot(x_vector, np.dot(cov_matrix, x_vector.T))
    std_dev = np.sqrt(variance)

    return mean_prediction, std_dev

def make_chase_prediction(racer_result_with_predictions, prediction_year: int = None, verbose: bool = False) -> Tuple[float, float]:
    """
    Make a single prediction for the chase based on a series of individual predictions.
    
    This function prepares each result by calculating initial weights based on the results we have.
    It then adjusts these weights based on the recency of the results.
    
    From this it adds the results to a bayesian update model to get a final prediction.
    
    Args:
        racer_result_with_predictions (pd.DataFrame): DataFrame containing the results with predictions.
        prediction_year (int): The year for which the prediction is made. Defaults to current year.
        verbose (bool): If True, prints additional information about the prediction process.
    
    Returns:
        Tuple[float, float]: The predicted mean and standard deviation of the chase ZScore.
    """
    assert (racer_result_with_predictions['Season'] < prediction_year).all(), "All results must be from seasons BEFORE the prediction year"
    if prediction_year is None:
        prediction_year = date.today().year
    
    #  All values based on z distribution
    prior_mu = 0
    prior_sigma = 1
    
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
    predicted_mu, predicted_sigma = recency_weighted_bayesian(prior_mu, prior_sigma, mu_values, sigma_values, weights, race_names=race_names) 
    return predicted_mu, predicted_sigma
    

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


def get_prediction_from_parkrun_time(con, parkrun_time: str, coeffs: pd.DataFrame, cov_matrices: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Get the predicted times based on a parkrun time.
    
    Args:
        con: Database connection.
        parkrun_time (str): The parkrun time in HH:MM:SS format.
        coeffs (pd.DataFrame): Coefficients for the prediction model.
        cov_matrices (Dict[str, np.ndarray]): Covariance matrices for the prediction model.
        
    Returns:
        pd.DataFrame: DataFrame with predicted times.
    """
    # Convert parkrun time to seconds
    parkrun_seconds = sum(int(x) * 60 ** i for i, x in enumerate(reversed(parkrun_time.split(':'))))
    
    log_seconds = np.log(parkrun_seconds)
    
    stats = parkrun_mean_std(con, season = (date.today().year)-1)
    
    z_score = ((log_seconds - stats['Mean']) / stats['StdDev']).squeeze()
    
    mean, std = get_prediction_with_uncertainty(coeffs, cov_matrices, z_score)
    
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
    get_prediction_from_parkrun_time(con, "23:30", coeffs, covars)