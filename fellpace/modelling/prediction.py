from scipy.stats import norm
from fellpace.analysis_tools import convert_Chase_ZScore_logs_avg
from fellpace.extract.racers import get_racers_results
from fellpace.modelling.bayesian import recency_weighted_bayesian
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
    
    
    """
    if prediction_year is None:
        prediction_year = date.today().year
    
    #  All values based on z distribution
    prior_mu = 0
    prior_sigma = 1
    
    # Extract values from the DataFrame
    performance_years = racer_result_with_predictions['Season'].values
    mu_values = racer_result_with_predictions['Zpred_mu'].values
    sigma_values = racer_result_with_predictions['Zpred_sig'].values
    
    # Calculate the time since the performance year    
    times_since_performance = np.array([prediction_year - year for year in performance_years])
    if verbose:
        race_names = (racer_result_with_predictions['Race_Name'] + ' ' + racer_result_with_predictions['Season'].astype(str)).values
    else:
        race_names = None
    predicted_mu, predicted_sigma = recency_weighted_bayesian(prior_mu, prior_sigma, mu_values, sigma_values, times_since_performance, race_names=race_names) 
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