from scipy.stats import norm
from fellpace.analysis_tools import convert_Chase_ZScore_logs_avg
from fellpace.extract.racers import get_racers_results


import numpy as np
import pandas as pd


def get_predicted_times(con, coeffs: pd.DataFrame, racer_ID: int, season: int = -1) -> pd.DataFrame:

    racer_results = get_racers_results(con, racer_ID, season)
    if racer_results.empty:
        return pd.DataFrame()
    racer_results['PredZ'] = racer_results.apply(lambda x: np.polyval(coeffs[x['Race_Name']], x['ZScore']), axis=1)
    racer_results['Predicted Time'] = convert_Chase_ZScore_logs_avg(con, racer_results['PredZ'])
    
    return racer_results[['Racer_Name', 'Race_Name', 'Season', 'ZScore', 'PredZ', 'Predicted Time']].sort_values(['Season','Race_Name'])


def get_prediction_probability_distribution(coeffs, cov_matrix, x, a = -3, b = 3, step=0.01):
    """
    Calculates the probability distribution of predictions within bounds [a, b],
    where each prediction is treated as a discrete second.

    Args:
        coeffs: Coefficients of the linear regression (output of np.polyfit).
        cov_matrix: Covariance matrix of the regression coefficients.
        x: The x value for which to make the prediction.
        a: Lower bound of the range (inclusive).
        b: Upper bound of the range (inclusive).
        step: Step size for the range (default is 1 second).

    Returns:
        A dictionary where keys are seconds in the range [a, b] and values are probabilities.
    """
    # Calculate the predicted mean and variance
    mean_prediction = np.polyval(coeffs, x)
    x_vector = np.array([x, 1])  # For linear regression: [x, 1] corresponds to [slope, intercept]
    variance = np.dot(x_vector, np.dot(cov_matrix, x_vector.T))
    std_dev = np.sqrt(variance)

    # Create a probability distribution for each second in the range [a, b]
    probabilities = {}
    current = a
    while current <= b:
        # Calculate the probability of the prediction being within this range
        prob = (
                norm.cdf(current + step / 2, loc=mean_prediction, scale=std_dev) -
                norm.cdf(current - step / 2, loc=mean_prediction, scale=std_dev)
               )
        probabilities[current] = prob
        current += step

    return probabilities