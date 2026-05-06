import numpy as np
import pandas as pd
from tabulate import tabulate
from loguru import logger


def calculate_initial_weights(racer_results: pd.DataFrame, lower_weight: float = 0.8, heavier_weight: float = 1.2) -> pd.Series:
    """
    Calculate what the initial weights should be for each race.
    
    Most races are weighted as 1, Parkrun has a lower weight due to uncertainty in the results.
    Previous Hallam Chase results are weighted more heavily as they are direct comparisons.

    """
    
    initial_weights = np.where(
        racer_results['Race_Name'].str.contains('PR_'),
        lower_weight,
        np.where(
            racer_results['Race_Name'].str.contains('Hallam Chase'),
            heavier_weight,
            1.0
        )
    )
    
    return pd.Series(initial_weights, index=racer_results.index, name='Initial_Weight')

def calculate_recency_weights(year_to_predict: int, season: np.ndarray, initial_weights: np.ndarray, lambda_decay: float=0.25):
    """
    Calculate recency weights based on the time since the race.
    
    Weights are an exponential decay function with a customisable decay rate.
    
    This function does NOT normalise the weights, as they are used to scale the precision of the observed values.
    
    Args:
        year_to_predict (int): The year that is being predicted.
        season (np.ndarray): The seasons of the races.
        initial_weights (np.ndarray): The initial weights for each race.
        lambda_decay (float): The decay rate for the exponential function.
        
    """
    # Guard against object-dtype arrays coming from mixed pandas frames.
    season = pd.to_numeric(pd.Series(season), errors='coerce').to_numpy(dtype=float)
    initial_weights = pd.to_numeric(pd.Series(initial_weights), errors='coerce').fillna(0.0).to_numpy(dtype=float)

    if np.isnan(season).any():
        raise ValueError("Season values must be numeric for recency weighting.")

    if (season >= year_to_predict).any():
        logger.critical(f"Cannot adjust weights for seasons ahead of the prediction year.")
        raise ValueError("All seasons must be before the prediction year.")
    time_since_race = (year_to_predict - 1) - season    
    return initial_weights * np.exp(-lambda_decay * time_since_race)


def recency_weighted_bayesian(
    prior_mu,
    prior_sigma,
    observed_mu,
    observed_sigma,
    weights,
    race_names=None,
    small_n_tau: float = 0.6,
    small_n_offset: float = 1.0,
):
    """
    Compute the posterior mean and standard deviation using a recency-weighted Bayesian approach.

    Uses overdispersion scaling to inflate the posterior variance when observed predictions
    disagree more than their individual uncertainties would suggest. This is a multiplicative
    correction (like quasi-likelihood) rather than an additive one, so it only inflates
    uncertainty when the data genuinely conflicts.

    Args:
        prior_mu: Prior mean (e.g. population average).
        prior_sigma: Prior standard deviation.
        observed_mu: Array of observed prediction means.
        observed_sigma: Array of observed prediction standard deviations.
        weights: Recency weights for each observation.
        race_names: Optional array of race names for logging.
        small_n_tau: Scale of small-n uncertainty penalty (z-score units).
            Higher values increase posterior uncertainty when data is sparse.
        small_n_offset: Controls how quickly the small-n penalty decays as
            effective sample size increases.

    Returns:
        (posterior_mu, posterior_sigma): Posterior mean and standard deviation.
    """

    precisions = weights / (observed_sigma ** 2)
    prior_precision = 1 / (prior_sigma ** 2)

    if race_names is not None:
        table_data = list(zip(race_names, weights))
        headers = ["Race Name", "Weight"]
        print(tabulate(table_data, headers=headers, tablefmt="rounded_outline"))

    # Standard Bayesian combination
    total_precision = np.sum(precisions) + prior_precision
    posterior_mu = (np.sum(precisions * observed_mu) + prior_precision * prior_mu) / total_precision
    naive_posterior_variance = 1 / total_precision

    # Overdispersion: inflate variance if observations scatter more than expected
    n = len(observed_mu)
    if n > 1:
        chi_sq = np.sum(precisions * (observed_mu - posterior_mu) ** 2)
        dof = n - 1
        overdispersion_factor = max(1.0, chi_sq / dof)
    else:
        overdispersion_factor = 1.0

    # Small-n penalty: add extra variance when effective sample size is low.
    # This preserves mean behavior while preventing overconfident estimates for sparse data.
    effective_n = max(float(np.sum(np.clip(weights, 0.0, None))), 1e-9)
    if small_n_tau > 0:
        small_n_penalty_variance = (small_n_tau ** 2) / (effective_n + small_n_offset)
    else:
        small_n_penalty_variance = 0.0

    posterior_variance = (naive_posterior_variance * overdispersion_factor) + small_n_penalty_variance
    posterior_sigma = np.sqrt(posterior_variance)

    logger.debug(
        f"Overdispersion factor: {overdispersion_factor:.3f}, "
        f"small_n_penalty_variance: {small_n_penalty_variance:.4f}, "
        f"effective_n: {effective_n:.3f}"
    )

    return posterior_mu, posterior_sigma


def hierarchical_bayesian_model(global_race_mu, global_race_sigma, observed_times, observed_race_variability, time_since_race, lambda_decay = 0.1):
    """
    Compute the posterior mean and standard deviation using a hierarchical Bayesian model.
    
    Two-level hierarchy:
        1. Race-level: combines observed race means with a global prior, weighted by recency.
        2. Runner-level: combines the race-level posterior with the individual observations.
    
    Args:
        global_race_mu: Global prior mean for race times.
        global_race_sigma: Global prior standard deviation.
        observed_times: Array of observed race times.
        observed_race_variability: Array of per-race standard deviations.
        time_since_race: Array of years since each race.
        lambda_decay: Exponential decay rate for recency weighting.
    
    Returns:
        (runner_posterior_mean, runner_posterior_sigma): Posterior mean and standard deviation.
    """
    recency_weights = np.exp(-lambda_decay * time_since_race)

    # Scale race-specific precision by recency weight
    race_precisions = recency_weights / (observed_race_variability ** 2)
    global_precision = 1 / (global_race_sigma ** 2)

    # Race-level posterior
    race_mean_posterior = (np.sum(race_precisions * observed_times) + global_precision * global_race_mu) / (np.sum(race_precisions) + global_precision)
    race_sigma_posterior = np.sqrt(1 / (np.sum(race_precisions) + global_precision))

    # Runner-level posterior
    runner_precision = 1 / (race_sigma_posterior ** 2)
    runner_posterior_mean = (np.sum(race_precisions * observed_times) + runner_precision * race_mean_posterior) / (np.sum(race_precisions) + runner_precision)
    runner_posterior_sigma = np.sqrt(1 / (np.sum(race_precisions) + runner_precision))

    return runner_posterior_mean, runner_posterior_sigma

# Example usage
if __name__ == "__main__":

    # Recency-weighted Bayesian example
    prior_mu = 300
    prior_sigma = 20
    observed_mu = np.array([305, 290, 355])
    observed_sigma = np.array([15, 18, 12])
    year_to_predict = 2026
    season = np.array([2025, 2021, 2016])
    initial_weights = np.array([1.0, 1.0, 1.0])
    lambda_decay = 0.1

    weights = calculate_recency_weights(year_to_predict, season, initial_weights, lambda_decay)
    posterior_mu, posterior_sigma = recency_weighted_bayesian(
        prior_mu, prior_sigma, observed_mu, observed_sigma, weights,
        race_names=["Race A", "Race B", "Race C"]
    )
    print(f"Posterior Mean (Recency-Weighted): {posterior_mu:.2f} seconds")
    print(f"Posterior Standard Deviation: {posterior_sigma:.2f} seconds")

    print("\n" + "="*70)
    print("Small-n uncertainty example")
    print("="*70)

    observed_mu_single = np.array([450.0])
    observed_sigma_single = np.array([10.0])
    weights_single = np.array([1.0])

    posterior_mu_no_small_n, posterior_sigma_no_small_n = recency_weighted_bayesian(
        prior_mu=300,
        prior_sigma=20,
        observed_mu=observed_mu_single,
        observed_sigma=observed_sigma_single,
        weights=weights_single,
        small_n_tau=0.0,
    )
    print(f"No small-n term: mean={posterior_mu_no_small_n:.2f}, sigma={posterior_sigma_no_small_n:.2f}")

    posterior_mu_small_n, posterior_sigma_small_n = recency_weighted_bayesian(
        prior_mu=300,
        prior_sigma=20,
        observed_mu=observed_mu_single,
        observed_sigma=observed_sigma_single,
        weights=weights_single,
        small_n_tau=0.6,
    )
    print(f"With small-n term: mean={posterior_mu_small_n:.2f}, sigma={posterior_sigma_small_n:.2f}")

    # Hierarchical Bayesian example
    print("\n" + "="*70)
    global_race_mu = 300
    global_race_sigma = 20
    observed_times = np.array([305, 290, 355])
    observed_race_variability = np.array([15, 18, 12])
    time_since_race = np.array([1, 5, 19])

    runner_posterior_mean, runner_posterior_sigma = hierarchical_bayesian_model(
        global_race_mu, global_race_sigma, observed_times, observed_race_variability, time_since_race
    )
    print(f"Final Runner-Specific Posterior Mean: {runner_posterior_mean:.2f} seconds")
    print(f"Final Runner-Specific Std Dev: {runner_posterior_sigma:.2f} seconds")
