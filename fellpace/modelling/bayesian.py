import numpy as np
from tabulate import tabulate

def recency_weighted_bayesian(prior_mu, prior_sigma, observed_mu, observed_sigma, time_since_race,race_names = None, lambda_decay=0.25):
    """
    Compute the posterior mean and standard deviation using a recency-weighted Bayesian approach.
    """
    weights = np.exp(-lambda_decay * time_since_race)
    precisions = weights / (observed_sigma ** 2)
    prior_precision = 1 / (prior_sigma ** 2)
    if race_names is not None:
        table_data = list(zip(race_names, weights))
        headers = ["Race Name", "Weight"]
        print(tabulate(table_data, headers=headers, tablefmt="rounded_outline"))

    # Adjust dispersion variance to account for race weights based on observed_mu values
    normalised_weights = weights / np.sum(weights)
    
    weighted_variances = normalised_weights * ((observed_mu - np.mean(observed_mu)) ** 2)
    dispersion_variance = np.sum(weighted_variances)   # Weighted variance based on observed_mu

    posterior_mu = (np.sum(precisions * observed_mu) + prior_precision * prior_mu) / (np.sum(precisions) + prior_precision)
    # Adjust posterior variance calculation to scale dispersion variance
    posterior_variance = 1 / (np.sum(precisions) + prior_precision) + (dispersion_variance/2)
    posterior_sigma = np.sqrt(posterior_variance)

    return posterior_mu, posterior_sigma

def hierarchical_bayesian_model(global_race_mu, global_race_sigma, observed_times, observed_race_variability, time_since_race, lambda_decay = 0.1):
    """
    Compute the posterior mean and standard deviation using a hierarchical Bayesian model.
    """
    race_means = observed_times + np.random.normal(0, observed_race_variability)
    # Define exponential decay weights for recency
    lambda_decay = 0.1  # Controls how fast older races lose importance
    recency_weights = np.exp(-lambda_decay * time_since_race)

    # Scale race-specific precision by recency weight
    race_precisions = (recency_weights / (observed_race_variability ** 2))
    

    race_mean_posterior = (np.sum(race_precisions * race_means) + (1 / global_race_sigma**2) * global_race_mu) / (np.sum(race_precisions) + (1 / global_race_sigma**2))
    race_sigma_posterior = np.sqrt(1 / (np.sum(race_precisions) + (1 / global_race_sigma**2)))

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
    time_since_race = np.array([1, 5, 10])
    lambda_decay = 0.1

    posterior_mu, posterior_sigma = recency_weighted_bayesian(prior_mu, prior_sigma, observed_mu, observed_sigma, time_since_race, lambda_decay)
    print(f"Posterior Mean (Recency-Weighted): {posterior_mu:.2f} seconds")
    print(f"Posterior Standard Deviation: {posterior_sigma:.2f} seconds")

    # Hierarchical Bayesian example
    global_race_mu = 300
    global_race_sigma = 20
    observed_times = np.array([305, 290, 355])
    observed_race_variability = np.array([15, 18, 12])
    time_since_race = np.array([1, 5, 19])

    runner_posterior_mean, runner_posterior_sigma = hierarchical_bayesian_model(global_race_mu, global_race_sigma, observed_times, observed_race_variability, time_since_race)
    print(f"Final Runner-Specific Posterior Mean: {runner_posterior_mean:.2f} seconds")
    print(f"Final Runner-Specific Std Dev: {runner_posterior_sigma:.2f} seconds")
