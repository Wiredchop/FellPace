from fellpace.modelling.prediction import get_probability_distribution, make_chase_prediction
from fellpace.analysis_tools import convert_Chase_ZScore_logs_avg
from fellpace.convert_tools import seconds_to_time_string
from matplotlib import pyplot as plt
from fellpace.config import ENTRIES_PATH
from datetime import date
from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np
from loguru import logger

def plot_time_normal(con, mu: float, sigma: float, label: str, ax, convert_to_seconds: bool = True, **kwargs):
    """
    Plot a normal distribution with given mean and standard deviation.
    
    Args:
        mu (float): Mean of the normal distribution.
        sigma (float): Standard deviation of the normal distribution.
        label (str): Label for the plot.
        ax: Matplotlib axis to plot on.
        convert_to_seconds (bool): Whether to convert x-axis values to seconds.
        **kwargs: Additional keyword arguments for matplotlib plot function.
    """
    # Some fitted races can produce zero/invalid uncertainty; skip these safely
    # so plotting a full racer profile does not crash.
    if not np.isfinite(sigma) or sigma <= 0:
        logger.warning(f"Skipping plot for '{label}' due to non-positive sigma: {sigma}")
        return

    p = get_probability_distribution(mean=mu, std_dev=sigma)
    
    if convert_to_seconds:
        p.index = convert_Chase_ZScore_logs_avg(con, p.index)
    
    ax.plot(p.index, p.values, label=label, **kwargs)
    
    
def plot_racers_results(racer_results: pd.DataFrame, con: sqlite3.Connection, linestyle: str = '-', ax=None, save_path: str = None, year: int = None):
    """
    Plot all racer results as normal distributions.
    Optionally save the plot to a file if save_path is provided.
    
    Args:
        year (int): Year for calculating recency decay. Defaults to current year if not provided.
    """
    if year is None:
        year = date.today().year
    
    for _, result in racer_results.iterrows():
        season = result['Season']
        race = result['Race_Name']
        age = max(0, (year - 1) - season)
        plot_time_normal(con, result['Zpred_mu'], result['Zpred_sig'], f'{race}: {season}', ax, alpha = 1/(1 + age), linestyle=linestyle)

        
        
def plot_racer_entry(
    con: sqlite3.Connection,
    racer_results: pd.DataFrame,
    chase_mu: float,
    chase_sig: float,
    prediction_t: float,
    racer_name: str,
    prediction_year: int = date.today().year,
    save_figure: bool = False
):
    """
    Plot the results of a single racer.
    
    If the racer_results DataFrame does not contain an 'include' column, all races will be plotted in 
    the same style.
    
    Args:
        con (sqlite3.Connection): Database connection.
        racer_results (pd.DataFrame): DataFrame containing the racer's results.
        chase_mu (float): Mean of the chase prediction.
        chase_sig (float): Standard deviation of the chase prediction.
        prediction_t (float): Predicted time for the racer.
        prediction_year (int): Year of the prediction, defaults to current year.
        racer_name (str): Name of the racer.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if 'include' in racer_results.columns:
        included_results = racer_results[racer_results['include']]
        excluded_results = racer_results[~racer_results['include']]
        plot_racers_results(included_results, con, ax=ax, linestyle='-', year=prediction_year)
        plot_racers_results(excluded_results, con, ax=ax, linestyle=':', year=prediction_year)
    else:
        plot_racers_results(racer_results, con, ax=ax, linestyle='-', year=prediction_year)
    
    
    plot_time_normal(con, chase_mu, chase_sig, 'Chase Prediction',ax, color='black', linewidth=2)
    
    plt.vlines(prediction_t, 0, 0.2, color='black', linestyle='--', label='Predicted time')
    
    plt.xlabel("Predicted Time")
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: seconds_to_time_string(x)))  # Format xticks
    plt.title(f"Results for {racer_name}")
    plt.legend()    
    if save_figure:
        save_path = ENTRIES_PATH / f'predictions_{prediction_year}'
        if not save_path.exists():
            save_path.mkdir(parents=True)
        plt.savefig(save_path / f'{racer_name}.png')
    return fig, ax


def generate_racer_prediction_plot(
    con: sqlite3.Connection,
    model_tuple: tuple,
    racer_id: int,
    racer_name: str,
    year: int = date.today().year,
    output_dir: Path = None,
    display: bool = False
):
    """
    Generate a racer's prediction plot with flexible output handling.
    
    This is the core abstracted function used for generating racer prediction plots.
    It handles data retrieval, plot generation, and optional saving/display.
    
    Args:
        con (sqlite3.Connection): Database connection.
        model_tuple (tuple): Tuple of (coeffs, covar, resid_stds) preloaded models.
        racer_id (int): The racer's ID.
        racer_name (str): The racer's name.
        year (int): Year for predictions. Defaults to current year.
        output_dir (Path): Directory to save the plot. If None, plot is not saved.
        display (bool): Whether to display the plot with plt.show(). Defaults to False.
    
    Returns:
        bool: True if plot was generated successfully, False otherwise.
    """
    try:
        # Import here to avoid circular import: entries -> plotting.racetimes
        from fellpace.entries import process_results_for_racer

        coeffs, covar, resid_stds = model_tuple
        
        racer_results, _ = process_results_for_racer(
            con,
            coeffs,
            covar,
            resid_stds=resid_stds,
            racer_id=racer_id,
            prediction_year=year,
        )
        if racer_results.empty:
            logger.warning(f"No valid results found for {racer_name}.")
            return False
        
        chase_mu, chase_sig = make_chase_prediction(
            racer_results.loc[racer_results['include']],
            prediction_year=year,
            verbose=False,
        )
        
        prediction = chase_mu - (1.96 * chase_sig)
        prediction_t = convert_Chase_ZScore_logs_avg(con, prediction)
        
        # Generate the plot using the shared plot function
        plot_racer_entry(
            con, 
            racer_results, 
            chase_mu, 
            chase_sig, 
            prediction_t, 
            racer_name, 
            prediction_year=year,
            save_figure=False
        )
        
        # Handle saving if output directory is provided
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)
            filename = f"{racer_name.replace(' ', '_')}_{year}_prediction.png"
            filepath = output_dir / filename
            plt.savefig(filepath, dpi=100, bbox_inches='tight')
            logger.info(f"Saved plot for {racer_name} to {filepath}")
        
        # Handle display if requested
        if display:
            plt.show()
        else:
            plt.close()
        
        return True
        
    except Exception as e:
        logger.error(f"Error generating plot for {racer_name}: {e}")
        return False