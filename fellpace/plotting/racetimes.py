from fellpace.modelling.prediction import get_probability_distribution
from fellpace.analysis_tools import convert_Chase_ZScore_logs_avg
from fellpace.convert_tools import seconds_to_time_string
from matplotlib import pyplot as plt
from fellpace.config import ENTRIES_PATH
from datetime import date
import sqlite3
import pandas as pd

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
    p = get_probability_distribution(mean=mu, std_dev=sigma)
    
    if convert_to_seconds:
        p.index = convert_Chase_ZScore_logs_avg(con, p.index)
    
    ax.plot(p.index, p.values, label=label, **kwargs)
    
    
def plot_racers_results(racer_results: pd.DataFrame, con: sqlite3.Connection, linestyle: str = '-', ax=None, save_path: str = None):
    """
    Plot all racer results as normal distributions.
    Optionally save the plot to a file if save_path is provided.
    """
    for _, result in racer_results.iterrows():
        season = result['Season']
        race = result['Race_Name']
        plot_time_normal(con, result['Zpred_mu'], result['Zpred_sig'], f'{race}: {season}', ax, alpha = 1/(1+2024-season), linestyle=linestyle)

        
        
def plot_racer_entry(con: sqlite3.Connection, racer_results: pd.DataFrame, excluded_results: pd.DataFrame,chase_mu, chase_sig, prediction_t, racer_name: str, prediction_year: int = date.today().year):
    """
    Plot the results of a single racer.
    
    Args:
        racer_results (pd.DataFrame): DataFrame containing the racer's results.
        excluded_results (pd.DataFrame): DataFrame containing the excluded results.
        racer_name (str): Name of the racer.
    """
    _, ax = plt.subplots(figsize=(10, 6))
    
    plot_racers_results(racer_results,con, ax=ax, linestyle='-')
    plot_racers_results(excluded_results,con, ax=ax, linestyle=':')
    
    
    plot_time_normal(con, chase_mu, chase_sig, 'Chase 2024',ax, color='black', linewidth=2)
    
    plt.vlines(prediction_t, 0, 0.2, color='black', linestyle='--', label='Predicted time')
    
    plt.xlabel("Predicted Time")
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: seconds_to_time_string(x)))  # Format xticks
    plt.title(f"Results for {racer_name}")
    plt.legend()    
    save_path = ENTRIES_PATH / f'predictions_{prediction_year}'
    if not save_path.exists():
        save_path.mkdir(parents=True)
    plt.savefig(save_path / f'{racer_name}.png')