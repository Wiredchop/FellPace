from fellpace.modelling.prediction import get_probability_distribution
from fellpace.analysis_tools import convert_Chase_ZScore_logs_avg
from matplotlib import pyplot as plt

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