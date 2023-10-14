from scipy.stats import zscore, percentileofscore
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from typing import Tuple
def calculate_position_stats(time: npt.ArrayLike) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
    """Calculate position stats, specifically the percentile AND zscore for position data TIME!!!
    It is useful to have both, the zscore assumes normality but has additional power at the tails of the distribution
    as it calculates how far ahead/behind someone may be from the average rather than just relative position within a group

    Args:
        time (np.ndarray): A 1D np array of time values in seconds, returned from a pandas dataframe

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the (zscore, percentile)
    """
    
    zscores = zscore(time,nan_policy = 'omit')
    
    percentiles = percentileofscore(time,time,nan_policy = 'omit')
    #Percentile rounding errors can occur, so enforcing 0 - 100
    percentiles[percentiles < 0] = 0
    percentiles[percentiles > 100] = 100
    
    return (zscores,percentiles)