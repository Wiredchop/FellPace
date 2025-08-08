"A module to filter any data according to specific criteria. For example, if it's an outlier or if it's an un-needed parkrun entry."

from fellpace.config import EXCLUDE_LIST
import pandas as pd

def filter_race_results(racer_results: pd.DataFrame) -> pd.DataFrame:
    """Filter races out of results if we think they should not be included in the analysis.
    
    1. Any race names that are in an exclusion list (can be tweaked in the config file)
    2. If we have more than three results that are NOT parkrun, remove any parkrun results.
    3. If we are using outliers, remove any results that are outliers.

    Args:
        racer_results (_type_): _description_
    """
    
    if len(EXCLUDE_LIST) > 0:
        exclude_mask = ~racer_results['Race_Name'].isin(EXCLUDE_LIST)
    else:
        exclude_mask = pd.Series([True] * len(racer_results), index=racer_results.index)
    
    parkrun_mask = ~racer_results['Race_Name'].str.contains('PR_')    
    if parkrun_mask.sum() < 3:
        # Only include if enough other races
        parkrun_mask = pd.Series([True] * len(racer_results), index=racer_results.index)
        
    if racer_results.columns.str.contains('outlier').any():
        outlier_mask = ~racer_results['outlier']
    else:
        outlier_mask = pd.Series([True] * len(racer_results), index=racer_results.index)
        
    final_mask = exclude_mask & parkrun_mask & outlier_mask
    return racer_results[final_mask].reset_index(drop=True), racer_results[~final_mask].reset_index(drop=True)