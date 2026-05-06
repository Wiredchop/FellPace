"A module to filter any data according to specific criteria. For example, if it's an outlier or if it's an un-needed parkrun entry."

from fellpace.config import EXCLUDE_LIST
from fellpace.race_overrides import _load_overrides, build_override_key
import pandas as pd

def filter_race_results(racer_results: pd.DataFrame) -> None:
    """Create an 'Included' column for race results.
    
    A race is included/excluded if:
    
    1. Any race names that are in an exclusion list (can be tweaked in the config file)
    2. If we have more than three results that are NOT parkrun, remove any parkrun results.
    3. If we are using outliers, remove any results that are outliers.
    4. A manual override keyed by (Racer_ID or Racer_Name, Season, Race_Name) stored in
       DB/race_include_overrides.json takes absolute precedence over all automatic rules.

    Args:
        racer_results (_type_): _description_
    """
    
    if len(EXCLUDE_LIST) > 0:
        exclude_mask = ~racer_results['Race_Name'].isin(EXCLUDE_LIST)
    else:
        exclude_mask = pd.Series([True] * len(racer_results), index=racer_results.index)
    
    parkrun_mask = ~racer_results['Race_Name'].str.contains('PR_')
    given_pr_rows = racer_results['Race_Name'] == 'PR_given'
    if parkrun_mask.sum() < 3:
        # Only include if enough other races
        parkrun_mask = pd.Series([True] * len(racer_results), index=racer_results.index)
    else:
        # Always retain the explicitly given PR regardless of how many fell races exist
        parkrun_mask = parkrun_mask | given_pr_rows
        
    if racer_results.columns.str.contains('outlier').any():
        outlier_mask = ~racer_results['outlier']
    else:
        outlier_mask = pd.Series([True] * len(racer_results), index=racer_results.index)
        
    final_mask = exclude_mask & parkrun_mask & outlier_mask

    # Apply manual overrides: a JSON entry for a composite key takes full precedence.
    overrides = _load_overrides()
    if overrides:
        for idx, row in racer_results.iterrows():
            key = build_override_key(row)
            if key is not None and key in overrides:
                final_mask.at[idx] = bool(overrides[key])

    racer_results['include'] = final_mask