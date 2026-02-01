"""A module to calculate handicaps to give each racer."""

from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from fellpace.convert_tools import seconds_to_time_string
from fellpace.config import START_TIME

def calculate_handicaps_for_entries(processed_entries: pd.DataFrame) -> pd.DataFrame:
    """Calculate handicaps for each entry in the provided DataFrame.
    
    Handicap raw is the raw difference in seconds.
    
    Handicap seconds is rounded up to the nearest 5 seconds to make feasible in race conditions.

    Args:
        process_entries (pd.DataFrame): DataFrame containing entries to process.
    Returns:
        pd.DataFrame: DataFrame with calculated handicaps.
        """
    
    # Create mask for valid predictions
    valid_mask = processed_entries['Predicted_Time_seconds'].notna()
        
    processed_entries_sorted = processed_entries.sort_values(by='Predicted_Time_seconds', ascending=False)
    
    # Only calculate handicaps for entries with valid predictions
    if valid_mask.any():
        max_time = processed_entries_sorted.loc[valid_mask, 'Predicted_Time_seconds'].max()
        processed_entries_sorted.loc[valid_mask, 'Handicap_seconds_raw'] = (
            max_time - processed_entries_sorted.loc[valid_mask, 'Predicted_Time_seconds']
        )
        processed_entries_sorted.loc[valid_mask, 'Handicap_seconds'] = (
            np.ceil(processed_entries_sorted.loc[valid_mask, 'Handicap_seconds_raw']/5) * 5
        )
        processed_entries_sorted.loc[valid_mask, 'Handicap'] = (
            processed_entries_sorted.loc[valid_mask, 'Handicap_seconds'].apply(seconds_to_time_string)
        )
        
        start_time = datetime.strptime(START_TIME, "%H:%M:%S")
        processed_entries_sorted.loc[valid_mask, 'Off_time'] = (
            processed_entries_sorted.loc[valid_mask, 'Handicap_seconds'].apply(
                lambda x: (start_time + timedelta(seconds=x)).time().strftime("%H:%M:%S")
            )
        )
    
    return processed_entries_sorted
    