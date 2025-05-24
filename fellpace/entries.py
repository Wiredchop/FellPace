from fellpace.FellPace_tools import get_table_from_URL
from fellpace.config import ENTRIES_PATH
from datetime import date
import pandas as pd

def get_entries_from_url(url: str, year_of_entry: int = date.today().year):
    """
    Get entries from a URL. Save out to csv
    
    It is intended that the csv will be manually tweaked after saving.
    
    Args:
        url (str): URL to get entries from.
        
    Returns:
        pd.DataFrame: DataFrame containing the entries.
    """
    data, _ = get_table_from_URL(url)
    save_path = ENTRIES_PATH / f"entries_{year_of_entry}.csv"
    data.to_csv(save_path, index=False)
    return data

def load_entries(year_of_entry: int = date.today().year):
    """
    Load entries from a csv file.
    
    Args:
        year_of_entry (int): Year of the entries to load.
        
    Returns:
        pd.DataFrame: DataFrame containing the entries.
    """
    save_path = ENTRIES_PATH / f"entries_{year_of_entry}.csv"
    if not save_path.exists():
        raise FileNotFoundError(f"Entries file for {year_of_entry} does not exist.")
    
    data = pd.read_csv(save_path)
    return data

if __name__ == "__main__":
    pass