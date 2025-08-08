from fellpace.FellPace_tools import get_table_from_URL
from fellpace.modelling.training import load_models
from fellpace.modelling.prediction import get_racers_results, get_prediction_with_uncertainty_many, make_chase_prediction, get_prediction_from_parkrun_time
from fellpace.extract.racers import secure_racer_id
from fellpace.analysis_tools import identify_outliers_in_predictions, convert_Chase_ZScore_logs_avg
from fellpace.filter import filter_race_results
from fellpace.extract.chase import get_previous_chase_results, extract_result_for_year
from fellpace.convert_tools import seconds_to_time_string
from fellpace.plotting.racetimes import plot_racer_entry

from fellpace.config import ENTRIES_PATH
from datetime import date
import pandas as pd
from tabulate import tabulate

from sqlite3 import Connection
from loguru import logger
import re

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

def load_PR_entries(year_of_entry: int = date.today().year):
    """
    Load PR entries from a csv file.
    
    Args:
        year_of_entry (int): Year of the entries to load.
        
    Returns:
        pd.DataFrame: DataFrame containing the PR entries.
    """
    save_path = ENTRIES_PATH / f"PR_{year_of_entry}.csv"
    if not save_path.exists():
        raise FileNotFoundError(f"PR entries file for {year_of_entry} does not exist.")
    
    data = pd.read_csv(save_path)
    
    return data


def process_results_for_racer(con: Connection, racer_name: str, coeffs, covar) -> pd.DataFrame:
    """
    Process results for a single racer to separate them into results to use in prediction and excluded results.
    
    Args:
        racer_results (pd.DataFrame): DataFrame containing the racer's results.
        coeffs (pd.Series): Coefficients for the model.
        covar (pd.Series): Covariance matrix for the model.
        
    Returns:
       Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
           - pd.DataFrame: The results to use in prediction.
           - pd.DataFrame: The excluded results.
    """
    racer_id = secure_racer_id(con, racer_name.lower().strip())
    if racer_id is None:
        logger.warning(f"Racer {racer_name} not found in database.")
        return None, None
    racer_results = get_racers_results(con, racer_id)
    if racer_results.empty:
        logger.warning(f"{racer_name} has not run in any valid races.")
        return racer_results, pd.DataFrame()  # No results to process
    racer_results_with_predictions = get_prediction_with_uncertainty_many(coeffs, covar, racer_results)
    racer_results_with_predictions['outlier'] = identify_outliers_in_predictions(racer_results_with_predictions['Zpred_mu'], threshold=1.2)
    racer_results, excluded_results = filter_race_results(racer_results_with_predictions)
    return racer_results, excluded_results

def process_entries(entries: pd.DataFrame, con: Connection, with_parkrun: bool = False) -> pd.DataFrame:
    """
    Process entries DataFrame to get predicted times and previous results.
    
    Args:
        entries (pd.DataFrame): DataFrame containing the entries.
        
    Returns:
        pd.DataFrame: Processed DataFrame with necessary columns.
    """
    coeffs, covar = load_models()
    processed_entries = pd.DataFrame()
    this_year = date.today().year
    for i, entry in entries.iterrows():
        racer_name = entry['Name']
        logger.info(f"Processing entry for {racer_name}")
        
        if with_parkrun:
            PR_time = entry.get('PR_time', None)
            if PR_time is not None:
                pr_prediction_t = get_prediction_from_parkrun_time(con, PR_time, coeffs['PR_Endcliffe'], covar['PR_Endcliffe'])
                logger.info(f"PR time for {racer_name}: {seconds_to_time_string(pr_prediction_t)}")
                pr_prediction_str = seconds_to_time_string(pr_prediction_t)
            else:
                pr_prediction_str = "N/A"
                
        racer_results, excluded_results = process_results_for_racer(con, racer_name, coeffs, covar)
        racer_id = secure_racer_id(con, racer_name.lower().strip())
        if racer_results is None:
            logger.warning(f"Creating blank entry for {racer_name} as racer not found.")
            entry_series = pd.Series({
            'Name': racer_name,
            'Num_results_used': 0,
            'Num_excluded_results': 0,
            'Predicted_Time': "N/A",
            'Given PR time': pr_prediction_str,
            f'Chase {this_year-1}': "N/A",
            f'Chase {this_year-2}': "N/A",
            f'Chase {this_year-3}': "N/A"
            })
            processed_entries = pd.concat([processed_entries, entry_series.to_frame().T], ignore_index=True)
            continue
       
        logger.info(f"Including {len(racer_results)} in calculation:\n {tabulate(racer_results, headers='keys', tablefmt='rounded_outline')}")
            
        if not excluded_results.empty:
            logger.info(f"Excluded results:\n {tabulate(excluded_results, headers='keys', tablefmt='rounded_outline')}")
        
        if racer_results.empty:
            prediction_str = 'N/A'
        else:
            chase_mu, chase_sig = make_chase_prediction(racer_results, verbose=True)   
            prediction = chase_mu - (1.96 * chase_sig)
            prediction_t = convert_Chase_ZScore_logs_avg(con, prediction)[0]
            prediction_str = seconds_to_time_string(prediction_t)
                
            plot_racer_entry(con, racer_results, excluded_results,chase_mu, chase_sig, prediction_t, racer_name, prediction_year=this_year)

        chases_results = get_previous_chase_results(con, racer_id)
        
        # Create a series for this racer's entry
        # Add last three years of chase results too
        
        
        entry_series = pd.Series({
            'Name': racer_name,
            'Num_results_used': len(racer_results),
            'Num_excluded_results': len(excluded_results),
            'Predicted_Time': prediction_str,
            'Given PR time': pr_prediction_str,
            f'Chase {this_year-1}': extract_result_for_year(chases_results, this_year - 1),
            f'Chase {this_year-2}': extract_result_for_year(chases_results, this_year - 2),
            f'Chase {this_year-3}': extract_result_for_year(chases_results, this_year - 3),
        })
        processed_entries = pd.concat([processed_entries, entry_series.to_frame().T], ignore_index=True)
        

    logger.info(f"Processed entries:\n {tabulate(processed_entries, headers='keys', tablefmt='rounded_outline')}")
    filepath = ENTRIES_PATH / f"processed_entries_{date.today().year}.csv"
    processed_entries.to_csv(filepath, index=False)
    return processed_entries
  
def clean_pr_time_column(df, time_col='PR_time', new_col='PR_time_clean'):
    """
    Clean and standardize a time column (e.g., 'PR_time') in a DataFrame to MM:SS format.
    Non-parseable times are set to None.
    """
    def extract_time(val):
        if pd.isnull(val):
            return None
        s = str(val).lower()
        # Try HH:MM:SS
        match = re.search(r'(\d{1,2}):(\d{2}):(\d{2})', s)
        if match:
            hours, minutes, seconds = match.groups()
            total_minutes = int(hours) * 60 + int(minutes)
            return f"{total_minutes:02d}:{int(seconds):02d}"
        # Try MM:SS or MM.SS
        match = re.search(r'(\d{1,2})[:\.](\d{2})', s)
        if match:
            minutes, seconds = match.groups()
            return f"{int(minutes):02d}:{int(seconds):02d}"
        # Try MM min SS sec
        match = re.search(r'(\d{1,2})\s*min(?:ute)?s?\s*(\d{1,2})\s*sec(?:ond)?s?', s)
        if match:
            minutes, seconds = match.groups()
            return f"{int(minutes):02d}:{int(seconds):02d}"
        # Try MM min
        match = re.search(r'(\d{1,2})\s*min(?:ute)?s?', s)
        if match:
            minutes = match.group(1)
            return f"{int(minutes):02d}:00"
        # Try MM.SS (e.g., 25.00)
        match = re.search(r'(\d{1,2})\.(\d{2})', s)
        if match:
            minutes, seconds = match.groups()
            return f"{int(minutes):02d}:{int(seconds):02d}"
        # Try MM (whole minutes)
        match = re.fullmatch(r'(\d{1,2})', s.strip())
        if match:
            minutes = match.group(1)
            return f"{int(minutes):02d}:00"
        return None

    return df[time_col].apply(extract_time)

def process_PR_entries(PR_entries: pd.DataFrame, year_of_entry: int = date.today().year, forename_surname: bool = True, forename_column: str = 'First Name', surname_column: str = 'Surname'):
    """
    Process PR entries for a given year, clean the PR_time column, and save results.
    
    Args:
        year_of_entry (int): Year of the entries to process.
        
    Returns:
        pd.DataFrame: Processed DataFrame with cleaned PR_time.
    """
    if 'PR_time' not in PR_entries.columns:
        raise ValueError("The DataFrame must contain a 'PR_time' column.")
    
    if forename_surname:
        PR_entries['Name'] = PR_entries[forename_column].str.lower() + ' ' + PR_entries[surname_column].str.lower()
        
    PR_entries['PR_time'] = clean_pr_time_column(PR_entries, time_col='PR_time')
    
    logger.info(tabulate(PR_entries, headers='keys', tablefmt='rounded_outline'))
    

    return PR_entries
    
if __name__ == "__main__":

    from fellpace.db.db_setup import setup_db
    from fellpace.config import DB_PATH, ENTRIES_PATH
    con = setup_db(DB_PATH)
    # Path to your CSV file (adjust as needed)
    csv_path = ENTRIES_PATH / "PR_2025.csv"
    
    PR_entries = load_PR_entries(year_of_entry=2025)
    PR_entries_processed = process_PR_entries(PR_entries, year_of_entry=2025, forename_surname=True)
    process_entries(PR_entries_processed, con, with_parkrun=True)


