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

def prepare_chase_results_for_prediction(chase_results: pd.DataFrame, racer_name: str) -> pd.DataFrame:
    """
    Prepare chase results for prediction.
    
    Args:
        chase_results (pd.DataFrame): DataFrame containing chase results.
        
    Returns:
        pd.DataFrame: Prepared DataFrame with necessary columns.
    """
    chase_results['Zpred_sig'] = 0.01  # Assuming very low uncertainty for chase results
    chase_results['ZScore'] = chase_results['Zpred_mu']  # ZScore is the same as Zpred_mu for chase results
    chase_results['Racer_Name'] = racer_name  # Assuming all chase results are for the same racer
    chase_results[['Racer_ID', 'Racer_Name','Race_Name', 'Season', 'ZScore', 'Zpred_mu', 'Zpred_sig']]

def combine_results_with_chase_results(racer_results: pd.DataFrame, chase_results: pd.DataFrame) -> pd.DataFrame:
    """
    Combine racer results with chase results.
    
    Make assumptions of a very low uncertainty for the chase results as it's a recorded time.
    
    Args:
        racer_results (pd.DataFrame): DataFrame containing the racer's results.
        chase_results (pd.DataFrame): DataFrame containing the chase results.
        
    Returns:
        pd.DataFrame: Combined DataFrame with both racer and chase results.
    """
    
    prepared_chase_results = prepare_chase_results_for_prediction(chase_results, racer_name=racer_results['Racer_Name'].iloc[0])
    
    return pd.concat(
        [
            racer_results,
            prepared_chase_results
        ],
        ignore_index=True)

def process_results_for_racer(con: Connection,coeffs, covar, racer_name: str = None, racer_id = None) -> pd.DataFrame:
    """
    Process results for a single racer to separate them into results to use in prediction and excluded results.
    
    Provide either racer_name or racer_id, not both.
    Args:
        racer_results (pd.DataFrame): DataFrame containing the racer's results.
        coeffs (pd.Series): Coefficients for the model.
        covar (pd.Series): Covariance matrix for the model.
        racer_name (str): Name of the racer. Defaults to None.
        racer_id (int): ID of the racer. Defaults to None.
        
    Returns:
       Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
           - pd.DataFrame: The results to use in prediction.
           - pd.DataFrame: The excluded results.
    """
    assert (racer_id is not None) ^ (racer_name is not None), "Provide either racer_id or racer_name, not both."
    if racer_name:
        racer_id = secure_racer_id(con, racer_name.lower().strip())
    chase_results = get_previous_chase_results(con, racer_id)
    racer_results = get_racers_results(con, racer_id)
    
    if racer_results.empty:
        logger.warning(f"{racer_name} has not run in any valid races.")
        # Assuming racer in DB due to running in the chase only. 
        return prepare_chase_results_for_prediction(chase_results, racer_name), chase_results  
    
    racer_results_with_predictions = get_prediction_with_uncertainty_many(coeffs, covar, racer_results)
    
    if chase_results.empty:
        logger.warning(f"{racer_name} has no chase results.")
        all_results = racer_results_with_predictions
    else:
        all_results = combine_results_with_chase_results(racer_results_with_predictions, chase_results)
    all_results['outlier'] = identify_outliers_in_predictions(all_results['Zpred_mu'], threshold=1.2)
    filter_race_results(all_results)
    return all_results, chase_results

def process_entries(entries: pd.DataFrame, con: Connection,year_of_entry: int, with_parkrun: bool = False, plot: bool = False) -> pd.DataFrame:
    """
    Process entries DataFrame to get predicted times and previous results.
    
    Args:
        entries (pd.DataFrame): DataFrame containing the entries.
        
    Returns:
        pd.DataFrame: Processed DataFrame with necessary columns.
    """
    coeffs, covar = load_models()
    processed_entries = pd.DataFrame()
    all_racer_results = pd.DataFrame()
    all_racer_predictions = pd.DataFrame()
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
        
        racer_id = secure_racer_id(con, racer_name.lower().strip())        
        if racer_id is None:
            logger.warning(f"Racer {racer_name} not found in database.")
            racer_results = None
            excluded_results = None
            chase_results = None
        else:

            racer_results, chase_results = process_results_for_racer(con, coeffs, covar, racer_id = racer_id)
        
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
        all_racer_results = pd.concat([all_racer_results, racer_results], ignore_index=True)
            
        if (~racer_results['include']).any():
            logger.info(f"Excluded results:\n {tabulate(racer_results.loc[~racer_results['include']], headers='keys', tablefmt='rounded_outline')}")

        if racer_results.empty:
            prediction_str = 'N/A'
        else:
            # Subtracting 1 from the year of entry as year before race is most recent possible
            chase_mu, chase_sig = make_chase_prediction(racer_results.loc[racer_results['include']],prediction_year = year_of_entry,  verbose=True)   
            prediction = chase_mu - (1.96 * chase_sig)
            prediction_t = convert_Chase_ZScore_logs_avg(con, prediction)[0]
            prediction_str = seconds_to_time_string(prediction_t)
            all_racer_predictions = pd.concat(
                [
                    all_racer_predictions,
                    pd.DataFrame(
                        [{
                            'Racer_Name': racer_name,
                            'chase_mu': chase_mu,
                            'chase_sig': chase_sig
                        }]
                    )
                ],
                ignore_index=True
            )

            if plot:
                plot_racer_entry(con, racer_results, excluded_results,chase_mu, chase_sig, prediction_t, racer_name, prediction_year=this_year)

        # Create a series for this racer's entry
        # Add last three years of chase results too
        
        
        entry_series = pd.Series({
            'Name': racer_name,
            'Num_results_used': len(racer_results.loc[racer_results['include']]),
            'Num_excluded_results': len(racer_results.loc[~racer_results['include']]),
            'Predicted_Time': prediction_str,
            'Given PR time': pr_prediction_str,
            f'Chase {this_year-1}': extract_result_for_year(chase_results, this_year - 1),
            f'Chase {this_year-2}': extract_result_for_year(chase_results, this_year - 2),
            f'Chase {this_year-3}': extract_result_for_year(chase_results, this_year - 3),
        })
        processed_entries = pd.concat([processed_entries, entry_series.to_frame().T], ignore_index=True)
        
        

    logger.info(f"Processed entries:\n {tabulate(processed_entries, headers='keys', tablefmt='rounded_outline')}")
    entries_filepath = ENTRIES_PATH / f"processed_entries_{year_of_entry}.csv"
    results_filepath = ENTRIES_PATH / f"racer_results_{year_of_entry}.json"
    predictions_filepath = ENTRIES_PATH / f"racer_predictions_{year_of_entry}.json"
    
    processed_entries.to_csv(entries_filepath, index=False)
    all_racer_results.to_json(results_filepath, index=False, indent=4)
    all_racer_predictions.to_json(predictions_filepath, index=False, indent=4)
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
    process_entries(PR_entries_processed, con, year_of_entry=2025, with_parkrun=True)


