from fellpace.FellPace_tools import get_table_from_URL
from fellpace.modelling.training import load_models, load_time_models, load_road_time_models
from fellpace.modelling.prediction import get_racers_results, get_chase_z_from_time, get_prediction_with_uncertainty_many, make_chase_prediction, get_prediction_time_from_parkrun_time, get_prediction_from_parkrun_time
from fellpace.extract.racers import secure_racer_id
from fellpace.analysis_tools import identify_outliers_in_predictions, convert_Chase_ZScore_logs_avg
from fellpace.filter import filter_race_results
from fellpace.extract.chase import get_previous_chase_results, extract_result_for_year
from fellpace.convert_tools import seconds_to_time_string
from fellpace.plotting.racetimes import plot_racer_entry
from fellpace.p10.scraper import get_racer_p10_results_for_prediction

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

def prepare_p10_results_for_prediction(
    p10_results: pd.DataFrame,
    racer_name: str,
    con,
    time_coeffs
) -> pd.DataFrame:
    """
    Prepare Power of 10 results for prediction.

    Converts each yearly best time into a z-score prediction and
    uncertainty using the delta method via get_prediction_from_parkrun_time. This gives
    the same (Zpred_mu, Zpred_sig) structure as road race predictions so that all inputs to the Bayesian combiner are on an equal footing.
    
    Args:
        p10_results: DataFrame containing the Power of 10 results, must contain
                    'best_time_seconds' and 'year' columns.
        racer_name: Name of the racer, added as 'Racer_Name' column.
        con: Active SQLite connection (for time model stats in delta method).
        time_coeffs: Loaded time models from load_time_models().
    Returns:
        pd.DataFrame with columns:
            Racer_ID, Racer_Name, Race_Name, Season, ZScore, Zpred_mu, Zpred_sig
    """
    zpred = p10_results.apply(
        lambda r: get_chase_z_from_time(
            r['Time'],
            r['Race_Name'],
            time_coeffs,
            con,
        ),
        axis=1,
    )
    p10_results = p10_results.copy()
    p10_results['Zpred_mu'] = zpred.apply(lambda x: x[0])
    p10_results['Zpred_sig'] = zpred.apply(lambda x: x[1])
    p10_results['ZScore'] = p10_results['Zpred_mu']
    p10_results['Racer_Name'] = racer_name
    return p10_results[['Racer_ID', 'Racer_Name', 'Race_Name', 'Season', 'ZScore', 'Zpred_mu', 'Zpred_sig']]

def prepare_chase_results_for_prediction(
    chase_results: pd.DataFrame,
    racer_name: str,
    con,
    time_coeffs,
    prediction_year: int = None,
) -> pd.DataFrame:
    """
    Prepare chase results for prediction.

    Converts each previous Chase finishing time into a z-score prediction and
    uncertainty using the delta method via get_chase_z_from_time. This gives
    the same (Zpred_mu, Zpred_sig) structure as road race predictions so that
    all inputs to the Bayesian combiner are on an equal footing.

    Args:
        chase_results: DataFrame from get_previous_chase_results(), must contain
                       'Time' (seconds) and 'Racer_ID' columns.
        racer_name:    Name of the racer, added as 'Racer_Name' column.
        con:           Active SQLite connection (for Chase log stats in delta method).
        time_coeffs:   Loaded time models from load_time_models().
        prediction_year: Year being predicted. If provided, uses:
            - previous_chase model for Season == prediction_year - 1
            - older_chase model otherwise

    Returns:
        pd.DataFrame with columns:
            Racer_ID, Racer_Name, Race_Name, Season, ZScore, Zpred_mu, Zpred_sig
    """
    def _model_for_row(season: int) -> str:
        if prediction_year is not None and season == (prediction_year - 1):
            return 'previous_chase'
        return 'older_chase'

    zpred = chase_results.apply(
        lambda r: get_chase_z_from_time(
            r['Time'],
            _model_for_row(r['Season']),
            time_coeffs,
            con,
        ),
        axis=1,
    )
    chase_results = chase_results.copy()
    chase_results['Zpred_mu'] = zpred.apply(lambda x: x[0])
    chase_results['Zpred_sig'] = zpred.apply(lambda x: x[1])
    chase_results['ZScore'] = chase_results['Zpred_mu']
    chase_results['Racer_Name'] = racer_name
    return chase_results[['Racer_ID', 'Racer_Name', 'Race_Name', 'Season', 'ZScore', 'Zpred_mu', 'Zpred_sig']]

def combine_results_with_chase_results(
    racer_results: pd.DataFrame,
    chase_results: pd.DataFrame,
    con,
    time_coeffs,
    prediction_year: int = None,
) -> pd.DataFrame:
    """
    Combine racer results with chase results.

    Previous Chase results are converted to z-score predictions using the
    time-domain delta method, giving realistic uncertainty rather than the
    former hardcoded near-zero value.

    Args:
        racer_results (pd.DataFrame): DataFrame containing the racer's results.
        chase_results (pd.DataFrame): DataFrame containing the chase results.
        con:           Active SQLite connection (passed through to delta method).
        time_coeffs:   Loaded time models from load_time_models().
        prediction_year: Year being predicted, used to choose
            previous_chase vs older_chase model per row.

    Returns:
        pd.DataFrame: Combined DataFrame with both racer and chase results.
    """
    prepared_chase_results = prepare_chase_results_for_prediction(
        chase_results,
        racer_name=racer_results['Racer_Name'].iloc[0],
        con=con,
        time_coeffs=time_coeffs,
        prediction_year=prediction_year,
    )
    return pd.concat([racer_results, prepared_chase_results], ignore_index=True)

def combine_results_with_p10_results(
    racer_results: pd.DataFrame,
    p10_results: pd.DataFrame,
    con,
    road_time_coeffs,
) -> pd.DataFrame:
    """
    Combine racer results with Power of 10 results.

    Power of 10 yearly best results are converted to z-score predictions using the
    time-domain delta method, giving realistic uncertainty.

    Args:
        racer_results (pd.DataFrame): DataFrame containing the racer's results.
        p10_results (pd.DataFrame): DataFrame containing the Power of 10 results.
        con:           Active SQLite connection (passed through to delta method).
        road_time_coeffs: Loaded road time models from load_road_time_models().

    Returns:
        pd.DataFrame: Combined DataFrame with both racer and Power of 10 results.
    """
    prepared_p10_results = prepare_p10_results_for_prediction(
        p10_results,
        racer_name=racer_results['Racer_Name'].iloc[0],
        con=con,
        time_coeffs=road_time_coeffs,
    )
    return pd.concat([racer_results, prepared_p10_results], ignore_index=True)


def _build_given_pr_prediction_row(
    con,
    coeffs,
    covar,
    resid_stds,
    given_pr_time: str | None,
    racer_id,
    racer_name: str,
    prediction_year: int | None,
) -> pd.DataFrame:
    """Create a PR_given prediction row from submitted PR time."""
    if given_pr_time is None:
        return pd.DataFrame()

    try:
        pr_mean, pr_sig, pr_zscore = get_prediction_from_parkrun_time(
            con,
            given_pr_time,
            coeffs['PR_Endcliffe'],
            covar['PR_Endcliffe'],
            residual_std=float(resid_stds.get('PR_Endcliffe', 0.0)) if resid_stds is not None else 0.0,
        )
        season = (prediction_year - 1) if prediction_year is not None else (date.today().year - 1)
        return pd.DataFrame([{
            'Racer_ID': racer_id,
            'Racer_Name': racer_name,
            'Race_Name': 'PR_given',
            'Season': season,
            'ZScore': pr_zscore,
            'Zpred_mu': pr_mean,
            'Zpred_sig': pr_sig,
        }])
    except Exception as e:
        logger.warning(f"Could not incorporate given PR time for {racer_name}: {e}")
        return pd.DataFrame()


def _build_p10_prediction_rows(
    con,
    road_time_coeffs,
    racer_id,
    racer_name: str,
    po10_url: str | None,
    prediction_year: int,
) -> pd.DataFrame:
    """Create prepared prediction rows from Power of 10 URL."""
    if road_time_coeffs is None:
        return pd.DataFrame()
    if not (po10_url is not None and pd.notna(po10_url) and isinstance(po10_url, str) and po10_url.strip()):
        return pd.DataFrame()

    try:
        p10_results = get_racer_p10_results_for_prediction(
            racer_id=racer_id,
            racer_name=racer_name,
            athlete_url=po10_url,
            prediction_year=prediction_year,
        )
        if p10_results is None or p10_results.empty:
            return pd.DataFrame()

        p10_results = p10_results[p10_results['Season'] < prediction_year].reset_index(drop=True)
        if p10_results.empty:
            return pd.DataFrame()

        return prepare_p10_results_for_prediction(
            p10_results,
            racer_name=racer_name,
            con=con,
            time_coeffs=road_time_coeffs,
        )
    except Exception as e:
        logger.warning(f"Could not use P10 results for {racer_name}: {e}")
        return pd.DataFrame()

def process_results_for_racer(
    con: Connection,
    coeffs,
    covar,
    resid_stds: pd.Series = None,
    racer_name: str = None,
    racer_id=None,
    prediction_year: int = None,
    po10_url: str | None = None,
    given_pr_time: str | None = None,
) -> pd.DataFrame:
    """
    Process results for a single racer to separate them into results to use in prediction and excluded results.
    
    Provide either racer_name or racer_id, not both.
    Args:
        racer_results (pd.DataFrame): DataFrame containing the racer's results.
        coeffs (pd.Series): Coefficients for the model.
        covar (pd.Series): Covariance matrix for the model.
        racer_name (str): Name of the racer. Defaults to None.
        racer_id (int): ID of the racer. Defaults to None.
        po10_url (str): URL of the Power of 10 athlete page. Defaults to None.
        prediction_year (int): If provided, only include seasons before this year
            for both race and chase results.
        
    Returns:
       Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
           - pd.DataFrame: The results to use in prediction.
           - pd.DataFrame: The excluded results.
    """
    assert (racer_id is not None) ^ (racer_name is not None), "Provide either racer_id or racer_name, not both."
    if racer_name:
        input_racer_name = racer_name
        racer_id, resolved_name = secure_racer_id(con, racer_name.lower().strip())
        racer_name = resolved_name if racer_id is not None else input_racer_name
    else:
        racer_name = con.execute("SELECT Racer_Name FROM Racers WHERE Racer_ID = ?", (int(racer_id),)).fetchone()[0]
    if racer_id is None:
        logger.warning(f"Racer {racer_name} not found in database. Building inferred results from submitted PR/P10 data.")
        inferred_results = pd.concat([
            _build_given_pr_prediction_row(
                con=con,
                coeffs=coeffs,
                covar=covar,
                resid_stds=resid_stds,
                given_pr_time=given_pr_time,
                racer_id=-1,
                racer_name=racer_name,
                prediction_year=prediction_year,
            ),
            _build_p10_prediction_rows(
                con=con,
                road_time_coeffs=load_road_time_models(),
                racer_id=-1,
                racer_name=racer_name,
                po10_url=po10_url,
                prediction_year=prediction_year if prediction_year is not None else date.today().year,
            ),
        ], ignore_index=True)
        if inferred_results.empty:
            return pd.DataFrame(), pd.DataFrame()
        inferred_results['outlier'] = identify_outliers_in_predictions(inferred_results['Zpred_mu'], threshold=1.2)
        filter_race_results(inferred_results)
        return inferred_results, pd.DataFrame(columns=['Time', 'Season'])
    time_coeffs = load_time_models()
    road_time_coeffs = load_road_time_models()
    chase_results = get_previous_chase_results(con, racer_id)
    racer_results = get_racers_results(con, racer_id)
    p10_results = None
    if po10_url is not None and pd.notna(po10_url) and isinstance(po10_url, str) and po10_url.strip():
        try:
            p10_results = get_racer_p10_results_for_prediction(racer_id, racer_name, po10_url, prediction_year=prediction_year)
        except Exception as e:
            logger.warning(f"Failed to fetch Power of 10 results for {racer_name} from {po10_url}: {e}")
            p10_results = None
    else:
        p10_results = None


    if prediction_year is not None:
        racer_results, chase_results, p10_results = limit_results_to_requested_years(
            racer_results,
            chase_results,
            prediction_year,
            p10_results=p10_results,
        )
    
    if racer_results.empty:
        logger.warning(f"{racer_name} has not run in any valid races.")
        # Assuming racer in DB due to running in the chase only. Always including chase results
        return (
            prepare_chase_results_for_prediction(
                chase_results,
                racer_name,
                con=con,
                time_coeffs=time_coeffs,
                prediction_year=prediction_year,
            ).assign(include=True),
            chase_results
            )

    racer_results_with_predictions = get_prediction_with_uncertainty_many(coeffs, covar, racer_results, residual_stds=resid_stds)
    
    if chase_results.empty:
        logger.warning(f"{racer_name} has no chase results.")
        all_results = racer_results_with_predictions
    else:
        all_results = combine_results_with_chase_results(
            racer_results_with_predictions,
            chase_results,
            con=con,
            time_coeffs=time_coeffs,
            prediction_year=prediction_year,
        )
        
    if p10_results is not None and not p10_results.empty:
        if road_time_coeffs is None:
            logger.warning("Road time coefficients unavailable; skipping Power of 10 results.")
        else:
            all_results = combine_results_with_p10_results(
                all_results,
                p10_results,
                con=con,
                road_time_coeffs=road_time_coeffs,
            )

    given_pr_row = _build_given_pr_prediction_row(
        con=con,
        coeffs=coeffs,
        covar=covar,
        resid_stds=resid_stds,
        given_pr_time=given_pr_time,
        racer_id=racer_id,
        racer_name=racer_name,
        prediction_year=prediction_year,
    )
    if not given_pr_row.empty:
        all_results = pd.concat([all_results, given_pr_row], ignore_index=True)

    all_results['outlier'] = identify_outliers_in_predictions(all_results['Zpred_mu'], threshold=1.2)
    filter_race_results(all_results)

    return all_results, chase_results

def gather_results_for_entries(entries: pd.DataFrame, con: Connection, year_of_entry: int, with_parkrun: bool = False) -> pd.DataFrame:
    
    coeffs, covar, resid_stds = load_models(include_residuals=True)
    all_racer_results = pd.DataFrame()
    
    for i, entry in entries.iterrows():
        racer_name = entry['Name']
        logger.info(f"Processing entry for {racer_name}")
        
        if with_parkrun:
            PR_time = entry.get('PR_time', None)
        else:
            PR_time = None

        racer_id, racer_name = secure_racer_id(con, racer_name.lower().strip())
        racer_given_PR = _build_given_pr_prediction_row(
            con=con,
            coeffs=coeffs,
            covar=covar,
            resid_stds=resid_stds,
            given_pr_time=PR_time if (pd.notna(PR_time) and isinstance(PR_time, str) and ':' in PR_time) else None,
            racer_id=racer_id,
            racer_name=racer_name,
            prediction_year=year_of_entry,
        )
        if racer_id is None:
            # Just put name in dataframe with given parkrun time if they have one.
            logger.warning(f"Racer {racer_name} not found in database. Just adding given PR time.")
            all_racer_results = pd.concat([all_racer_results, racer_given_PR], ignore_index=True)
            continue
        racer_results, chase_results = process_results_for_racer(
            con,
            coeffs,
            covar,
            resid_stds=resid_stds,
            racer_id=racer_id,
            prediction_year=year_of_entry,
        )
        
        all_racer_results = pd.concat([all_racer_results, racer_given_PR ,racer_results], ignore_index=True)
        
        if racer_results is None:
            logger.warning(f"Racer {racer_name} has no valid results.")
            continue

    return all_racer_results

def process_entries(entries: pd.DataFrame, con: Connection,year_of_entry: int, with_parkrun: bool = False, plot: bool = False) -> pd.DataFrame:
    """
    Process entries DataFrame to get predicted times and previous results.
    
    Args:
        entries (pd.DataFrame): DataFrame containing the entries.
        
    Returns:
        pd.DataFrame: Processed DataFrame with necessary columns.
    """
    coeffs, covar, resid_stds = load_models(include_residuals=True)
    road_time_coeffs = load_road_time_models()
    processed_entries = pd.DataFrame()
    all_racer_results = pd.DataFrame()
    all_racer_predictions = pd.DataFrame()
    this_year = year_of_entry
    for i, entry in entries.iterrows():
        racer_name = entry['Name']
        input_racer_name = racer_name
        logger.info(f"Processing entry for {racer_name}")
        
        pr_prediction_str = "N/A"
        pr_time_for_model = None
        if with_parkrun:
            PR_time = entry.get('PR_time', None)
            if pd.notna(PR_time) and isinstance(PR_time, str) and ':' in PR_time:
                pr_time_for_model = PR_time
                pr_prediction_t = get_prediction_time_from_parkrun_time(
                    con,
                    PR_time,
                    year_of_entry,
                    coeffs['PR_Endcliffe'],
                    covar['PR_Endcliffe'],
                    residual_std=float(resid_stds.get('PR_Endcliffe', 0.0)),
                )
                logger.info(f"PR time for {racer_name}: {seconds_to_time_string(pr_prediction_t)}")
                pr_prediction_str = seconds_to_time_string(pr_prediction_t)
        
        racer_id, racer_name = secure_racer_id(con, racer_name.lower().strip())
        if racer_id is None:
            logger.warning(f"Racer {input_racer_name} not found in database. Building prediction from submitted PR/P10 data.")
            racer_name = input_racer_name
            chase_results = pd.DataFrame(columns=['Time', 'Season'])

            inferred_results = pd.DataFrame(columns=['Racer_ID', 'Racer_Name', 'Race_Name', 'Season', 'ZScore', 'Zpred_mu', 'Zpred_sig'])

            inferred_results = pd.concat([
                inferred_results,
                _build_given_pr_prediction_row(
                    con=con,
                    coeffs=coeffs,
                    covar=covar,
                    resid_stds=resid_stds,
                    given_pr_time=pr_time_for_model,
                    racer_id=-1 * (i + 1),
                    racer_name=racer_name,
                    prediction_year=year_of_entry,
                ),
                _build_p10_prediction_rows(
                    con=con,
                    road_time_coeffs=road_time_coeffs,
                    racer_id=-1 * (i + 1),
                    racer_name=racer_name,
                    po10_url=entry.get('URL', None),
                    prediction_year=year_of_entry,
                ),
            ], ignore_index=True)

            racer_results = inferred_results
            if not racer_results.empty:
                racer_results['outlier'] = identify_outliers_in_predictions(racer_results['Zpred_mu'], threshold=1.2)
                filter_race_results(racer_results)
        else:

            racer_results, chase_results = process_results_for_racer(
                con,
                coeffs,
                covar,
                resid_stds=resid_stds,
                racer_id=racer_id,
                prediction_year=year_of_entry,
                po10_url=entry.get('URL', None),
                given_pr_time=pr_time_for_model,
            )
        if racer_results is None or racer_results.empty:
            logger.warning(f"Creating blank entry for {racer_name}; no usable results found.")
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

        prediction_t = None
        if racer_results.empty or racer_results.loc[racer_results['include']].empty:
            prediction_str = 'N/A'
        else:
            # Subtracting 1 from the year of entry as year before race is most recent possible
            chase_mu, chase_sig = make_chase_prediction(
                racer_results.loc[racer_results['include']],
                prediction_year=year_of_entry,
                verbose=False,
            )
            prediction = chase_mu - (1.96 * chase_sig)
            prediction_t = convert_Chase_ZScore_logs_avg(con, prediction)[0]
            prediction_str = seconds_to_time_string(prediction_t)
            all_racer_predictions = pd.concat(
                [
                    all_racer_predictions,
                    pd.DataFrame(
                        [{
                            'Racer_Name': racer_name.lower(),
                            'chase_mu': chase_mu,
                            'chase_sig': chase_sig
                        }]
                    )
                ],
                ignore_index=True
            )

            if plot:
                plot_racer_entry(con, racer_results, None, chase_mu, chase_sig, prediction_t, racer_name, prediction_year=this_year)

        # Create a series for this racer's entry
        # Add last three years of chase results too
        entry_series = pd.Series({
            'Name': racer_name,
            'Num_results_used': len(racer_results.loc[racer_results['include']]),
            'Num_excluded_results': len(racer_results.loc[~racer_results['include']]),
            'Predicted_Time': prediction_str,
            'Given PR prediction': pr_prediction_str,
            'Predicted_Time_seconds': prediction_t,
            f'Chase {this_year-1}': extract_result_for_year(chase_results, this_year - 1),
            f'Chase {this_year-2}': extract_result_for_year(chase_results, this_year - 2),
            f'Chase {this_year-3}': extract_result_for_year(chase_results, this_year - 3),
        })
        processed_entries = pd.concat([processed_entries, entry_series.to_frame().T], ignore_index=True)
        
        

    logger.info(f"Processed entries:\n {tabulate(processed_entries, headers='keys', tablefmt='rounded_outline')}")
    results_filepath = ENTRIES_PATH / f"racer_results_{year_of_entry}.json"
    predictions_filepath = ENTRIES_PATH / f"racer_predictions_{year_of_entry}.json"
    
    
    all_racer_results.to_json(results_filepath, index=False, indent=4)
    all_racer_predictions.to_json(predictions_filepath, index=False, indent=4)
    return processed_entries


def export_entries_to_csv(processed_entries: pd.DataFrame, year_of_entry: int):
    """Export procssed entries to a csv file, denominated by year of entry.

    Args:
        processed_entries (pd.DataFrame): The processed entries dataframe
        year_of_entry (int): The year of entry
    """
    entries_filepath = ENTRIES_PATH / f"processed_entries_{year_of_entry}.csv"
    processed_entries.to_csv(entries_filepath, index=False)
    
 
def limit_results_to_requested_years(
    racer_results: pd.DataFrame,
    chase_results: pd.DataFrame,
    year_of_entry: int,
    p10_results: pd.DataFrame = None)-> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Limit racer results and chase results to only those before the requested year.
    
    Args:
        racer_results (pd.DataFrame): DataFrame containing the racer's results.
        chase_results (pd.DataFrame): DataFrame containing the chase results.
        year_of_entry (int): Year of the entry.
        
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
            - pd.DataFrame: The limited racer's results.
            - pd.DataFrame: The limited chase results.
    """
    limited_racer_results = racer_results[racer_results['Season'] < year_of_entry].reset_index(drop=True)
    limited_chase_results = chase_results[chase_results['Season'] < year_of_entry].reset_index(drop=True)
    if p10_results is not None:
        limited_p10_results = p10_results[p10_results['Season'] < year_of_entry].reset_index(drop=True)
        return limited_racer_results, limited_chase_results, limited_p10_results
    return limited_racer_results, limited_chase_results, None
  
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


