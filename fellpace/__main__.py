import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from datetime import date
import typer
from typer import Option
from fellpace.FellPace_tools import append_to_DB, process_data_for_DB, get_table_from_URL

from fellpace.extract.racers import secure_racer_id
from fellpace.scraping_tools import get_avtiming_api, get_racetek_api
from tabulate import tabulate

from fellpace.config import DB_PATH, TIME_COEFFS_FILE_PATH, ROAD_TIME_COEFFS_FILE_PATH, RESID_STD_FILE_PATH, COEFFS_FILE_PATH, COVAR_FILE_PATH
from fellpace.db.db_setup import setup_db
from fellpace.extract.zscores import (
    extract_all_zscore_data,
    extract_previous_year_chase_data,
    extract_older_chase_data,
    extract_previous_year_chase_times,
    extract_older_chase_times,
)
from fellpace.modelling.ransac import add_inliers
from fellpace.modelling.training import train_models, train_time_models, get_rmse_in_seconds, load_models
from fellpace.p10.chase_combiner import extract_road_chase_training_data
from fellpace.modelling.prediction import get_predicted_times, make_chase_prediction
from fellpace.plotting.races import plot_all_race_Zscores
from fellpace.plotting.racetimes import plot_time_normal, plot_racers_results, generate_racer_prediction_plot
from fellpace.extract.races import get_race_series_summary, get_chase_summary
from fellpace.analysis_tools import convert_Chase_ZScore_logs_avg, identify_outliers_in_predictions

from fellpace.extract.racers import get_racers_results
from fellpace.convert_tools import seconds_to_time_string
from fellpace.scrape_chase import process_chase_csv

from fellpace.entries import load_entries, process_entries, load_PR_entries, process_PR_entries, export_entries_to_csv, process_results_for_racer
from fellpace.handicaps import calculate_handicaps_for_entries
from fellpace.filter import filter_race_results
from loguru import logger

#Connect to the DB
con = setup_db(DB_PATH)

# Configure loguru to log to a file
logger.add("fellpace.log", rotation="10 MB")

app = typer.Typer()
# Choose method of getting data
# 1: AV Timing
# 2: csv files
# 3: html table

@app.command()
def process_url(url: str):
    if ('avtiming' in url) or ('raceresult' in url):
        data = get_avtiming_api(url)
    elif 'racetek' in url:
        data = get_racetek_api(url)
    add_data(data)
        
def ensure_extension(filename, extension=".csv"):
    path = Path(filename)
    if path.suffix != extension:
        path = path.with_suffix(extension)
    return path        
        
@app.command()        
def process_csv(filename: str):
    # Get path with extension
    filepath =  Path('./csv') / ensure_extension(filename)
    data = pd.read_csv(filepath)
    add_data(data)
    
@app.command()
def process_html(url: str,):
    data,_ = get_table_from_URL(url)
    add_data(data)
    
def add_data(data):
    # Strip whitespace from all string columns at ingestion point
    data = data.map(lambda x: x.strip() if isinstance(x, str) else x)
    (metadata,entries) = process_data_for_DB(data)
    #Clean any null entries for time, which can't be converted to a Zscore
    valid_data = entries.data.loc[~entries.data.Time.isnull()]
    append_to_DB(con,valid_data,metadata)

@app.command()
def train_model(
    plot: bool = Option(
        False, "--plot", "-p", help="Whether to plot the results or not"
    )
):
    """Get coefficients from all race data."""
    con = setup_db(DB_PATH)
    data_Zs_race = extract_all_zscore_data(con)
    data_Zs_HC1 = extract_previous_year_chase_data(con)
    data_Zs_HC2 = extract_older_chase_data(con)
    data_Zs = pd.concat([data_Zs_race, data_Zs_HC1, data_Zs_HC2], ignore_index=True)
    data_Zs = add_inliers(data_Zs)
    coeffs, covar, resid_stds = train_models(data_Zs)
    rmse = get_rmse_in_seconds(data_Zs, coeffs)
    logger.info(tabulate(pd.DataFrame(rmse), headers=['Race Name','RMSE'], tablefmt='rounded_outline'))
    if plot:
        plot_all_race_Zscores(data_Zs)

    coeffs.to_json(COEFFS_FILE_PATH)
    covar.to_json(COVAR_FILE_PATH)
    resid_stds.to_json(RESID_STD_FILE_PATH)

    # Train time-domain models for previous_chase and older_chase
    time_data = pd.concat(
        [extract_previous_year_chase_times(con), extract_older_chase_times(con)],
        ignore_index=True,
    )
    time_coeffs = train_time_models(time_data)
    time_coeffs.to_json(TIME_COEFFS_FILE_PATH)
    logger.info(f"Time models trained for: {list(time_coeffs.index)}")

    # Train road time models (5k and 10k PO10 best times -> Chase time)
    road_data = extract_road_chase_training_data()
    if not road_data.empty:
        road_time_coeffs = train_time_models(road_data)
        road_time_coeffs.to_json(ROAD_TIME_COEFFS_FILE_PATH)
        logger.info(f"Road time models trained for: {list(road_time_coeffs.index)}")
    else:
        logger.warning("No road time training data available — skipping road time models.")

    return coeffs

@app.command()
def print_race_data():
    con = setup_db(DB_PATH)
    race_summary = get_race_series_summary(con)
    con.close() #TODO: Have a class that closes this automatically
    logger.info(tabulate(race_summary, headers='keys', tablefmt='rounded_outline'))

@app.command()
def print_chase_data():
    con = setup_db(DB_PATH)
    chase_summary = get_chase_summary(con)
    con.close() #TODO: Have a class that closes this automatically
    logger.info(tabulate(chase_summary, headers='keys', tablefmt='rounded_outline'))

@app.command()
def print_racers_results(racer_name:str = 'nick hamillton'):
    con = setup_db(DB_PATH)
    logger.info(f"Getting results for {racer_name}")
    racer_id = secure_racer_id(con, racer_name)
    if racer_id:
        results = get_racers_results(con, racer_id, -1)
        logger.info(
            tabulate(
                results.sort_values(['Season','Race_Name']).reset_index(drop=True),
                headers='keys',
                tablefmt='rounded_outline'
                )
            )

@app.command()
def print_racer_prediction(racer_name: str = 'nick hamilton'):
    con = setup_db(DB_PATH)
    coeffs, _ = load_models()
    logger.info(f"Predicting finish time for {racer_name}")
    racer_id = secure_racer_id(con, racer_name)
    prediction = get_predicted_times(con, coeffs,racer_id)
    prediction['Predicted Time'] = prediction['Predicted Time'].apply(seconds_to_time_string)
    logger.info(
        tabulate(
            prediction.sort_values(['Season','Race_Name']).reset_index(drop=True),
            headers='keys',
            tablefmt='rounded_outline'
            )
        )
    con.close()
    
@app.command()
def examine_entries(year: int = date.today().year):
    """
    Examine the entries for a given year.
    
    Prints names and how many times they appear in the database.
    
    Args:
        year (int): The year to examine entries for.
    """
    entries = load_entries(year)
    if entries.empty:
        logger.info(f"No entries found for {year}.")
        return
    all_results = pd.DataFrame()
    for i, row in entries.iterrows():
        racer_name = row['Name'].lower().strip()
        racer_id = secure_racer_id(con, racer_name)
        if racer_id is None:
            logger.info(f"Racer {racer_name} not found in database.")
            continue
        all_results = pd.concat([all_results, get_racers_results(con, racer_id)], ignore_index=True)
        racer_counts = all_results.groupby('Racer_Name').size()
        
    # merge back to entries
    entries = entries.merge(racer_counts.rename('Count'), left_on='Name', right_index=True, how='left')
        
    logger.info(
        tabulate(
            entries.sort_values('Count', ascending=False).reset_index(drop=True),
            headers='keys',
            tablefmt='rounded_outline'
        ))

@app.command()
def show_race_outliers(racer_name: str = 'nick hamilton'):
    con = setup_db(DB_PATH)
    coeffs, covar, resid_stds = load_models(include_residuals=True)
    logger.info(f"Examining potential outliers for {racer_name}")
    racer_id = secure_racer_id(con, racer_name)
    
    racer_results = get_predicted_times(con, coeffs,racer_id).sort_values('PredZ', ascending=True)
    
    identify_outliers_in_predictions(racer_results['PredZ'], threshold=1.2)
    
    racer_results['Expanding Mean'] = racer_results['PredZ'].expanding().mean()
    racer_results['Expanding Std'] = racer_results['PredZ'].expanding().std()
    mean_mean = racer_results['Expanding Mean'].mean()
    racer_results['Distance from Mean'] = (racer_results['Expanding Mean'] - mean_mean).abs()
    # drop Racer_Name column
    racer_results = racer_results.drop(columns=['Racer_Name'])
    # use tabulate to print racer_results
    logger.info(tabulate(racer_results, headers='keys', tablefmt='rounded_outline'))
    logger.info(racer_results['Expanding Mean'].mean())
    con.close()

@app.command()
def plot_racer_likelihoods(racer_name: str = 'nick hamilton', year: int = date.today().year):
    con = setup_db(DB_PATH)
    coeffs, covar, resid_stds = load_models(include_residuals=True)
    logger.info(f"Predicting finish time for {racer_name} in {year}")
    racer_id, racer_name = secure_racer_id(con, racer_name)
    if racer_id is None:
        logger.warning(f"Racer {racer_name} not found in database.")
        return
    
    generate_racer_prediction_plot(
        con,
        (coeffs, covar, resid_stds),
        racer_id,
        racer_name,
        year=year,
        display=True,
    )
    con.close()

@app.command()
def process_chase(file: str, date: str):
    """
    Process a Hallam Chase CSV file and insert its data into the database.

    Args:
        file (str): The name of the CSV file (e.g., 'Chase 2016.csv').
        date (str): The date of the Chase in 'yyyy-mm-dd' format.
    """
    con = setup_db(DB_PATH)
    process_chase_csv(file, date, con)
    con.close()
    
def generate_racer_plot(
    con,
    model_tuple: tuple,
    racer_id: int,
    racer_name: str,
    year: int,
    output_dir: Path,
):
    """
    Generate and save a prediction likelihood plot for a single racer.
    Wrapper around the plotting module's generate_racer_prediction_plot.
    """
    generate_racer_prediction_plot(
        con,
        model_tuple,
        racer_id,
        racer_name,
        year=year,
        output_dir=output_dir,
        display=False,
    )

@app.command()
def entries(
    year: int = date.today().year,
    use_parkrun: bool = Option(True, "--use-parkrun", "-p", help="Whether to use parkrun data"),
    plot_likelihoods: bool = Option(False, "--plot-likelihoods", "-pl", help="Generate and export prediction plots for each racer")
):
    con = setup_db(DB_PATH)
    if use_parkrun:
        entries = load_PR_entries(year_of_entry=year)
        entries = process_PR_entries(entries, con, year)
    else:
        entries = load_entries(year_of_entry=year)
    processed_entries = process_entries(entries, con, year, with_parkrun=use_parkrun)
    processed_entries = calculate_handicaps_for_entries(processed_entries)
    export_entries_to_csv(processed_entries, year_of_entry=year)
    
    if plot_likelihoods:
        coeffs, covar, resid_stds = load_models(include_residuals=True)
        model_tuple = (coeffs, covar, resid_stds)
        # Create output directory for plots
        output_dir = Path(f"./racer_likelihoods_{year}")
        output_dir.mkdir(exist_ok=True)
        logger.info(f"Generating prediction plots for all racers in {output_dir}")
        
        # Get unique racer names from processed entries
        unique_racers = processed_entries['Name'].unique()
        
        for racer_name in unique_racers:
            racer_id, resolved_name = secure_racer_id(con, racer_name)
            if racer_id is None:
                logger.warning(f"Racer {racer_name} not found in database, skipping.")
                continue
            generate_racer_plot(con, model_tuple, racer_id, resolved_name, year, output_dir)
        
        logger.info(f"Finished generating plots. Saved to {output_dir}")

if __name__ == "__main__":
    app()
    #process_csv('ShefHalf-2022.csv')

con.close()