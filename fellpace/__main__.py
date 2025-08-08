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

from fellpace.config import DB_PATH
from fellpace.db.db_setup import setup_db
from fellpace.extract.zscores import extract_all_zscore_data
from fellpace.modelling.ransac import add_inliers
from fellpace.modelling.training import train_models, get_rmse_in_seconds, load_models
from fellpace.modelling.prediction import get_predicted_times,get_prediction_with_uncertainty_many, make_chase_prediction
from fellpace.plotting.races import plot_all_race_Zscores
from fellpace.plotting.racetimes import plot_time_normal
from fellpace.extract.races import get_race_series_summary, get_chase_summary
from fellpace.analysis_tools import convert_Chase_ZScore_logs_avg, identify_outliers_in_predictions

from fellpace.extract.racers import get_racers_results
from fellpace.config import DB_PATH, COEFFS_FILE_PATH, COVAR_FILE_PATH
from fellpace.db.db_setup import setup_db
from fellpace.convert_tools import seconds_to_time_string
from fellpace.scrape_chase import process_chase_csv

from fellpace.entries import load_entries, process_entries
from fellpace.filter import filter_race_results
from fellpace.plotting.racetimes import plot_racers_results
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
    data_Zs = extract_all_zscore_data(con)
    data_Zs = add_inliers(data_Zs)
    coeffs, covar = train_models(data_Zs)
    rmse = get_rmse_in_seconds(data_Zs, coeffs)
    logger.info(tabulate(pd.DataFrame(rmse), headers=['Race Name','RMSE'], tablefmt='rounded_outline'))
    if plot:
        plot_all_race_Zscores(data_Zs)

    coeffs.to_json(COEFFS_FILE_PATH)
    covar.to_json(COVAR_FILE_PATH)
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
    coeffs, covar = load_models()
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
def plot_racer_likelihoods(racer_name: str = 'nick hamilton'):
    con = setup_db(DB_PATH)
    coeffs, covar = load_models()
    logger.info(f"Predicting finish time for {racer_name}")
    racer_id = secure_racer_id(con, racer_name)
    racer_results = get_racers_results(con, racer_id)
        
    racer_results = get_prediction_with_uncertainty_many(coeffs, covar, racer_results)
    
    racer_results['outlier'] = identify_outliers_in_predictions(racer_results['Zpred_mu'], threshold=1.2)
        
    racer_results, excluded_results = filter_race_results(racer_results)
    
    chase_mu, chase_sig = make_chase_prediction(racer_results, prediction_year=2024, verbose=True)
    
    _, ax = plt.subplots(figsize=(10, 6))    
    
    plot_racers_results(racer_results, con, ax=ax, linestyle='-')
    plot_racers_results(excluded_results, con, ax=ax, linestyle=':')
    plot_time_normal(con, chase_mu, chase_sig, 'Chase 2024',ax, color='black', linewidth=2)
    
    prediction = chase_mu - (1.96 * chase_sig)
    prediction_t = convert_Chase_ZScore_logs_avg(con, prediction)
    plt.vlines(prediction_t, 0, 0.2, color='black', linestyle='--', label='Predicted time')
        
    plt.xlabel("Predicted Time")
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: seconds_to_time_string(x)))  # Format xticks
    plt.legend()  # Add legend to display the labels
    plt.show()

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
    
@app.command()
def entries():
    con = setup_db(DB_PATH)
    entries = load_entries()
    process_entries(entries, con)

if __name__ == "__main__":
    app()
    #process_csv('ShefHalf-2022.csv')

con.close()