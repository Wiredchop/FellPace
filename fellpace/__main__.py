import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import typer
from typer import Option
from fellpace.FellPace_tools import append_to_DB, process_data_for_DB, get_table_from_URL
from fellpace.scraping_tools import get_avtiming_api, get_racetek_api
from tabulate import tabulate

from fellpace.config import DB_PATH
from fellpace.db.db_setup import setup_db
from fellpace.extract.zscores import extract_all_zscore_data
from fellpace.modelling.ransac import add_inliers
from fellpace.modelling.training import get_race_coeffs, get_rmse_in_seconds
from fellpace.modelling.prediction import get_predicted_times, get_prediction_probability_distribution
from fellpace.plotting.races import plot_all_race_Zscores
from fellpace.extract.races import get_race_series_summary, get_chase_summary

from fellpace.extract.racers import find_similar_name, find_racer_ID, get_racers_results
from fellpace.config import DB_PATH
from fellpace.db.db_setup import setup_db
from fellpace.convert_tools import seconds_to_time_string
from fellpace.scrape_chase import process_chase_csv

#Connect to the DB
con = setup_db(DB_PATH)

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
def train_model(plot: bool = Option(False, help="Whether to plot the results or not")):
    """Get coefficients from all race data."""
    con = setup_db(DB_PATH)
    data_Zs = extract_all_zscore_data(con)
    data_Zs = add_inliers(data_Zs)
    coeffs, _ = get_race_coeffs(data_Zs)
    rmse = get_rmse_in_seconds(data_Zs, coeffs)
    print(tabulate(rmse, headers='keys', tablefmt='rounded_outline'))
    if plot:
        plot_all_race_Zscores(data_Zs)
    return coeffs
    
@app.command()
def print_race_data():
    con = setup_db(DB_PATH)
    race_summary = get_race_series_summary(con)
    con.close() #TODO: Have a class that closes this automatically
    print(tabulate(race_summary, headers='keys', tablefmt='rounded_outline'))

@app.command()
def print_chase_data():
    con = setup_db(DB_PATH)
    chase_summary = get_chase_summary(con)
    con.close() #TODO: Have a class that closes this automatically
    print(tabulate(chase_summary, headers='keys', tablefmt='rounded_outline'))

def secure_racer_id(con, racer_name: str):
    racer_id = find_racer_ID(con, name = racer_name)
    if racer_id is None:
        print(f"Racer {racer_name} not found in database.")
        print("Looking for similar names...")
        names = find_similar_name(con, name = racer_name)
        # Print each name with a number and prompt to select one
        if names.empty:
            print("No similar names found.")
            return
        for i, row in names.iterrows():
            print(f"{i}: {row['Racer_Name']}")
        print(f"{i+1}: None of these are correct")
        selected_index = int(input("Select the number of the name you want to use: "))
        if selected_index == i+1:
            print("No name selected, exiting.")
            return
        racer_id = names.iloc[selected_index]['Racer_ID']
    return racer_id

@app.command()
def print_racers_results(racer_name:str = 'nick hamillton'):
    
    con = setup_db(DB_PATH)
    print(f"Getting results for {racer_name}")
    racer_id = secure_racer_id(con, racer_name)
    if racer_id:
        results = get_racers_results(con, racer_id, -1)
        print(
            tabulate(
                results.sort_values(['Season','Race_Name']).reset_index(drop=True),
                headers='keys',
                tablefmt='rounded_outline'
                )
            )
    
@app.command()
def print_racer_prediction(racer_name: str = 'nick hamilton'):
    con = setup_db(DB_PATH)
    data_Zs = extract_all_zscore_data(con)
    data_Zs = add_inliers(data_Zs)
    coeffs, _ = get_race_coeffs(data_Zs)
    print(f"Predicting finish time for {racer_name}")
    racer_id = secure_racer_id(con, racer_name)
    prediction = get_predicted_times(con, coeffs,racer_id)
    prediction['Predicted Time'] = prediction['Predicted Time'].apply(seconds_to_time_string)
    print(
        tabulate(
            prediction.sort_values(['Season','Race_Name']).reset_index(drop=True),
            headers='keys',
            tablefmt='rounded_outline'
            )
        )
    con.close()
    
@app.command()
def plot_racer_likelihoods(racer_name: str = 'nick hamilton'):
    con = setup_db(DB_PATH)
    data_Zs = extract_all_zscore_data(con)
    data_Zs = add_inliers(data_Zs)
    coeffs, covar = get_race_coeffs(data_Zs)
    print(f"Predicting finish time for {racer_name}")
    racer_id = secure_racer_id(con, racer_name)
    racer_results = get_racers_results(con, racer_id)
    
    plt.figure(figsize=(10, 6))
    
    for _, result in racer_results.iterrows():
        season = result['Season']
        race = result['Race_Name']
        ZScore = result['ZScore']    
        p = get_prediction_probability_distribution(coeffs[race], covar[race], ZScore)

        plt.plot(p.keys(), p.values(), label=f"{race} ({season})")
        
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

if __name__ == "__main__":
    app()
    #process_csv('ShefHalf-2022.csv')

con.close()