import sqlite3
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
from fellpace.modelling.training import get_race_coeffs
from fellpace.plotting.races import plot_all_race_Zscores
from fellpace.extract.races import get_race_series_summary, get_chase_summary

from fellpace.extract.racers import find_similar_name, find_racer_ID, get_racers_results
from fellpace.config import DB_PATH
from fellpace.db.db_setup import setup_db
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
    coeffs = get_race_coeffs(data_Zs)
    if plot:
        plot_all_race_Zscores(data_Zs)
    
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

@app.command()
def print_racers_results(racer_name:str = 'nick hamillton'):
    
    con = setup_db(DB_PATH)
    print(f"Getting results for {racer_name}")
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
    results = get_racers_results(con, racer_id, -1)
    print(
        tabulate(
            results.sort_values(['Season','Race_Name']).reset_index(drop=True),
            headers='keys',
            tablefmt='rounded_outline'
            )
        )

if __name__ == "__main__":
    app()
    #process_csv('ShefHalf-2022.csv')

con.close()