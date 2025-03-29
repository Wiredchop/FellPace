import sqlite3
import pandas as pd
from pathlib import Path
import typer
from fellpace.FellPace_tools import append_to_DB, process_data_for_DB, get_table_from_URL
from fellpace.scraping_tools import get_avtiming_api, get_racetek_api
#Connect to the DB
con = sqlite3.connect('fellpace.db')

app = typer.Typer()
# Choose method of getting data
# 1: AV Timing
# 2: csv files
# 3: html table

@app.command()
def process_url(URL: str):
    if ('avtiming' in URL) or ('raceresult' in URL):
        data = get_avtiming_api(URL)
    elif 'racetek' in URL:
        data = get_racetek_api(URL)
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
def process_html(URL: str):
    data,_ = get_table_from_URL(URL)
    add_data(data)
    
def add_data(data):
    (metadata,entries) = process_data_for_DB(data)
    #Clean any null entries for time, which can't be converted to a Zscore
    valid_data = entries.data.loc[~entries.data.Time.isnull()]
    append_to_DB(con,valid_data,metadata)

if __name__ == "__main__":
    # Run the app
    app()