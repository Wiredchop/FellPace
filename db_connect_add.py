import sqlite3
import pandas as pd
from FellPace_tools import append_to_DB, get_avtiming_api, process_data_for_DB, get_table_from_URL
#Connect to the DB
con = sqlite3.connect('fellpace.db')

# Choose method of getting data
# 1: AV Timing
# 2: csv files
# 3: html table

option = 2

if option == 1:
    # Get from AV timing
    URL = input('What is the URL of the API?\n')
    data = get_avtiming_api(URL)
elif option == 2:
    # Get from csv file
    file = input('What is the name of the csv?')
    dirfile = './csv/' + file
    data = pd.read_csv(dirfile)
elif option == 3:
    # Get from html table
    URL = input('What is the URL of the table?')
    data,_ = get_table_from_URL(URL)
else:
    data = pd.DataFrame()


(metadata,entries) = process_data_for_DB(data)
append_to_DB(con,entries.data,metadata)