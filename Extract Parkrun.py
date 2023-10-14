import FellPace_tools
import sqlite3
from typing import Literal
import convert_tools
import time
# Headers work for Parkrun
headers = {
"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0",
"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
"Accept-Language": "en-GB,en;q=0.5",
"Accept-Encoding": "gzip, deflate, br",
"Referer": "https://www.parkrun.org.uk/hillsborough/results/eventhistory/",
"Connection": "keep-alive",
"Cookie": "cookiesDisclosureCount=14",
"Upgrade-Insecure-Requests": "1",
"Sec-Fetch-Dest": "document",
"Sec-Fetch-Mode": "navigate",
"Sec-Fetch-Site": "same-origin",
"Sec-Fetch-User": "?1"}
parkrun: Literal['hillsborough','endcliffe'] = 'hillsborough'
parkrun_climbs = {'hillsborough':53,'endcliffe':47}




start_ID = 365 #For updates, can extract start ID from the database to ensure don't overwrite
end_ID = 465
from re import search
for i in range(start_ID,end_ID):
    #Connect to DB
    con = sqlite3.connect('fellpace.db')
    cur = con.cursor()
    #Will do in every loop as append function closes connection, probably horribly inefficient
    
    
    
    URL = f'https://www.parkrun.org.uk/{parkrun}/results/{i}/' 
    table,resp_text = FellPace_tools.get_table_from_URL(URL,headers=headers)
    #Going to use regular expressions to get date rather than beautiful soup as only need to do once
    matches = search("(?<=class=\"format-date\">)[0-9/]+",resp_text)
    if not matches:
        date = ""
    else:
        date = matches.group()
        (day,month,year) = date.split("/")
        date = "-".join((year,month,day))
    #Create the race metadata for the entry
    this_parkrun = FellPace_tools.race_meta()
    this_parkrun.race_distance = 5000
    this_parkrun.race_climb = parkrun_climbs[parkrun]
    this_parkrun.race_date = date
    this_parkrun.race_name = f"Parkrun_{parkrun}_{i}" #Parkrun name has the ID appended to the back so can easily parse in future if want to update
    print(f'adding {this_parkrun.race_name}')
    ParkRun = convert_tools.ParkRunConverter(table)
    FellPace_tools.append_to_DB(con,ParkRun.entries.data,this_parkrun,check=False)
    time.sleep(2)

#test = ConvertCat(None,table['Age Group'])