import fellpace.FellPace_tools as FellPace_tools
import sqlite3
import toml
from typing import Literal
import fellpace.convert_tools as convert_tools
from fellpace.parkrun.settings import PRSettings
from re import search
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
parkrun: Literal['hillsborough','endcliffe'] = 'endcliffe'
parkrun_climbs = {'hillsborough':53,'endcliffe':47}

def scrape_parkruns(settings: PRSettings, con: sqlite3.Connection):
    
    for parkrun, these_settings in settings.__dict__.items():
        continue_scrape = True
        PR_id = these_settings.start_ID
        while continue_scrape:
            URL = f'https://www.parkrun.org.uk/{parkrun}/results/{PR_id}/'
            try:
                table, resp_text = FellPace_tools.get_table_from_URL(URL, headers=headers)
            except Exception as e:
                print(f"Exception occurred: {e}")
                continue_scrape = False
                these_settings.start_ID = PR_id
                print(f'No more parkruns found for {parkrun} at ID {PR_id}')
                with open('settings.toml', 'w') as f:
                    f.write(toml.dumps(settings.model_dump()))
                continue
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
            this_parkrun.race_name = f"Parkrun_{parkrun}_{PR_id}" #Parkrun name has the ID appended to the back so can easily parse in future if want to update
            print(f'adding {this_parkrun.race_name}')
            ParkRun = convert_tools.ParkRunConverter(table)
            FellPace_tools.append_to_DB(con,ParkRun.entries.data,this_parkrun,check=False)
            PR_id += 1
            time.sleep(2)

if __name__ == "__main__":
    from fellpace.config import DB_PATH
    from fellpace.db.db_setup import setup_db
    #Connect to the DBy
    con = setup_db(DB_PATH)
    settings = PRSettings.load_toml_settings('settings.toml')
    scrape_parkruns(settings,con)
    con.close()