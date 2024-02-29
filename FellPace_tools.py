import pandas as pd
import numpy as np
import numpy.typing as npt
from typing import Literal, Tuple, List
import convert_tools
import analysis_tools
import requests
import json
import datetime
from datetime import datetime as dt 
import sqlite3
import Levenshtein

class race_entries:   
    
    def __init__(self,num_entries):
        self.data = pd.DataFrame(
            {
                'Racer_Name': pd.Series(dtype='str'),
                'Club': pd.Series(dtype='str'),
                'Time': pd.Series(dtype='int'),
                'Position': pd.Series(dtype='int'),
                'Cat_Name': pd.Series(dtype='str')                
            },range(num_entries)
        )       
    
    @property
    def data(self):
        #Only return the entries where time is not None
        return self._data
    
    @data.setter
    def data(self,value: pd.DataFrame):
        self._data = value
        
    def add_column_of_data(self,column: Literal['Racer_Name','Club','Time','Position','Cat_Name']
                           ,column_data: npt.ArrayLike):
        if column not in self.data.columns:
            raise Exception('Incorrect column name')
        if column == 'Cat_Name':
            column_data = convert_tools.convert_categories(column_data)
        if column == 'Time':
            column_data = convert_tools.time_string_to_seconds(column_data)
        if column == 'Position':
            #Need to ensure there isn't any text in the position data, remove and just return number
            column_data = convert_tools.clean_position_date(column_data)
        if column == 'Racer_Name':
            #Ensure is title case
            column_data = [name.title() for name in column_data]
        self.data.loc[:,column] = column_data 
    
        
def get_table_from_URL(url: str,headers: dict = {}) -> Tuple[pd.DataFrame,str]:   #Export the text response if useful at other end
    
    print('Getting data from URL')
    if headers:
        response = requests.get(url,headers= headers)
    else:
        response = requests.get(url)
    # The vastly superior method for obtaining html tables from a webpage
    df_list = pd.read_html(response.text) # this parses all the tables in webpages to a list
    print(f'The reading has returned {len(df_list)} tables')
    print(f'The size of each is {[(len(df.index), len(df.columns)) for df in df_list]}')
    if len(df_list) == 1:
        print(f'Only 1 table, so using that one')
        i = 0
    else:
        i_str = input('Which table would you like to scrape, based on expected size?\n')
        i = int(i_str)
        

    return df_list[i],response.text

class race_meta:
    
    def __init__(self) -> None:
        self.race_date = '0001-1-1'
        self.race_climb = -1
        self.race_distance = -1
        self.race_name = 'invalid'
        self.series_id = -1
    @property
    def get_DB_entry(self) -> Tuple[str,str,int,int,int]:
        return (self.race_name,self.race_date,self.race_distance,self.race_climb,self.series_id)
        
    @property
    def race_name(self) -> str:
        return self._race_name
    @race_name.setter
    def race_name(self,value: str):
        self._race_name = value
    
    @property
    def race_date(self) -> str:
        return self._race_date
    @property
    def race_date_as_datetime(self) -> datetime.datetime:
        return datetime.datetime.strptime(self.race_date,'%Y-%m-%d')
    @race_date.setter
    def race_date(self,value: str):
        #Check value for validity
        try:
            datetime.datetime.strptime(value,'%Y-%m-%d')     
        except:
            raise Exception('String cannot be parsed as date in format DD-MM-YYYY')
        self._race_date = value
        
    @property
    def race_distance(self) -> int:
        return self._race_distance
    @race_distance.setter
    def race_distance(self,value: str|int|float):
        self._race_distance = int(value)

    @property
    def race_climb(self) -> int:
        return self._race_climb
    @race_climb.setter
    def race_climb(self,value: str|int|float):
        self._race_climb = int(value)
        
    @property
    def series_id(self) -> int:
        return self._series_id
    @series_id.setter
    def series_id(self,value: str|int|float):
        self._series_id = int(value)

def process_data_for_DB(scraped_data : pd.DataFrame) -> Tuple[race_meta,race_entries]:
    race_metadata = race_meta()
    race_meta.race_name = input('What is the name of the race, for the database?\n')
    race_meta.race_date = input('What was the date of the race, in the format yyyy-mm-dd\n')
    race_meta.race_distance = input('What was the distance of the race, in metres?\n')
    race_meta.race_climb = input('What was the total climb of the race, in metres?\n')
    
    #Create the data structure to store the main dataset
    num_entries = len(scraped_data.index)
    entries = race_entries(num_entries)

    print('...')
    print(f'The column names in the chosen table are')

    for index, col in enumerate(list(scraped_data)):
        print(f'{index}: {col}')

    print('\nSome example data')
    
    pd.set_option('display.max_columns', None) # Prevent truncating columns, we need them!
    print(scraped_data.head())

    def get_column_data(scraped_data : pd.DataFrame,col_name : str) -> Tuple[npt.ArrayLike,List[int]]:
        def print_choices():
            print(scraped_data.head())
            print('\nYou can choose from')
            index = 0
            for index, col_scraped in enumerate(list(scraped_data.columns)):
                print(f'{index}: {col_scraped}')
            print(f'{index+1}: None\n')
            return index + 1
        
        print(f'Which of the scraped columns should be used for: << {col_name} >>\n')
        if col_name == 'Racer_Name':
            print('You are choosing a name, are firstname/surname in different columns?')
            two_cols_for_name = input('y/(n)')
            if two_cols_for_name == 'y':
                print_choices()
                forename_i = int(input('\nWhich column for forename?'))
                surname_i = int(input('\nWhich for surname?'))
                print('\n Will join with a space to match convention')
                forename = scraped_data.iloc[:,forename_i]
                surname = scraped_data.iloc[:,surname_i]
                name = forename + ' ' + surname
                return (name.values, [forename_i,surname_i])
        if col_name == 'Cat_Name':
            print('Is gender given in a separate column?')
            sep_gender = input('y/(n)')
            if sep_gender == 'y':
                print_choices()
                gen_col = int(input('\nWhich column for gender?'))
                cat_col = int(input('\nWhich column for category name?'))
                gen = scraped_data.iloc[:,gen_col]
                cat = scraped_data.iloc[:,cat_col]
                category = gen+cat
                return (category.values,[gen_col,cat_col])
        num_choices = print_choices()
        print(f'Which of the scraped columns should be used for: << {col_name} >>\n')
        choice = int(input('Enter column index:'))
        if choice == num_choices: #Selected None
            return (np.empty(0), [])
        
        return (scraped_data.iloc[:,choice].values,[choice])

    print('\n\nWe are looping through the data we need')
    for index, col_entries in enumerate(list(entries.data)):
        (data,choices) = get_column_data(scraped_data,col_entries)
        if data.size == 0:
            continue # don't need to add data, already an empty column
        else:
            entries.add_column_of_data(col_entries,data)
            chosen_columns = scraped_data.columns[choices]
            scraped_data = scraped_data.drop(chosen_columns,axis = 1)
    
    return (race_metadata,entries)        

# This is a CHASE version of the append to DB due to the specific nature of the Chase data tables 
def append_CHASE(con: sqlite3.Connection, data_to_insert: pd.DataFrame):
    Racers = pd.read_sql_query('SELECT * FROM Racers',con)
    Categories = pd.read_sql_query('SELECT * FROM Categories',con)
    # The data will have the Race_ID added already
    
    
    #RACERS
    #Get list of Racers not already in the database
    Racers_new = check_db_for_duplicate_racers(data_to_insert, Racers)
    
    # Checks data is ok
    print(f'About to insert {len(Racers_new.index)} racers')
    print(f'{len(data_to_insert.index) - len(Racers_new.index)} were already in the database')
    print('Examples below')
    print(Racers_new.head())
    key  = input('Continue, or quit? Any key to quit')
    if key:
        exit()
        
    #Commit to the database
    Racers_new.to_sql('Racers',con,if_exists='append',index=False)
    #Get the new data, including the Racer_IDs
    Racers = pd.read_sql_query('SELECT * FROM Racers',con)
    
    #FINALISE DATA
    # Racers
    data_to_insert = pd.merge(data_to_insert,Racers,'inner',['Racer_Name','Racer_Name'])
    # Categories
    data_to_insert = pd.merge(data_to_insert,Categories,'left',['Cat_Name','Cat_Name'])

    data_to_insert = data_to_insert[['Chase_ID','Racer_ID','Time','Cat_ID','Position', 'Handicap']]
    # Trim to the columns needed
    
    # Add the statistics columns
    (data_to_insert['ZScore'],data_to_insert['ZScore_log'],data_to_insert['Percentile']) = analysis_tools.calculate_position_stats(data_to_insert['Time'].values)

    print(f'About to insert {len(data_to_insert.index)} records')
    print(data_to_insert.head())
    key = input('Proceed? Any key to quit')
    if key:
        exit()
    data_to_insert.to_sql('Results_Chase',con,index=False,if_exists='append')
    print('Data Written')
    con.commit()
    con.close()  

def append_to_DB(con: sqlite3.Connection, data_to_insert: pd.DataFrame,data_metadata: race_meta,check: bool = True):
    # GET TABLES FROM THE CONNECTION TO THE DATABASE
    Categories = pd.read_sql_query('SELECT * FROM Categories',con)
    Racers = pd.read_sql_query('SELECT * FROM Racers',con)
    Races = pd.read_sql_query('SELECT * FROM Races',con)
    Series = pd.read_sql_query('SELECT * FROM Race_Series',con)
    cur = con.cursor()
    
    #Ensure the race isn't not duplicated
    duplicated =  check_db_for_duplicate_races(Races,data_metadata)
    
    #Check the race series and return the appropriate series_ID (or exit)
    data_metadata.series_id,series_name = suggest_race_series(Series,data_metadata)
    
    #Insert the race into the database!
    if not duplicated:
        SQL_insert = "INSERT INTO Races (Race_Name, Race_Date, Race_Distance, Race_Climb, Series_ID)"\
                    "VALUES (?,?,?,?,?);" 
        tryagain = True
        while tryagain and check:
            print('Inserting Race into DB')
            print('You are about to add the following race data:')
            print(f'Race Name:\t{data_metadata.race_name}')
            print(f'Race Date:\t{data_metadata.race_date}')
            print(f'Race Distance:\t{data_metadata.race_distance}')
            print(f'Race Climb: {data_metadata.race_climb}')
            print(f'Series ID: {data_metadata.series_id} ({series_name})') #TODO: Series ID should be added to race metadata
            key = input('Does everything look ok? Input any key to try again')
            if not key:
                tryagain = False                   
                cur.execute(SQL_insert,data_metadata.get_DB_entry)
                con.commit() # Commit here otherwise funny things happen
        if not check: #Commit straight away if not duplicated
            cur.execute(SQL_insert,data_metadata.get_DB_entry)
            con.commit() # Commit here otherwise funny things happen
    #RACES
    #Once we have multiple races with the same name, can't do a simple query to get race ID as
    # multiple values will be returned.
    # Instead you can query the cursor to find out what the number of the previous row added was
    Race_ID = cur.lastrowid
    query = 'SELECT * FROM Races '\
            'WHERE Race_Name = ?'
    Races = pd.read_sql_query(query,con,params=(data_metadata.race_name,))

    #RACERS
    #Get list of Racers not already in the database
    Racers_new = check_db_for_duplicate_racers(data_to_insert, Racers)
    
    # Checks data is ok
    print(f'About to insert {len(Racers_new.index)} racers')
    print(f'{len(data_to_insert.index) - len(Racers_new.index)} were already in the database')
    print('Examples below')
    print(Racers_new.head())
    if check:
        key  = input('Continue, or quit? Any key to quit')
        if key:
            exit()
        
    #Commit to the database
    Racers_new.to_sql('Racers',con,if_exists='append',index=False)
    #Get the new data, including the Racer_IDs
    Racers = pd.read_sql_query('SELECT * FROM Racers',con)
    
    #FINALISE DATA
    # Racers
    data_to_insert = pd.merge(data_to_insert,Racers,'inner',['Racer_Name','Racer_Name'])
    # Categories
    data_to_insert = pd.merge(data_to_insert,Categories,'left',['Cat_Name','Cat_Name'])
    # Races add Race_ID
    data_to_insert['Race_ID'] = Race_ID #Just get the first value and insert that over and over again

    data_to_insert = data_to_insert[['Race_ID','Racer_ID','Time','Cat_ID','Position']]
    # Trim to the columns needed
    
    # Add the statistics columns
    (data_to_insert['ZScore'],data_to_insert['ZScore_log'],data_to_insert['Percentile']) = analysis_tools.calculate_position_stats(data_to_insert['Time'].values)

    print(f'About to insert {len(data_to_insert.index)} records')
    print(data_to_insert.head())
    if check:
        key = input('Proceed? Any key to quit')
        if key:
            exit()
    data_to_insert.to_sql('Results',con,index=False,if_exists='append')
    print('Data Written')
    con.commit()
    con.close()  

def suggest_race_series(Series: pd.DataFrame, data_metadata: race_meta) -> Tuple[int,str]:
    # Find the closest matching race series
    Series['distances'] = Series['Series_Name'].map(lambda i: Levenshtein.distance(i,data_metadata.race_name))
    Series = Series.sort_values(by = 'distances', ascending=True)
    print(f'The closest matching race series is: {Series.iloc[0]["Series_Name"]}')
    if input('Is this the series you want to add the race to? (y/n)') == 'y':
        print(f'Adding the race to: {Series.iloc[0]["Series_Name"]}')
        return Series.iloc[0]['Series_ID'],Series.iloc[0]['Series_Name']
    else:
        print('Not currently adding more series, exiting')
        exit()
    
def check_db_for_duplicate_races(Races: pd.DataFrame, data_metadata: race_meta) -> bool:
    # Find possible matches for race name
    Races['Race_Name_distances'] = Races['Race_Name'].map(lambda i:Levenshtein.distance(i,data_metadata.race_name))
    # Find possible matches for race date
    Races['Race_Date_date'] = Races['Race_Date'].map(lambda x: dt.strptime(x,'%Y-%m-%d'))
    Races['Race_Date_distances'] = Races['Race_Date_date'].map(lambda x: abs(data_metadata.race_date_as_datetime - x).days)
    #Reducing race date distance to 6 to allow weekly parkruns to be added
    Races_Possible_duplicates = Races.query("Race_Name_distances <= 5 and Race_Date_distances <= 6")

    if not Races_Possible_duplicates.empty: # Only go here if it's not empty
        print('There may be existing races in the database do any of the below races look like the race you are trying to add?')
        print(Races_Possible_duplicates)
        response = input('y/(n)')
        if response == 'y':
            return True
    
    return False

def check_db_for_duplicate_racers(data_to_insert: pd.DataFrame, Racers: pd.DataFrame) -> pd.DataFrame:
    # Find the racers that aren't already included in the database with a left excluding join based on the name
    # The query looks for items that came from the left table only in the '_merge' column that is a result of indicator
    # Finally we drop this column as it's no longer needed
    # Only need Racer_Name from imported table
    Racers_new = data_to_insert.merge(Racers[['Racer_Name']],on="Racer_Name", how='left', indicator=True)\
        .query('_merge == "left_only"')\
        .drop('_merge',axis = 1)

    #Keep only the columns we need to add to the database
    Racers_new = Racers_new[['Racer_Name', 'Club']]

    #Ensure no NULL entries
    Racers_new = Racers_new.dropna(subset=['Racer_Name'])
    
    #Ensure no repeated entries
    Racers_new = Racers_new.drop_duplicates(subset=['Racer_Name'])
    return Racers_new
def get_racetek_api(URL: str) -> pd.DataFrame:
    response = requests.get(URL)
    # racetek API gives a list of entries (each a single list).
    fulljson = response.json()
    data = pd.DataFrame(data = fulljson,
                        columns = ['No.','First Name','Surname','Gender','Age Cat','Course','Club','Time',''],
                        )
    data['Position']=data.index
    return data
def get_avtiming_api(URL: str) -> pd.DataFrame:
    api_url = URL
    response = requests.get(api_url)
    fulljson = response.json()
    data = fulljson['data']
    
    #These aren't consistent but should be ok to spoof column headings, will automatically pad column names according to the size of the array
    col_names = ['Bib','Pos','Name','M/F','Cat','Cat_Pos','Club','Chip','Time']
    pads_needed = len(data[0]) - len(col_names)
    if pads_needed > 0:
        for i in range(pads_needed):
            col_names.append(f'Pad{i}')

    return pd.DataFrame(data,columns=col_names)
    