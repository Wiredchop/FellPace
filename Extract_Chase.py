import pandas as pd
import convert_tools
import analysis_tools
import FellPace_tools
import sqlite3
#file = input('Enter the name of the HC file')
file = 'Chase 2016.csv'
dirfile = './csv/HC/' + file
data = pd.read_csv(dirfile)

# The Hallam Chase files must have the following column names:
# Name: Athlete name
# Club: Club
# Class: Category
# Time Correct: Handicapped Time, Regex: ^[0-9]{2}:[0-9]{2}
# Actual Correct: Actual Time
# Finish: Position -- This is the position after Handicap has been applied

Categories = convert_tools.convert_categories(data['Class'].values)
HTime = convert_tools.time_string_to_seconds(data['Time Correct'].values)
ATime = convert_tools.time_string_to_seconds(data['Actual Correct'].values)
Handicap = HTime - ATime
Names = data['Name'].values


Date = input('What was the date of the Chase? (yyyy-mm-dd)\n')

#Connect to DB
con = sqlite3.connect('fellpace.db')
cur = con.cursor()

#Check
key = input(f'About to insert Date: {Date} any key to cancel')
if key:
    exit()

#Commit Chase to DB
query_insert = 'INSERT INTO Chases (Chase_Date) '\
                'VALUES (?)'
cur.execute(query_insert,(Date,))
#Get the commit back to get Chase_ID
query_retrieve = 'SELECT Chase_ID from Chases '\
                 'WHERE Chase_Date = ?'
The_Chase = pd.read_sql(query_retrieve,con,params=(Date,))
id = The_Chase['Chase_ID'][0]

# Form the data to Insert
data_to_insert = pd.DataFrame({'Time':ATime,'Chase_ID':id,'Cat_Name':Categories,'Club':data['Club'],'Handicap':Handicap,'Racer_Name':Names,'Position':data['Finish']})

FellPace_tools.append_CHASE(con,data_to_insert)