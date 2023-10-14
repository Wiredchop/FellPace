# Fucked up calculation of Zscore, have to get all the data back and inject back into the database
# Get list of races
import pandas as pd
import convert_tools
import analysis_tools
import sqlite3

#Connect
con = sqlite3.connect('fellpace.db')
cur = con.cursor()

#Get the Races table
Races = pd.read_sql('SELECT * FROM Races',con)

Results_query = 'SELECT Results.Result_ID, Results.Time '\
                'FROM Results WHERE Race_ID = ?'
Score_query = 'UPDATE Results '\
              'SET ZScore = ?, Percentile = ? '\
              'WHERE Result_ID = ?'
for index, Race in Races.iterrows():
    Results = pd.read_sql(Results_query,con,params=(Race['Race_ID'],))
    Results['NewZ'],Results['NewP'] = analysis_tools.calculate_position_stats(Results['Time'].values)
    for index, Result in Results.iterrows():
        cur.execute(Score_query,(Result['NewZ'],Result['NewP'],Result['Result_ID']))
    # Not sure how to change multiple results at once so will loop with multiple queries instead
    con.commit()
con.close()