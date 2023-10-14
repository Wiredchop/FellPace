import sqlite3
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as mp


con = sqlite3.connect('fellpace.db')

Year = 2022
sql_extract =   'Select R.Racer_Name, RC.Time, RC.Handicap FROM Results_Chase as RC '\
                'INNER JOIN Racers as R '\
                'ON RC.Racer_ID = R.Racer_ID '\
                'INNER JOIN Chases as C '\
                'ON C.Chase_ID = RC.Chase_ID '\
                'WHERE strftime("%Y",C.Chase_Date) = ?'

data = pd.read_sql(sql_extract,con,params= (str(Year),))


sb.set_theme()

sb.relplot(data = data, x='Handicap', y='Time')
mp.show()