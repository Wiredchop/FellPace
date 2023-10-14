import sqlite3
import pandas as pd
con = sqlite3.connect('fellpace.db')
cur = con.cursor()

cur.execute("DELETE FROM Racers WHERE Racer_ID >= 4560")
con.commit()
con.close()