import Levenshtein
import sqlite3
import pandas as pd
import numpy as np

con = sqlite3.connect('fellpace.db')



#Get a list of the racers and racer_ids from the database
Get_racer_query = """
SELECT Racer_ID, Racer_Name
FROM Racers
"""
racers = pd.read_sql_query(Get_racer_query,con)
removed_IDs = []
for i, racer in racers.iterrows():
    racer_name = str.lower(racer['Racer_Name'])
    racer_ID = racer['Racer_ID']
    if racer_name.lower().startswith(('a','b','c','d','e','f','g','h','i','j','k','l','m')):
        continue
        
    if str(racer_ID) in removed_IDs:
        continue
    names = list(racers['Racer_Name'].values)
    IDs = list(racers['Racer_ID'].values)
    d = [Levenshtein.distance(racer_name,str.lower(r)) for r in racers['Racer_Name']]
    d.pop(i)
    names.pop(i)
    IDs.pop(i)
    min_d = np.min(d)
    if min_d > 1:
        print(f'Skipping: {racer_name}\n')
        continue
    locs = np.where(d == min_d)[0]
    
    # GET RACER STATS
    racer_stat_query = """
    SELECT avg(Zscore_log) as ZScore, count(Zscore_log) as races
    FROM Results
    WHERE Racer_ID = ?"""
    racer_zscore = pd.read_sql(racer_stat_query,con,params=(racer_ID,))
    print(f'Possible repeats for {racer_name}, avg ZS {racer_zscore.ZScore[0]}, races {racer_zscore.races[0]}:\n')
    match_IDs = [''] * len(locs)
    for i, loc in enumerate(locs):
        match_IDs[i] = str(IDs[loc])
        match_zscore = pd.read_sql(racer_stat_query,con,params=(match_IDs[i],))
        print(f'{i}:\t{names[loc]}, ZScore: {match_zscore.ZScore[0]}, races: {match_zscore.races[0]}')
    resp = input('Do any of these match? Enter index if so, otherwise hit enter to contiue.')
    
    if resp == '':
        continue
    else:
        resp = int(resp)
    matching_racer = names[locs[resp]]
    print(f'Removing {resp}: {matching_racer} from the database.')
    #Replace the IDs in the results with the original IDs
    results_ID_update = """
    UPDATE Results
    SET Racer_ID = ?
    WHERE Racer_ID = ?
    """
    
    #Replace the IDs in the chase results with the original IDs
    chase_results_ID_update = """
    UPDATE Results_Chase
    SET Racer_ID = ?
    WHERE Racer_ID = ?
    """
    cur = con.cursor()
    cur.execute(results_ID_update,(racer_ID,match_IDs[resp]))
    cur.execute(chase_results_ID_update,(racer_ID,match_IDs[resp]))
    
    # Remove the matching racer from the database
    racer_remove = """
    DELETE FROM Racers
    WHERE Racer_ID = ?
    """
    cur.execute(racer_remove,(match_IDs[resp],))
    con.commit()
    removed_IDs.append(match_IDs[resp])