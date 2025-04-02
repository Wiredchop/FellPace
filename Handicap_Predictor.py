import pandas as pd
import numpy as np
from pathlib import Path
from IPython.display import display, HTML
import seaborn as sns
import matplotlib.pyplot as plt
import math

from fellpace.config import DB_PATH
from fellpace.db.db_setup import setup_db
from fellpace.extract.zscores import extract_all_zscore_data
from fellpace.modelling.prediction import get_predicted_times
from fellpace.plotting.races import plot_all_race_Zscores
from fellpace.stats.polyfit import get_coeffs
from fellpace.stats.ransac import add_inliers




def axis_time(seconds_in, pos):
    seconds = divmod(seconds_in, 60)[1]
    minutes = divmod(seconds_in, 60)[0]
    hours = divmod(minutes, 60)[0]
    return f"{minutes:02.0f}:{seconds:02.0f}"

def generate_predictions(con, coeffs, entries_file):
    a4_dims = (8, 12)
    fig, ax = plt.subplots(figsize=a4_dims)
    predictions = pd.DataFrame()
    with open(entries_file) as entries:
        for entry in entries:
            entry = entry.rstrip()
            predictions = pd.concat([predictions, get_predicted_times(con, coeffs, entry, 2023)])
            predictions = pd.concat([predictions, get_predicted_times(con, coeffs, entry, 2022)])
    predictions = predictions.sort_values(['Racer_Name', 'Season', 'Race_Name'])
    predictions['Racer_Name'] = predictions['Racer_Name'].str.title()
    predictions['ParkRun'] = predictions['Race_Name'].str.contains("PR").replace({True: 'Parkrun', False: 'Race'})
    f = sns.stripplot(jitter=False, ax=ax, data=predictions.sort_values(['Predicted Time']), x="Predicted Time", y="Racer_Name", hue='ParkRun')
    ax = plt.gca()
    ax.xaxis.set_major_formatter(axis_time)
    plt.tight_layout()
    f.get_figure().savefig("predictions.png")
    return predictions

def generate_other_predictions(con, coeffs, entries_file):
    a4_dims = (8, 12)
    fig, ax = plt.subplots(figsize=a4_dims)
    predictions_other = pd.DataFrame()
    with open(entries_file) as entries:
        for entry in entries:
            entry = entry.rstrip()
            predictions_other = pd.concat([predictions_other, get_predicted_times(con, coeffs, entry)])
    predictions_other = predictions_other.loc[predictions_other['Season'] != 2023]
    predictions_other = predictions_other.sort_values(['Racer_Name', 'Season', 'Race_Name'])
    predictions_other['Racer_Name'] = predictions_other['Racer_Name'].str.title()
    predictions_other['ParkRun'] = predictions_other['Race_Name'].str.contains("PR").replace({True: 'Parkrun', False: 'Race'})
    f = sns.stripplot(data=predictions_other.sort_values(['Predicted Time']), x="Predicted Time", y="Racer_Name", hue='ParkRun')
    ax = plt.gca()
    ax.xaxis.set_major_formatter(axis_time)
    f.get_figure().savefig("predictions_other.png")
    return predictions_other

def format_time(group):
    group = group.sort_values(["Predicted Time"], ascending=True)
    group = group.reset_index(drop=True)
    group.index = group.index.map(lambda x: f"{x+1:02d}")
    grouped = pd.DataFrame()
    grouped["Racer Name"] = group["Racer_Name"]
    grouped["RaceTime"] = group["Race_Name"] + " (" + group['Season'].astype('str') + "): " + seconds_to_time_string(group["Predicted Time"])
    grouped["Race"] = "Race " + group.index.astype('str')
    return grouped

def prepare_predictions_for_excel(predictions):
    BestTimes = predictions.groupby("Racer_Name")["Predicted Time"].min().reset_index()
    predictions_export = predictions.groupby("Racer_Name", as_index=False, sort=False).apply(format_time).pivot(index="Racer Name", columns="Race", values="RaceTime")
    predictions_export = pd.merge(predictions_export, BestTimes, left_on="Racer Name", right_on="Racer_Name").sort_values("Predicted Time").drop("Predicted Time", axis=1)
    return predictions_export

def save_predictions_to_excel(predictions_export, entries_file, output_file):
    with open(entries_file, 'r') as entries:
        original_names = entries.readlines()
    original_names = pd.DataFrame(original_names, columns=['Entry List'])
    original_names['Entry List'] = original_names['Entry List'].str.rstrip().str.title()
    original_names = predictions_export.merge(original_names, how='left', left_on='Racer_Name', right_on='Entry List')
    original_names.to_excel(output_file)

def main():
    con = setup_db(DB_PATH)
    data_Zs = extract_all_zscore_data(con)
    data_Zs = add_inliers(data_Zs)
    plot_all_race_Zscores(data_Zs)
    coeffs = get_coeffs(data_Zs)
    entries_file = Path(r"Entries/entries_18.5.24")
    predictions = generate_predictions(con, coeffs, entries_file)
    predictions_other = generate_other_predictions(con, coeffs, entries_file)
    predictions_export = prepare_predictions_for_excel(predictions)
    predictions_other_export = prepare_predictions_for_excel(predictions_other)
    save_predictions_to_excel(predictions_export, entries_file, './entries/2024/Predictions.xlsx')
    save_predictions_to_excel(predictions_other_export, entries_file, './entries/2024/Predictions_old.xlsx')

if __name__ == "__main__":
    main()
