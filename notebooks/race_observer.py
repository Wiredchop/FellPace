import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    from pathlib import Path

    return mo, pd


@app.cell
def _():
    from fellpace.config import ENTRIES_PATH, DB_PATH
    from fellpace.plotting.racetimes import plot_racer_entry
    from fellpace.db.db_setup import setup_db
    from fellpace.analysis_tools import convert_Chase_ZScore_logs_avg
    return (
        DB_PATH,
        ENTRIES_PATH,
        convert_Chase_ZScore_logs_avg,
        plot_racer_entry,
        setup_db,
    )


@app.cell
def _(DB_PATH, setup_db):
    con = setup_db(DB_PATH)
    return (con,)


@app.cell
def _(mo):
    dropdown_year = mo.ui.dropdown(
        options = [2023,2024,2025],
        value = 2025,
        label= "Choose the year to examine"
    )
    return (dropdown_year,)


@app.cell
def _(dropdown_year, mo):
    mo.hstack([dropdown_year])
    return


@app.cell
def _(dropdown_year):
    year_of_entry = dropdown_year.value
    return (year_of_entry,)


@app.cell
def _(ENTRIES_PATH, year_of_entry):
    results_filepath = ENTRIES_PATH / f"racer_results_{year_of_entry}.json"
    predictions_filepath = ENTRIES_PATH / f"racer_predictions_{year_of_entry}.json"
    return predictions_filepath, results_filepath


@app.cell
def _(mo, pd, predictions_filepath, results_filepath, year_of_entry):
    if results_filepath.exists:
        racer_results_all = pd.read_json(results_filepath)
        racer_result_message = mo.md("Results loaded successfully")
    else:
        racer_result_message = mo.md(f"No results file found, please process results for year: {year_of_entry}")
        racer_results_all = pd.DataFrame()

    if predictions_filepath.exists:
        racer_predictions_all = pd.read_json(predictions_filepath)
        racer_predictions_message = mo.md("Predictions loaded successfully")
    else:
        racer_predictions_message = mo.md(f"No predictions file found, please process results for year: {year_of_entry}")
        racer_predictions_all = pd.DataFrame()

    return (
        racer_predictions_all,
        racer_predictions_message,
        racer_result_message,
        racer_results_all,
    )


@app.cell
def _(mo, racer_predictions_message, racer_result_message):
    mo.vstack(
        [
            racer_result_message,
            racer_predictions_message
        ])
    return


@app.cell
def _(mo, racer_results_all):
    sorted_names = racer_results_all['Racer_Name'].sort_values().unique() 
    dropdown_racer_name = mo.ui.dropdown(
        options = sorted_names,
        value = sorted_names[0],
        label = "Choose racer to examine"
    )
    
    return (dropdown_racer_name,)


@app.cell
def _(dropdown_racer_name, mo):
    mo.hstack([dropdown_racer_name])
    return


@app.cell
def _(dropdown_racer_name):
    racer_name = dropdown_racer_name.value
    return (racer_name,)


@app.cell
def _(racer_name, racer_results_all):
    racer_results = racer_results_all.loc[racer_results_all['Racer_Name'] == racer_name]
    return (racer_results,)


@app.cell
def _(racer_name, racer_predictions_all):
    racer_prediction = racer_predictions_all.loc[racer_predictions_all['Racer_Name'] == racer_name.lower()]
    mu = racer_prediction['chase_mu']
    sig = racer_prediction['chase_sig']
    return mu, sig


@app.cell
def _(con, convert_Chase_ZScore_logs_avg, mu, sig):
    prediction = mu - 1.96 * sig
    prediction_t = convert_Chase_ZScore_logs_avg(con=con, Zscore_logs=prediction)
    return (prediction_t,)


@app.cell
def _(
    con,
    mu,
    plot_racer_entry,
    prediction_t,
    racer_name,
    racer_results,
    sig,
    year_of_entry,
):
    f =plot_racer_entry(
        con=con,
        racer_results=racer_results,
        chase_mu= mu,
        chase_sig=sig,
        prediction_t=prediction_t,
        racer_name=racer_name,
        prediction_year=year_of_entry
    )
    f
    return


if __name__ == "__main__":
    app.run()
