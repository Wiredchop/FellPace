import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    from pathlib import Path

    return mo, pd


app._unparsable_cell(
    r"""
    from fellpace.config import ENTRIES_PATH
    from fellpace.plotting.racetimes import 
    """,
    name="_"
)


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

    return racer_predictions_message, racer_result_message, racer_results_all


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


if __name__ == "__main__":
    app.run()
