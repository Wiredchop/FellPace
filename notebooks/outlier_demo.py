import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    from fellpace.db.db_setup import setup_db
    from fellpace.config import DB_PATH, ENTRIES_PATH
    from fellpace.entries import load_PR_entries, process_PR_entries, process_results_for_racer
    from fellpace.modelling.training import load_models
    from fellpace.filter import filter_race_results
    from fellpace.plotting.racetimes import plot_racers_results
    return (
        DB_PATH,
        filter_race_results,
        load_PR_entries,
        load_models,
        mo,
        pd,
        plot_racers_results,
        process_PR_entries,
        process_results_for_racer,
        setup_db,
    )


@app.cell
def _(DB_PATH, setup_db):
    con = setup_db(DB_PATH)
    return (con,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Edge Cases from 2025 entries to test outlier removal
    - Rachel Smith - Slower runner with definite outlier
    - Nick Hails - Faster runner with definite outlier
    - Pat Goodall - Lots of entries no outliers slower more recently
    - Nick Burns - Lots of tight fast results
    - Helen Young - Wide distributed results no obvious outliers
    - James Thompson - two outliers at different levels
    """
    )
    return


@app.cell
def _(mo):
    dropdown_year = mo.ui.dropdown(label="Choose year of race", options = [2024, 2025], value = 2025)
    mo.hstack([dropdown_year])
    return (dropdown_year,)


@app.cell
def _(dropdown_year, load_PR_entries, mo, process_PR_entries):
    PR_entries = load_PR_entries(year_of_entry=dropdown_year.value)
    PR_entries_processed = process_PR_entries(PR_entries, year_of_entry=2025, forename_surname=True)

    dropdown_name = mo.ui.dropdown(label="Racer to examine", options = sorted(PR_entries_processed['Name']), value=sorted(PR_entries_processed['Name'])[0])
    mo.hstack([dropdown_name])
    return (dropdown_name,)


@app.cell
def _(
    con,
    dropdown_name,
    filter_race_results,
    load_models,
    pd,
    process_results_for_racer,
):
    coeffs, covar = load_models()
    included, excluded = process_results_for_racer(con, dropdown_name.value,coeffs, covar )
    all = pd.concat([included, excluded], axis=0)
    all = all.drop(columns=['outlier'])
    # Filter based on excluded and PR only
    inc_orig, exc = filter_race_results(all)

    return (inc_orig,)


@app.cell
def _():
    return


@app.cell
def _(mo):
    slider_outlier = mo.ui.slider(
        start= 0,
        stop = 5,
        step = 1,
        label = "Outliers to remove"
    )
    return (slider_outlier,)


@app.cell
def _(mo, slider_outlier):
    mo.hstack([slider_outlier, mo.md(f"Remove: {slider_outlier.value}")],widths=[0.3,1])
    return


@app.cell
def _(inc_orig, slider_outlier):
    outliers_to_remove = slider_outlier.value
    print(f"Going to remove {outliers_to_remove} outliers")
    inc_trimmed = (inc_orig
                   .sort_values('Zpred_mu')
                   .iloc[0:len(inc_orig)-outliers_to_remove]
                  )
    inc = inc_trimmed
    inc['mean_diff'] = inc['Zpred_mu'] - inc['Zpred_mu'].mean()
    return inc, inc_trimmed


@app.cell
def _(inc, mo):
    md_mean = mo.md(f"Mean Z value: **{inc['Zpred_mu'].mean():0.2f}**")

    md_prop_below_mean = mo.md(f"{(inc['mean_diff'] < 0).sum() / len(inc):.2%} of runs below the mean")

    mo.vstack([md_mean, md_prop_below_mean])
    return


@app.cell
def _(inc, mo):
    Q1 = inc['Zpred_mu'].quantile(0.25)
    Q3 = inc['Zpred_mu'].quantile(0.75)
    IQR = Q3-Q1
    Outlier_theshold = Q3 + 1.5 * IQR
    mo.md(f"Outlier threshold is above {Outlier_theshold:0.2f}")
    return


@app.cell
def _(inc_trimmed):
    inc_trimmed
    return


@app.cell
def _(con, inc, plot_racers_results):
    import matplotlib.pyplot as plt
    _, ax = plt.subplots(figsize=(10, 6))    
    plot_racers_results(inc, con, ax = ax)
    ax
    return


if __name__ == "__main__":
    app.run()
