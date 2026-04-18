import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd
    from datetime import date

    from fellpace.config import DB_PATH
    from fellpace.db.db_setup import setup_db
    from fellpace.entries import (
        load_entries,
        load_PR_entries,
        process_PR_entries,
        process_entries,
        process_results_for_racer,
    )
    from fellpace.extract.racers import secure_racer_id
    from fellpace.handicaps import calculate_handicaps_for_entries
    from fellpace.modelling.prediction import make_chase_prediction
    from fellpace.modelling.training import load_models
    from fellpace.analysis_tools import convert_Chase_ZScore_logs_avg
    from fellpace.plotting.racetimes import plot_racer_entry
    from fellpace.convert_tools import seconds_to_time_string

    return (
        DB_PATH,
        calculate_handicaps_for_entries,
        convert_Chase_ZScore_logs_avg,
        date,
        load_entries,
        load_models,
        load_PR_entries,
        make_chase_prediction,
        mo,
        pd,
        plot_racer_entry,
        plt,
        process_PR_entries,
        process_entries,
        process_results_for_racer,
        seconds_to_time_string,
        secure_racer_id,
        setup_db,
    )


@app.cell
def _(DB_PATH, setup_db):
    con = setup_db(DB_PATH)
    return (con,)


@app.cell
def _(date, mo):
    current_year = date.today().year
    year_selector = mo.ui.number(
        start=2010,
        stop=current_year + 1,
        step=1,
        value=current_year,
        label="Prediction year",
    )
    use_parkrun = mo.ui.switch(value=True, label="Use PR entries")
    mo.hstack([year_selector, use_parkrun], widths=[1, 1])
    return use_parkrun, year_selector


@app.cell
def _(year_selector):
    prediction_year = int(year_selector.value)
    return (prediction_year,)


@app.cell
def _(
    calculate_handicaps_for_entries,
    con,
    load_entries,
    load_PR_entries,
    mo,
    prediction_year,
    process_PR_entries,
    process_entries,
    use_parkrun,
):
    try:
        if use_parkrun.value:
            source_entries = load_PR_entries(year_of_entry=prediction_year)
            source_entries = process_PR_entries(
                source_entries,
                year_of_entry=prediction_year,
                forename_surname=True,
            )
        else:
            source_entries = load_entries(year_of_entry=prediction_year)

        processed_entries = process_entries(
            source_entries,
            con,
            year_of_entry=prediction_year,
            with_parkrun=use_parkrun.value,
        )
        processed_entries = calculate_handicaps_for_entries(processed_entries)
        processed_entries = processed_entries.sort_values(
            by="Predicted_Time_seconds", ascending=True
        )
        status = mo.md(
            f"Loaded {len(source_entries)} entries and generated {len(processed_entries)} predictions for {prediction_year}."
        )
    except Exception as exc:
        source_entries = None
        processed_entries = None
        status = mo.md(f"Could not generate entries for {prediction_year}: {exc}")

    status
    return processed_entries, source_entries


@app.cell
def _(con, pd, prediction_year, processed_entries):
    if processed_entries is None or processed_entries.empty:
        comparison_table = pd.DataFrame()
        has_chase_results = False
    else:
        sql = """
        SELECT
            lower(R.Racer_Name) AS name_key,
            R.Racer_Name,
            RC.Time AS Actual_Time_seconds,
            RC.Position
        FROM Results_Chase AS RC
        JOIN Chases AS C
            ON C.Chase_ID = RC.Chase_ID
        JOIN Racers AS R
            ON R.Racer_ID = RC.Racer_ID
        WHERE CAST(strftime('%Y', C.Chase_Date) AS INTEGER) = ?
        """
        chase_results = pd.read_sql_query(sql, con, params=(prediction_year,))
        has_chase_results = not chase_results.empty

        summary = processed_entries.copy()
        summary["name_key"] = summary["Name"].str.lower().str.strip()

        if has_chase_results:
            comparison_table = summary.merge(
                chase_results[["name_key", "Actual_Time_seconds", "Position"]],
                on="name_key",
                how="left",
            )
            comparison_table["Prediction_Error_seconds"] = (
                comparison_table["Predicted_Time_seconds"]
                - comparison_table["Actual_Time_seconds"]
            )
        else:
            comparison_table = summary
            comparison_table["Actual_Time_seconds"] = pd.NA
            comparison_table["Position"] = pd.NA
            comparison_table["Prediction_Error_seconds"] = pd.NA

        keep_cols = [
            "Name",
            "Predicted_Time",
            "Predicted_Time_seconds",
            "Handicap",
            "Off_time",
            "Num_results_used",
            "Num_excluded_results",
            "Actual_Time_seconds",
            "Position",
            "Prediction_Error_seconds",
        ]
        comparison_table = comparison_table[[c for c in keep_cols if c in comparison_table.columns]]
        comparison_table = comparison_table.sort_values("Predicted_Time_seconds", na_position="last")

    return comparison_table, has_chase_results


@app.cell
def _(comparison_table, has_chase_results, mo, seconds_to_time_string):
    if comparison_table.empty:
        mo.md("No summary table available for this year.")
        return

    table_display = comparison_table.copy()
    for col in ["Predicted_Time_seconds", "Actual_Time_seconds", "Prediction_Error_seconds"]:
        if col in table_display.columns:
            if col == "Prediction_Error_seconds":
                table_display[col] = table_display[col].apply(
                    lambda x: (
                        f"{int(x):+d}s" if x == x else "N/A"
                    )
                )
            else:
                table_display[col] = table_display[col].apply(
                    lambda x: seconds_to_time_string(x) if x == x else "N/A"
                )

    header = "### Prediction Summary"
    if has_chase_results:
        header += " (including actual Hallam Chase results)"

    mo.vstack([
        mo.md(header),
        table_display,
    ])
    return


@app.cell
def _(comparison_table, mo):
    if comparison_table.empty:
        racer_picker = mo.ui.dropdown(options=["No racers available"], value="No racers available", label="Racer")
    else:
        racer_names = sorted(comparison_table["Name"].dropna().astype(str).unique().tolist())
        racer_picker = mo.ui.dropdown(options=racer_names, value=racer_names[0], label="Racer")

    mo.hstack([racer_picker])
    return (racer_picker,)


@app.cell
def _(comparison_table, racer_picker):
    if comparison_table.empty:
        selected_row = None
    else:
        selected = comparison_table[comparison_table["Name"] == racer_picker.value]
        selected_row = selected.iloc[0] if not selected.empty else None
    return (selected_row,)


@app.cell
def _(
    con,
    convert_Chase_ZScore_logs_avg,
    has_chase_results,
    load_models,
    make_chase_prediction,
    plot_racer_entry,
    plt,
    prediction_year,
    process_results_for_racer,
    racer_picker,
    seconds_to_time_string,
    secure_racer_id,
    selected_row,
):
    if racer_picker.value == "No racers available":
        return

    racer_id, canonical_name = secure_racer_id(con, racer_picker.value.lower().strip())
    if racer_id is None:
        return

    coeffs, covar, resid_stds = load_models(include_residuals=True)
    racer_results, chase_results = process_results_for_racer(
        con,
        coeffs,
        covar,
        resid_stds=resid_stds,
        racer_id=racer_id,
    )

    if racer_results is None or racer_results.empty:
        return

    racer_results = racer_results[racer_results["Season"] < prediction_year].reset_index(drop=True)
    if racer_results.empty:
        return

    chase_mu, chase_sig = make_chase_prediction(
        racer_results.loc[racer_results["include"]],
        prediction_year=prediction_year,
        verbose=False,
    )
    prediction = chase_mu - (1.96 * chase_sig)
    prediction_t = convert_Chase_ZScore_logs_avg(con, prediction)[0]

    plot_racer_entry(
        con=con,
        racer_results=racer_results,
        chase_mu=chase_mu,
        chase_sig=chase_sig,
        prediction_t=prediction_t,
        racer_name=canonical_name,
        prediction_year=prediction_year,
    )

    if has_chase_results and selected_row is not None and selected_row.get("Actual_Time_seconds") == selected_row.get("Actual_Time_seconds"):
        actual_t = float(selected_row["Actual_Time_seconds"])
        ax = plt.gca()
        ax.axvline(actual_t, color="tab:red", linestyle="-.", linewidth=2, label="Actual time")
        ax.legend()

    plt.gcf()

    if selected_row is not None:
        pred_str = seconds_to_time_string(prediction_t)
        if selected_row.get("Actual_Time_seconds") == selected_row.get("Actual_Time_seconds"):
            actual_str = seconds_to_time_string(selected_row["Actual_Time_seconds"])
            err = selected_row.get("Prediction_Error_seconds")
            print(f"Predicted: {pred_str} | Actual: {actual_str} | Error: {int(err):+d}s")
        else:
            print(f"Predicted: {pred_str} | Actual: N/A")

    return


if __name__ == "__main__":
    app.run()
