import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd
    from datetime import date
    from pathlib import Path

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
    from fellpace.race_overrides import set_race_override

    return (
        DB_PATH,
        Path,
        calculate_handicaps_for_entries,
        convert_Chase_ZScore_logs_avg,
        date,
        load_PR_entries,
        load_entries,
        load_models,
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
        set_race_override,
    )


@app.cell
def _(DB_PATH, setup_db):
    con = setup_db(DB_PATH)
    return (con,)


@app.cell
def _(date, mo):
    current_year = date.today().year
    year_options = [str(y) for y in range(2024, current_year + 1)]
    year_selector = mo.ui.dropdown(
        options=year_options,
        value=str(current_year),
        label="Prediction year",
    )
    use_parkrun = mo.ui.switch(value=True, label="Use PR entries")
    recalculate_button = mo.ui.run_button(label="Re-calculate from DB")
    mo.hstack([year_selector, use_parkrun, recalculate_button], widths=[1, 1, 1])
    return recalculate_button, use_parkrun, year_selector


@app.cell
def _(year_selector):
    prediction_year = int(year_selector.value)
    return (prediction_year,)


@app.cell
def _(
    Path,
    calculate_handicaps_for_entries,
    con,
    load_PR_entries,
    load_entries,
    mo,
    pd,
    prediction_year,
    process_PR_entries,
    process_entries,
    recalculate_button,
    use_parkrun,
):
    try:
        mode_tag = "pr" if use_parkrun.value else "no_pr"
        cache_dir = Path("notebooks") / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"entries_explorer_predictions_{prediction_year}_{mode_tag}.json"

        recompute = bool(recalculate_button.value)

        if cache_file.exists() and not recompute:
            processed_entries = pd.read_json(cache_file)
            status = mo.md(
                f"Loaded {len(processed_entries)} cached predictions for {prediction_year} ({mode_tag}). "
                "Press 'Re-calculate from DB' to refresh."
            )
        else:
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
            processed_entries.to_json(cache_file, orient="records", indent=2)
            status = mo.md(
                f"Calculated {len(processed_entries)} predictions for {prediction_year} ({mode_tag}) and saved cache to {cache_file}."
            )
    except Exception as exc:
        processed_entries = None
        status = mo.md(f"Could not generate entries for {prediction_year}: {exc}")

    status
    return (processed_entries,)


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
def _(comparison_table, has_chase_results, mo, pd, seconds_to_time_string):
    comparison_selector = None
    if comparison_table.empty:
        views = [mo.md("No summary table available for this year.")]
    else:
        table_display = comparison_table.copy()
        predicted_num = pd.to_numeric(table_display.get("Predicted_Time_seconds"), errors="coerce")
        actual_num = pd.to_numeric(table_display.get("Actual_Time_seconds"), errors="coerce")
        error_num = pd.to_numeric(table_display.get("Prediction_Error_seconds"), errors="coerce")

        table_display["Predicted_Time"] = predicted_num.apply(
            lambda x: seconds_to_time_string(x) if pd.notna(x) else "N/A"
        )
        table_display["Actual_Time"] = actual_num.apply(
            lambda x: seconds_to_time_string(x) if pd.notna(x) else "N/A"
        )
        table_display["Prediction_Error"] = error_num.apply(
            lambda x: f"{int(round(x)):+d}s" if pd.notna(x) else "N/A"
        )

        prediction_cols = [
            "Name",
            "Predicted_Time",
            "Handicap",
            "Off_time",
            "Num_results_used",
            "Num_excluded_results",
        ]
        prediction_cols = [c for c in prediction_cols if c in table_display.columns]
        predictions_view = table_display[prediction_cols].copy()

        views = [
            mo.md("### Prediction Summary"),
            mo.ui.table(predictions_view, selection=None),
        ]

        if has_chase_results:
            comparison_mask = actual_num.notna() & predicted_num.notna()
            compared_count = int(comparison_mask.sum())
            total_count = int(len(table_display))

            comparison_cols = [
                "Name",
                "Predicted_Time",
                "Actual_Time",
                "Prediction_Error",
                "Position",
            ]
            comparison_cols = [c for c in comparison_cols if c in table_display.columns]
            comparison_view = table_display.loc[comparison_mask, comparison_cols].copy()

            if compared_count > 0:
                mae = float(error_num[comparison_mask].abs().mean())
                bias = float(error_num[comparison_mask].mean())
                metrics_md = mo.md(
                    f"### Actual vs Predicted ({compared_count}/{total_count} matched)\n"
                    f"MAE: {mae:.1f}s | Mean Error (bias): {bias:+.1f}s"
                )
                comparison_selector = mo.ui.table(
                    comparison_view,
                    selection="single",
                    initial_selection=[0],
                )
                views.extend([metrics_md, comparison_selector])
            else:
                views.append(
                    mo.md(
                        "### Actual vs Predicted\n"
                        "No racers could be matched between predictions and Hallam Chase results for this year."
                    )
                )
        else:
            views.append(
                mo.md("### Actual vs Predicted\nNo Hallam Chase results found for the selected year.")
            )

    return comparison_selector, views


@app.cell
def _(mo, views):
    mo.vstack(views)
    return


@app.cell
def _(comparison_selector, comparison_table, pd):
    if comparison_selector is None or comparison_table.empty:
        selected_row = None
    else:
        selected_value = comparison_selector.value
        selected_rows = pd.DataFrame(selected_value)
        if selected_rows.empty or "Name" not in selected_rows.columns:
            selected_row = None
        else:
            selected_name = selected_rows.iloc[0]["Name"]
            selected = comparison_table[comparison_table["Name"] == selected_name]
            selected_row = selected.iloc[0] if not selected.empty else None
    return (selected_row,)


@app.cell
def _(
    con,
    convert_Chase_ZScore_logs_avg,
    mo,
    comparison_selector,
    has_chase_results,
    load_models,
    make_chase_prediction,
    plot_racer_entry,
    prediction_year,
    process_results_for_racer,
    racer_refresh_count,
    seconds_to_time_string,
    secure_racer_id,
    selected_row,
):
    # Explicitly read refresh token so this cell reruns after manual override writes.
    racer_refresh_count()
    plot_info = mo.md("Select a row in the Actual vs Predicted table to view that racer's prediction distributions.")
    figure = None
    racer_detail_table = None
    racer_id = None

    if comparison_selector is not None and selected_row is not None:
        racer_id, canonical_name = secure_racer_id(con, selected_row["Name"].lower().strip())
        if racer_id is not None:
            coeffs, covar, resid_stds = load_models(include_residuals=True)
            racer_results, _ = process_results_for_racer(
                con,
                coeffs,
                covar,
                resid_stds=resid_stds,
                racer_id=racer_id,
                prediction_year=prediction_year,
            )

            if racer_results is not None and not racer_results.empty:
                racer_results = racer_results[
                    racer_results["Season"] < prediction_year
                ].reset_index(drop=True)

                if not racer_results.empty:
                    detail_cols = [
                        "Racer_ID",
                        "Racer_Name",
                        "Race_Name",
                        "Season",
                        "Zpred_mu",
                        "Zpred_sig",
                        "include",
                        "outlier",
                    ]
                    detail_cols = [c for c in detail_cols if c in racer_results.columns]
                    racer_detail_table = racer_results[detail_cols].copy().sort_values(
                        [c for c in ["Season", "Race_Name"] if c in detail_cols],
                        ascending=[False, True] if "Season" in detail_cols and "Race_Name" in detail_cols else True,
                    )

                    chase_mu, chase_sig = make_chase_prediction(
                        racer_results.loc[racer_results["include"]],
                        prediction_year=prediction_year,
                        verbose=False,
                    )
                    prediction = chase_mu - (1.96 * chase_sig)
                    prediction_t = convert_Chase_ZScore_logs_avg(con, prediction)[0]

                    figure, ax = plot_racer_entry(
                        con=con,
                        racer_results=racer_results,
                        chase_mu=chase_mu,
                        chase_sig=chase_sig,
                        prediction_t=prediction_t,
                        racer_name=canonical_name,
                        prediction_year=prediction_year,
                    )

                    if (
                        has_chase_results
                        and selected_row is not None
                        and selected_row.get("Actual_Time_seconds")
                        == selected_row.get("Actual_Time_seconds")
                    ):
                        actual_t = float(selected_row["Actual_Time_seconds"])
                        ax.axvline(
                            actual_t,
                            color="tab:red",
                            linestyle="-.",
                            linewidth=2,
                            label="Actual time",
                        )
                        ax.legend()

                    if selected_row is not None:
                        pred_str = seconds_to_time_string(prediction_t)
                        if (
                            selected_row.get("Actual_Time_seconds")
                            == selected_row.get("Actual_Time_seconds")
                        ):
                            actual_str = seconds_to_time_string(
                                selected_row["Actual_Time_seconds"]
                            )
                            err = selected_row.get("Prediction_Error_seconds")
                            plot_info = mo.md(
                                f"Predicted: {pred_str} | Actual: {actual_str} | Error: {int(err):+d}s"
                            )
                        else:
                            plot_info = mo.md(f"Predicted: {pred_str} | Actual: N/A")
    return figure, plot_info, racer_detail_table, racer_id


@app.cell
def _(figure, include_checkboxes, mo, plot_info, racer_detail_display_table, racer_detail_table):
    outputs = [plot_info]
    if figure is not None:
        outputs.append(figure)
    if racer_detail_table is not None and not racer_detail_table.empty:
        outputs.extend(
            [
                mo.md("### Races Used vs Excluded"),
                mo.ui.table(racer_detail_display_table, selection=None),
            ]
        )
        if include_checkboxes is not None:
            outputs.extend([
                mo.md(
                    "Toggle checkboxes to include/exclude each race. Changes are saved "
                    "immediately and this racer is recalculated automatically."
                ),
                include_checkboxes,
            ])
    mo.vstack(outputs)
    return


@app.cell
def _(mo):
    racer_refresh_count, set_racer_refresh_count = mo.state(0)
    return racer_refresh_count, set_racer_refresh_count


@app.cell
def _(mo, racer_detail_table):
    if racer_detail_table is not None and not racer_detail_table.empty:
        detail_table_cols = [
            c for c in ["Race_Name", "Season", "Zpred_mu", "Zpred_sig", "include", "outlier"]
            if c in racer_detail_table.columns
        ]
        racer_detail_display_table = racer_detail_table[detail_table_cols].copy()

        include_checkboxes = mo.ui.array(
            [
                mo.ui.checkbox(
                    value=bool(row["include"]),
                    label=f"{row['Race_Name']} ({int(row['Season'])})",
                )
                for _, row in racer_detail_table.iterrows()
            ],
            label="Manual include / exclude",
        )
    else:
        include_checkboxes = None
        racer_detail_display_table = None
    return include_checkboxes, racer_detail_display_table


@app.cell
def _(
    include_checkboxes,
    racer_detail_table,
    racer_id,
    racer_refresh_count,
    set_race_override,
    set_racer_refresh_count,
):
    _wrote_override = False
    if (
        include_checkboxes is not None
        and racer_detail_table is not None
        and racer_id is not None
    ):
        for _i, (_, _row) in enumerate(racer_detail_table.iterrows()):
            if _i < len(include_checkboxes.value):
                _new_val = bool(include_checkboxes.value[_i])
                if bool(_row["include"]) != _new_val:
                    set_race_override(
                        racer_id,
                        int(_row["Season"]),
                        _row["Race_Name"],
                        _new_val,
                    )
                    _wrote_override = True
    if _wrote_override:
        set_racer_refresh_count(racer_refresh_count() + 1)
    return


if __name__ == "__main__":
    app.run()
