import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd
    import re
    from datetime import date
    from pathlib import Path

    from fellpace.config import DB_PATH, ENTRIES_PATH
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
    from fellpace.prediction_time_overrides import (
        load_prediction_time_overrides,
        set_prediction_time_override,
        clear_prediction_time_override,
    )

    return (
        DB_PATH,
        ENTRIES_PATH,
        Path,
        calculate_handicaps_for_entries,
        clear_prediction_time_override,
        convert_Chase_ZScore_logs_avg,
        date,
        load_PR_entries,
        load_entries,
        load_models,
        load_prediction_time_overrides,
        make_chase_prediction,
        mo,
        pd,
        plot_racer_entry,
        process_PR_entries,
        process_entries,
        process_results_for_racer,
        re,
        seconds_to_time_string,
        secure_racer_id,
        set_prediction_time_override,
        set_race_override,
        setup_db,
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
    export_handicaps_button = mo.ui.run_button(label="Export handicaps CSV")
    mo.hstack([year_selector, use_parkrun, recalculate_button, export_handicaps_button], widths=[1, 1, 1, 1])
    return (
        export_handicaps_button,
        recalculate_button,
        use_parkrun,
        year_selector,
    )


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
    secure_racer_id,
    use_parkrun,
):
    try:
        mode_tag = "pr" if use_parkrun.value else "no_pr"
        cache_dir = Path("notebooks") / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"entries_explorer_predictions_{prediction_year}_{mode_tag}.json"

        recompute = bool(recalculate_button.value)

        if use_parkrun.value:
            source_entries = load_PR_entries(year_of_entry=prediction_year)
            source_entries = process_PR_entries(
                source_entries,
                year_of_entry=prediction_year,
                forename_surname=True,
            )
        else:
            source_entries = load_entries(year_of_entry=prediction_year)

        entry_url_map = {}
        entry_pr_time_map = {}
        if "Name" in source_entries.columns:
            def canonicalize_entry_name(name: str) -> str:
                racer_id, resolved_name = secure_racer_id(con, name)
                if racer_id is not None and isinstance(resolved_name, str):
                    return resolved_name.lower().strip()
                return name

            normalized_entries = (
                source_entries
                .dropna(subset=["Name"])
                .assign(Name=lambda df: df["Name"].str.lower().str.strip())
                .assign(Name=lambda df: df["Name"].map(canonicalize_entry_name))
                .drop_duplicates(subset=["Name"], keep="first")
            )
            if "URL" in normalized_entries.columns:
                entry_url_map = normalized_entries.set_index("Name")["URL"].to_dict()
            if "PR_time" in normalized_entries.columns:
                entry_pr_time_map = normalized_entries.set_index("Name")["PR_time"].to_dict()

        expected_count = int(len(source_entries))
        cache_valid = False
        cache_invalid_reason = ""

        if cache_file.exists() and not recompute:
            try:
                processed_entries = pd.read_json(cache_file, orient="records")
                required_cols = {"Name", "Predicted_Time_seconds", "Predicted_Time"}
                missing_cols = sorted(required_cols.difference(set(processed_entries.columns)))
                if missing_cols:
                    cache_invalid_reason = f"cache missing columns: {missing_cols}"
                elif expected_count > 0 and len(processed_entries) < max(1, int(expected_count * 0.5)):
                    cache_invalid_reason = (
                        f"cache row count {len(processed_entries)} is unexpectedly small "
                        f"for {expected_count} source entries"
                    )
                else:
                    cache_valid = True
            except Exception as cache_exc:
                cache_invalid_reason = f"cache read failed: {cache_exc}"

        if cache_valid:
            status = mo.md(
                f"Loaded {len(processed_entries)} cached predictions for {prediction_year} ({mode_tag}). "
                "Press 'Re-calculate from DB' to refresh."
            )
        else:
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
            if cache_file.exists() and not recompute and cache_invalid_reason:
                status = mo.md(
                    f"Recalculated {len(processed_entries)} predictions for {prediction_year} ({mode_tag}) "
                    f"because {cache_invalid_reason}. Saved cache to {cache_file}."
                )
            else:
                status = mo.md(
                    f"Calculated {len(processed_entries)} predictions for {prediction_year} ({mode_tag}) and saved cache to {cache_file}."
                )
    except Exception as exc:
        processed_entries = None
        entry_url_map = {}
        entry_pr_time_map = {}
        status = mo.md(f"Could not generate entries for {prediction_year}: {exc}")

    status
    return entry_pr_time_map, entry_url_map, processed_entries


@app.cell
def _(
    calculate_handicaps_for_entries,
    con,
    load_prediction_time_overrides,
    pd,
    prediction_year,
    processed_entries,
    racer_refresh_count,
    seconds_to_time_string,
):
    # Re-read when manual include/exclude or predicted-time overrides are changed.
    racer_refresh_count()

    if processed_entries is None or processed_entries.empty:
        comparison_table = pd.DataFrame()
        has_chase_results = False
    else:
        _overrides = load_prediction_time_overrides()
        summary = processed_entries.copy()
        applied = 0
        for idx, summary_row in summary.iterrows():
            _summary_name_key = str(summary_row["Name"]).lower().strip()
            _override_key = f"{_summary_name_key}|{int(prediction_year)}"
            if _override_key in _overrides:
                _override_seconds = float(_overrides[_override_key])
                summary.at[idx, "Predicted_Time_seconds"] = _override_seconds
                summary.at[idx, "Predicted_Time"] = seconds_to_time_string(_override_seconds)
                applied += 1

        if applied > 0:
            summary = calculate_handicaps_for_entries(summary)

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
def _(
    ENTRIES_PATH,
    comparison_table,
    export_handicaps_button,
    mo,
    prediction_year,
):
    export_status_md = mo.md("")

    if bool(export_handicaps_button.value):
        if comparison_table is None or comparison_table.empty:
            export_status_md = mo.md("No prediction summary available to export.")
        else:
            export_cols = [
                "Name",
                "Predicted_Time",
                "Predicted_Time_seconds",
                "Handicap",
                "Off_time",
                "Num_results_used",
                "Num_excluded_results",
            ]
            export_df = comparison_table[[c for c in export_cols if c in comparison_table.columns]].copy()
            export_path = ENTRIES_PATH / f"handicaps_{int(prediction_year)}.csv"
            export_df.to_csv(export_path, index=False)
            export_status_md = mo.md(f"Exported handicap summary to {export_path}.")
    return (export_status_md,)


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

        # Always create a selector for racer predictions
        comparison_selector = mo.ui.table(
            predictions_view,
            selection="single",
            initial_selection=[0],
        )

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
                views.extend([metrics_md, mo.md("### Select Racer")])
                views.append(comparison_selector)
            else:
                views.append(
                    mo.md(
                        "### Actual vs Predicted\n"
                        "No racers could be matched between predictions and Hallam Chase results for this year.\n"
                        "### Select Racer"
                    )
                )
                views.append(comparison_selector)
        else:
            views.append(
                mo.md("### Select Racer\nNo Hallam Chase results for this year. Viewing prediction plots only.")
            )
            views.append(comparison_selector)
    return comparison_selector, views


@app.cell
def _(mo, views):
    mo.vstack(views)
    return


@app.cell
def _(export_status_md):
    export_status_md
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
    comparison_selector,
    con,
    convert_Chase_ZScore_logs_avg,
    entry_pr_time_map,
    entry_url_map,
    has_chase_results,
    load_models,
    load_prediction_time_overrides,
    make_chase_prediction,
    mo,
    pd,
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
    plot_info = mo.md("Select a racer above to view that racer's prediction distributions.")
    figure = None
    racer_detail_table = None
    racer_id = None

    if comparison_selector is not None and selected_row is not None:
        racer_id, canonical_name = secure_racer_id(con, selected_row["Name"].lower().strip())
        display_name = canonical_name if canonical_name is not None else str(selected_row["Name"])
        coeffs, covar, resid_stds = load_models(include_residuals=True)
        selected_name_key = selected_row["Name"].lower().strip()
        po10_url = entry_url_map.get(selected_name_key)
        given_pr_time = entry_pr_time_map.get(selected_name_key)

        process_kwargs = {
            "con": con,
            "coeffs": coeffs,
            "covar": covar,
            "resid_stds": resid_stds,
            "prediction_year": prediction_year,
            "po10_url": po10_url,
            "given_pr_time": given_pr_time,
        }
        if racer_id is not None:
            process_kwargs["racer_id"] = racer_id
        else:
            process_kwargs["racer_name"] = str(selected_row["Name"])

        racer_results, _ = process_results_for_racer(**process_kwargs)

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
                    racer_name=display_name,
                    prediction_year=prediction_year,
                )

                # If a manual predicted-time override exists, add it as a separate blue line.
                _overrides = load_prediction_time_overrides()
                _override_key = f"{selected_row['Name'].lower().strip()}|{int(prediction_year)}"
                if _override_key in _overrides:
                    override_t = float(_overrides[_override_key])
                    ax.axvline(
                        override_t,
                        color="tab:blue",
                        linestyle="--",
                        linewidth=2,
                        label="Manual override",
                    )

                # Only add actual time line if we have chase results and a valid actual time
                if (
                    has_chase_results
                    and selected_row is not None
                    and pd.notna(selected_row.get("Actual_Time_seconds"))
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
                    if pd.notna(selected_row.get("Actual_Time_seconds")):
                        actual_str = seconds_to_time_string(
                            selected_row["Actual_Time_seconds"]
                        )
                        err = selected_row.get("Prediction_Error_seconds")
                        plot_info = mo.md(
                            f"Predicted: {pred_str} | Actual: {actual_str} | Error: {int(err):+d}s"
                        )
                    else:
                        plot_info = mo.md(f"Predicted: {pred_str} | Actual: N/A")
        else:
            _has_url = bool(isinstance(po10_url, str) and po10_url.strip())
            _has_pr = bool(isinstance(given_pr_time, str) and ":" in given_pr_time)
            plot_info = mo.md(
                f"No plottable race evidence found for {display_name}. "
                f"P10 URL present: {'yes' if _has_url else 'no'} | Given PR present: {'yes' if _has_pr else 'no'}."
            )
    return figure, plot_info, racer_detail_table, racer_id


@app.cell
def _(load_prediction_time_overrides, mo, prediction_year, selected_row):
    ovr_name = None
    ovr_key = None
    ovr_time_input = None
    ovr_set_btn = None
    ovr_clear_btn = None

    if selected_row is not None:
        ovr_name = str(selected_row["Name"])
        _selected_name_key = ovr_name.lower().strip()
        ovr_key = f"{_selected_name_key}|{int(prediction_year)}"
        _overrides = load_prediction_time_overrides()
        _existing_seconds = _overrides.get(ovr_key)

        if _existing_seconds is not None:
            _total = int(round(float(_existing_seconds)))
            _default_mmss = f"{_total // 60:02d}:{_total % 60:02d}"
        else:
            _default_mmss = ""

        ovr_time_input = mo.ui.text(
            value=_default_mmss,
            label="Manual predicted time (MM:SS)",
            placeholder="e.g. 27:45",
        )
        ovr_set_btn = mo.ui.run_button(label="Set override")
        ovr_clear_btn = mo.ui.run_button(label="Reinstate model")
    return ovr_clear_btn, ovr_key, ovr_name, ovr_set_btn, ovr_time_input


@app.cell
def _(
    clear_prediction_time_override,
    load_prediction_time_overrides,
    mo,
    ovr_clear_btn,
    ovr_key,
    ovr_name,
    ovr_set_btn,
    ovr_time_input,
    prediction_year,
    racer_refresh_count,
    re,
    set_prediction_time_override,
    set_racer_refresh_count,
):
    _status_text = ""

    if ovr_name is not None and ovr_key is not None:
        _wrote_override = False

        if ovr_set_btn is not None and bool(ovr_set_btn.value):
            _raw = (ovr_time_input.value or "").strip() if ovr_time_input is not None else ""
            if not re.match(r"^\d{1,3}:[0-5]\d$", _raw):
                _status_text = "Invalid format. Please use MM:SS, e.g. 27:45."
            else:
                _minutes, _input_seconds = _raw.split(":")
                _total_seconds = int(_minutes) * 60 + int(_input_seconds)
                set_prediction_time_override(ovr_name, prediction_year, _total_seconds)
                _wrote_override = True
                _status_text = f"Manual override set for {ovr_name}: {_raw}."

        if ovr_clear_btn is not None and bool(ovr_clear_btn.value):
            clear_prediction_time_override(ovr_name, prediction_year)
            _wrote_override = True
            _status_text = f"Manual override cleared for {ovr_name}. Model prediction reinstated."

        if _wrote_override:
            set_racer_refresh_count(racer_refresh_count() + 1)

        _overrides = load_prediction_time_overrides()
        if _status_text == "":
            if ovr_key in _overrides:
                _total = int(round(float(_overrides[ovr_key])))
                _status_text = f"Current override for {ovr_name}: {_total // 60:02d}:{_total % 60:02d}."
            else:
                _status_text = f"No manual override set for {ovr_name}."

    ovr_status_md = mo.md(_status_text)
    return (ovr_status_md,)


@app.cell
def _(
    mo,
    ovr_clear_btn,
    ovr_set_btn,
    ovr_status_md,
    ovr_time_input,
    selected_row,
):
    override_panel = None
    if selected_row is not None:
        override_panel = mo.vstack(
            [
                mo.md("### Manual Predicted Time Override (MM:SS)"),
                mo.hstack([ovr_time_input, ovr_set_btn, ovr_clear_btn], widths=[2, 1, 1]),
                ovr_status_md,
            ]
        )
    return (override_panel,)


@app.cell
def _(
    bulk_cutoff_panel,
    figure,
    include_checkboxes,
    mo,
    override_panel,
    plot_info,
    racer_detail_display_table,
    racer_detail_table,
):
    outputs = [plot_info]
    if figure is not None:
        outputs.append(figure)

    if override_panel is not None:
        outputs.append(override_panel)

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
                bulk_cutoff_panel,
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
                    value=bool(detail_row["include"]),
                    label=f"{detail_row['Race_Name']} ({int(detail_row['Season'])})",
                )
                for _, detail_row in racer_detail_table.iterrows()
            ],
            label="Manual include / exclude",
        )
    else:
        include_checkboxes = None
        racer_detail_display_table = None
    return include_checkboxes, racer_detail_display_table


@app.cell
def _(mo, racer_detail_table, selected_row):
    bulk_cutoff_button_ui = None
    bulk_cutoff_input_ui = None

    if selected_row is not None and racer_detail_table is not None and not racer_detail_table.empty:
        bulk_cutoff_input_ui = mo.ui.text(
            value="",
            label="Remove races before (exclusive year)",
            placeholder="e.g. 2023",
        )
        bulk_cutoff_button_ui = mo.ui.run_button(
            label="Remove races before (exclusive)"
        )
    return bulk_cutoff_button_ui, bulk_cutoff_input_ui


@app.cell
def _(
    bulk_cutoff_button_ui,
    bulk_cutoff_input_ui,
    mo,
    racer_detail_table,
    racer_id,
    racer_refresh_count,
    re,
    set_race_override,
    set_racer_refresh_count,
):
    bulk_cutoff_status_md = mo.md("")

    if (
        bulk_cutoff_button_ui is not None
        and bulk_cutoff_input_ui is not None
        and racer_detail_table is not None
        and racer_id is not None
        and bool(bulk_cutoff_button_ui.value)
    ):
        raw_cutoff = (bulk_cutoff_input_ui.value or "").strip()
        if not re.fullmatch(r"\d{4}", raw_cutoff):
            bulk_cutoff_status_md = mo.md("Enter a four-digit year, for example 2023.")
        else:
            cutoff_year = int(raw_cutoff)
            matching_rows = racer_detail_table[racer_detail_table["Season"] < cutoff_year]
            for _, matching_row in matching_rows.iterrows():
                set_race_override(
                    racer_id,
                    int(matching_row["Season"]),
                    matching_row["Race_Name"],
                    False,
                )
            set_racer_refresh_count(racer_refresh_count() + 1)
            bulk_cutoff_status_md = mo.md(
                f"Excluded {len(matching_rows)} races before {cutoff_year} for this racer."
            )
    return (bulk_cutoff_status_md,)


@app.cell
def _(
    bulk_cutoff_button_ui,
    bulk_cutoff_input_ui,
    bulk_cutoff_status_md,
    mo,
    racer_detail_table,
    selected_row,
):
    bulk_cutoff_panel = None

    if (
        selected_row is not None
        and racer_detail_table is not None
        and not racer_detail_table.empty
        and bulk_cutoff_input_ui is not None
        and bulk_cutoff_button_ui is not None
    ):
        bulk_cutoff_panel = mo.vstack(
            [
                mo.md(
                    "Bulk action: enter a cutoff year to exclude all races before that year. "
                    "For example, 2023 removes seasons up to and including 2022."
                ),
                mo.hstack([bulk_cutoff_input_ui, bulk_cutoff_button_ui], widths=[2, 1]),
                bulk_cutoff_status_md,
            ]
        )
    return (bulk_cutoff_panel,)


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
