import fellpace.FellPace_tools as FellPace_tools
import sqlite3
import toml
from typing import Literal
import fellpace.convert_tools as convert_tools
from fellpace.parkrun.settings import PRSettings
from re import search
import time

import pandas as pd
import importlib
# Headers work for Parkrun
headers = {
"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0",
"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
"Accept-Language": "en-GB,en;q=0.5",
"Accept-Encoding": "gzip, deflate, br",
"Referer": "https://www.parkrun.org.uk/hillsborough/results/eventhistory/",
"Connection": "keep-alive",
"Cookie": "cookiesDisclosureCount=14",
"Upgrade-Insecure-Requests": "1",
"Sec-Fetch-Dest": "document",
"Sec-Fetch-Mode": "navigate",
"Sec-Fetch-Site": "same-origin",
"Sec-Fetch-User": "?1"}
parkrun: Literal['hillsborough','endcliffe'] = 'endcliffe'
parkrun_climbs = {'hillsborough':53,'endcliffe':47}


def get_rendered_results_table(url: str) -> tuple[pd.DataFrame, str]:
    """Load a Parkrun results page in a browser context and parse the rendered results table."""
    try:
        sync_api = importlib.import_module("playwright.sync_api")
        sync_playwright = getattr(sync_api, "sync_playwright")
        PlaywrightTimeoutError = getattr(sync_api, "TimeoutError")
    except Exception as import_error:
        raise RuntimeError(
            "Playwright is required for Parkrun scraping. Install with: pip install playwright; "
            "then run: playwright install chromium"
        ) from import_error

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=headers["User-Agent"],
            locale="en-GB",
        )
        page = context.new_page()

        try:
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            # Allow challenge/redirect scripts to run and page to settle.
            page.wait_for_timeout(5000)
            page.wait_for_load_state("networkidle", timeout=30000)
        except PlaywrightTimeoutError:
            raise RuntimeError("Timed out waiting for Parkrun results page to finish loading.")

        # Identify the results table by headers and parse directly from rendered DOM rows.
        table = None
        table_count = page.locator("table").count()
        for table_idx in range(table_count):
            table_locator = page.locator("table").nth(table_idx)
            headers_raw = table_locator.locator("thead th").all_text_contents()
            table_headers = [h.strip() for h in headers_raw]
            normalised_headers = {h.lower(): i for i, h in enumerate(table_headers)}

            # Current Parkrun table typically has these columns.
            required = {"position", "parkrunner", "age group", "time"}
            if not required.issubset(set(normalised_headers.keys())):
                continue

            club_idx = normalised_headers.get("club", normalised_headers.get("group"))
            if club_idx is None:
                continue

            pos_idx = normalised_headers["position"]
            runner_idx = normalised_headers["parkrunner"]
            age_idx = normalised_headers["age group"]
            time_idx = normalised_headers["time"]

            rows = []
            row_count = table_locator.locator("tbody tr").count()
            for row_idx in range(row_count):
                cells = table_locator.locator("tbody tr").nth(row_idx).locator("td").all_inner_texts()
                if not cells:
                    continue
                # Normalize whitespace from cell text.
                cells = [" ".join(c.split()) for c in cells]
                max_idx = max(pos_idx, runner_idx, age_idx, time_idx, club_idx)
                if len(cells) <= max_idx:
                    continue
                rows.append(
                    {
                        "Position": cells[pos_idx],
                        "parkrunner": cells[runner_idx],
                        "Age Group": cells[age_idx],
                        "Time": cells[time_idx],
                        "Club": cells[club_idx],
                    }
                )

            if rows:
                table = pd.DataFrame(rows)
                break

        content = page.content()
        browser.close()

    if "AwsWafIntegration" in content or "verify that you're not a robot" in content:
        raise RuntimeError("Browser-rendered page is still blocked by WAF challenge.")

    if table is None or table.empty:
        raise RuntimeError("Could not find a usable rendered Parkrun results table in page DOM.")

    return table, content


def validate_parkrun_entries(entries_df: pd.DataFrame, race_name: str) -> None:
    """Validate converted entry data and print a concise ingestion summary."""
    required_cols = {"Position", "Club", "Racer_Name", "Cat_Name", "Time"}
    missing = required_cols - set(entries_df.columns)
    if missing:
        raise RuntimeError(f"{race_name}: converted entries missing required columns: {sorted(missing)}")

    non_null_time = int(entries_df["Time"].notna().sum())
    non_empty_name = int(entries_df["Racer_Name"].astype(str).str.strip().ne("").sum())
    valid_rows = int((entries_df["Time"].notna() & entries_df["Racer_Name"].astype(str).str.strip().ne("")).sum())

    if non_null_time == 0:
        raise RuntimeError(f"{race_name}: no valid time values found after conversion.")
    if non_empty_name == 0:
        raise RuntimeError(f"{race_name}: no valid runner names found after conversion.")

    print(
        f"{race_name}: scraped rows={len(entries_df)}, with_time={non_null_time}, "
        f"with_name={non_empty_name}, valid_rows={valid_rows}"
    )
    print(entries_df[["Position", "Racer_Name", "Club", "Cat_Name", "Time"]].head(5))

def scrape_parkruns(settings: PRSettings, con: sqlite3.Connection):
    
    for parkrun, these_settings in settings.__dict__.items():
        continue_scrape = True
        PR_id = these_settings.start_ID
        while continue_scrape:
            URL = f'https://www.parkrun.org.uk/{parkrun}/results/{PR_id}/'
            try:
                table, resp_text = get_rendered_results_table(URL)
            except Exception as e:
                print(f"Exception occurred: {e}")
                continue_scrape = False
                these_settings.start_ID = PR_id
                print(f'Stopping scrape for {parkrun} at ID {PR_id}.')
                with open('settings.toml', 'w') as f:
                    f.write(toml.dumps(settings.model_dump()))
                continue
            #Going to use regular expressions to get date rather than beautiful soup as only need to do once
            matches = search("(?<=class=\"format-date\">)[0-9/]+",resp_text)
            if not matches:
                date = ""
            else:
                date = matches.group()
                (day,month,year) = date.split("/")
                date = "-".join((year,month,day))
            #Create the race metadata for the entry
            this_parkrun = FellPace_tools.race_meta()
            this_parkrun.race_distance = 5000
            this_parkrun.race_climb = parkrun_climbs[parkrun]
            this_parkrun.race_date = date
            this_parkrun.race_name = f"Parkrun_{parkrun}_{PR_id}" #Parkrun name has the ID appended to the back so can easily parse in future if want to update
            print(f'adding {this_parkrun.race_name}')
            ParkRun = convert_tools.ParkRunConverter(table)
            validate_parkrun_entries(ParkRun.entries.data, this_parkrun.race_name)
            FellPace_tools.append_to_DB(con,ParkRun.entries.data,this_parkrun,check=False)
            PR_id += 1
            time.sleep(2)

if __name__ == "__main__":
    from fellpace.config import DB_PATH
    from fellpace.db.db_setup import setup_db
    #Connect to the DBy
    con = setup_db(DB_PATH)
    settings = PRSettings.load_toml_settings('settings.toml')
    scrape_parkruns(settings,con)
    con.close()