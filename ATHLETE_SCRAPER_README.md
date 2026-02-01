# Power of 10 / MyAthletics Athlete Performance Scraper

## Overview
This scraper extracts **5km and 10km race times** from Power of 10 / MyAthletics athlete profiles, including parkrun results and road race times.

## Files Created
- `scrape_power_of_10.py` - Main scraper script (simple usage)
- `athlete_scraper_with_search.py` - Enhanced version with documentation

## Important Note: Search Limitation
**The MyAthletics website uses reCAPTCHA for athlete searches, which cannot be automated.**

### How to Find Athletes

1. **Manual Search (Required)**
   - Visit: https://earlyaccess.myathletics.uk/Home/AthleteSearch
   - Enter athlete name (e.g., "John Kelley" or "Ben Heller")
   - Click on the athlete from results
   - Copy the URL or ID from the address bar

2. **Example URLs**
   - Full URL: `https://earlyaccess.myathletics.uk/Home/Athlete/3d5c4b1a-d532-4269-80e6-ebb58a5f00e7`
   - Just the ID: `3d5c4b1a-d532-4269-80e6-ebb58a5f00e7`

## Usage

### Option 1: Using scrape_power_of_10.py
```bash
# With full URL
uv run python scrape_power_of_10.py "https://earlyaccess.myathletics.uk/Home/Athlete/3d5c4b1a-d532-4269-80e6-ebb58a5f00e7"

# With just the ID
uv run python scrape_power_of_10.py "3d5c4b1a-d532-4269-80e6-ebb58a5f00e7"
```

### Option 2: Using athlete_scraper_with_search.py (same usage)
```bash
uv run python athlete_scraper_with_search.py "3d5c4b1a-d532-4269-80e6-ebb58a5f00e7"
```

## Test Cases

### John Kelley
When searching manually on MyAthletics, "John Kelley" returns **3 different athletes**:
- You would need to look at each profile to determine which one you want
- Then copy that athlete's URL/ID

### Ben Heller  
When searching manually on MyAthletics, "Ben Heller" returns **1 athlete**:
- ID: `3d5c4b1a-d532-4269-80e6-ebb58a5f00e7`
- **Personal Bests:**
  - 5km: 19:49 (10/03/2012 at Endcliffe)
  - 10km: 41:16 (02/12/2012 at Sheffield)
- **Total Results:** 446 parkrun results + 9 dedicated 10km races = 455 results

## Output

The scraper extracts:
- **Event name** (e.g., "parkrun", "10K")
- **Distance** (5km or 10km)
- **Time** (MM:SS format)
- **Date** 
- **Venue**
- **Race name** (e.g., "Endcliffe parkrun # 660")
- **Position**

### Console Output
```
Athlete: V60V60(62 YRS)
ID: 3d5c4b1a-d532-4269-80e6-ebb58a5f00e7
Total 5km/10km results found: 455

✓ Results saved to: athlete_3d5c4b1a-d532-4269-80e6-ebb58a5f00e7_results.csv

PERSONAL BESTS:
  5km: 19:49 (10/03/2012 at Endcliffe)
  10km: 41:16 (02/12/2012 at Sheffield)
```

### CSV Output
Results are saved to `athlete_[ID]_results.csv` with columns:
- event
- distance
- time
- time_seconds
- venue
- date
- race_name
- position

## Features

✅ Scrapes all parkrun (5km) results  
✅ Scrapes all dedicated 10km race results  
✅ Extracts complete race details (date, venue, position)  
✅ Sorts results by fastest time  
✅ Calculates personal bests for each distance  
✅ Exports to CSV for further analysis  
✅ Handles hundreds of results efficiently  

## How It Works

The scraper:
1. Fetches the athlete profile page
2. Parses embedded JavaScript data (where race results are stored)
3. Identifies 5km and 10km events (including parkrun)
4. Extracts all race details from JavaScript arrays
5. Converts times from centiseconds to MM:SS format
6. Organizes and exports the data

## Limitations

- ⚠️ **Cannot automate athlete search** (reCAPTCHA protected)
- Only extracts 5km and 10km distances
- Requires manual URL/ID lookup for each athlete
- Depends on MyAthletics website structure

## Future Enhancement Ideas

1. Create a batch processor that reads athlete IDs from a file
2. Add support for other distances (Half Marathon, Marathon)
3. Create a simple GUI for entering athlete IDs
4. Add data visualization (time progression charts)
5. Compare multiple athletes side-by-side

## Dependencies

- requests
- beautifulsoup4
- pandas
- re (built-in)
- sys (built-in)
