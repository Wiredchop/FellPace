"""
Power of 10 / MyAthletics Athlete Scraper with Search

NOTE: The MyAthletics search requires reCAPTCHA which cannot be automated easily.
For now, this script accepts direct athlete URLs or athlete IDs.

To find an athlete:
1. Go to: https://earlyaccess.myathletics.uk/Home/AthleteSearch
2. Search manually for the athlete
3. Copy their URL or ID from the results
4. Use this script with that URL/ID
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from typing import List, Dict
import re
import sys


def extract_athlete_id_from_url(url: str) -> str:
    """Extract athlete ID (UUID) from MyAthletics URL"""
    match = re.search(r'/Athlete/([a-f0-9\-]+)', url)
    if match:
        return match.group(1)
    return url  # Assume it's already an ID


def convert_centiseconds_to_time(centiseconds: int) -> str:
    """Convert centiseconds to MM:SS format"""
    if not centiseconds:
        return None
    
    total_seconds = centiseconds / 100
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    
    if hours > 0:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes}:{seconds:02d}"


def scrape_athlete_performances(athlete_url_or_id: str) -> Dict:
    """
    Scrape athlete performance data, focusing on 5km and 10km times
    
    Args:
        athlete_url_or_id: Full URL or just the athlete ID (UUID)
        
    Returns:
        Dictionary containing athlete info and race results
    """
    # Extract ID and construct full URL
    athlete_id = extract_athlete_id_from_url(athlete_url_or_id)
    athlete_url = f"https://earlyaccess.myathletics.uk/Home/Athlete/{athlete_id}"
    
    # Fetch the page
    response = requests.get(athlete_url)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Extract athlete name
    athlete_name = "Unknown"
    for tag in soup.find_all(['h2', 'h3', 'strong']):
        text = tag.get_text(strip=True)
        if text and len(text) > 3 and not any(x in text.lower() for x in ['club', 'lead', 'coach', 'event', 'performance']):
            athlete_name = text
            break
    
    # Extract performances from embedded JavaScript data
    performances = []
    scripts = soup.find_all('script')
    
    # Parse JavaScript variables in script tags
    for script in scripts:
        if not script.string:
            continue
            
        script_text = script.string
        
        # Find all event indices
        event_indices = re.findall(r"var dataEventName(\d+) = ", script_text)
        
        for idx in event_indices:
            # Extract event name
            event_name_pattern = f"var dataEventName{idx} = '([^']+)'"
            event_name_match = re.search(event_name_pattern, script_text)
            
            if not event_name_match:
                continue
                
            event_name = event_name_match.group(1)
            
            # Determine if this is 5km or 10km
            distance = None
            if 'parkrun' in event_name.lower():
                distance = '5km'
            elif '5k' in event_name.lower() or '5000' in event_name.lower() or '5 k' in event_name.lower():
                distance = '5km'
            elif '10k' in event_name.lower() or '10000' in event_name.lower() or '10 k' in event_name.lower():
                distance = '10km'
            
            # Only process 5km and 10km events
            if distance not in ['5km', '10km']:
                continue
            
            # Try to extract all performances data
            rp_values_pattern = f"var dataRpValues{idx} = \\[([\\d,\\s]+)\\]"
            rp_values_match = re.search(rp_values_pattern, script_text)
            
            if not rp_values_match:
                continue
                
            # Parse the values
            values_str = rp_values_match.group(1)
            values = [int(v.strip()) for v in values_str.split(',') if v.strip()]
            
            # Parse locations, dates, meetings, positions
            locations = []
            rp_locations_pattern = f"var dataRpLocations{idx} = \\[(.*?)\\];"
            rp_locations_match = re.search(rp_locations_pattern, script_text, re.DOTALL)
            if rp_locations_match:
                loc_str = rp_locations_match.group(1)
                locations = [l.strip().strip("'\"") for l in re.findall(r"'([^']*)'", loc_str)]
            
            dates = []
            rp_dates_pattern = f"var dataRpMeetDates{idx} = \\[(.*?)\\];"
            rp_dates_match = re.search(rp_dates_pattern, script_text, re.DOTALL)
            if rp_dates_match:
                date_str = rp_dates_match.group(1)
                dates = [d.strip().strip("'\"") for d in re.findall(r"'([^']*)'", date_str)]
            
            meetings = []
            rp_meetings_pattern = f"var dataRpMeetings{idx} = \\[(.*?)\\];"
            rp_meetings_match = re.search(rp_meetings_pattern, script_text, re.DOTALL)
            if rp_meetings_match:
                meet_str = rp_meetings_match.group(1)
                meetings = [m.strip().strip("'\"") for m in re.findall(r"'([^']*)'", meet_str)]
            
            positions = []
            rp_positions_pattern = f"var dataRpPositions{idx} = \\[(.*?)\\];"
            rp_positions_match = re.search(rp_positions_pattern, script_text, re.DOTALL)
            if rp_positions_match:
                pos_str = rp_positions_match.group(1)
                positions = [p.strip().strip("'\"") for p in re.findall(r"'([^']*)'", pos_str)]
            
            # Combine all data
            for i in range(len(values)):
                if values[i]:
                    perf = {
                        'event': event_name,
                        'distance': distance,
                        'time': convert_centiseconds_to_time(values[i]),
                        'time_seconds': values[i] / 100,
                        'venue': locations[i] if i < len(locations) else '',
                        'date': dates[i] if i < len(dates) else '',
                        'race_name': meetings[i] if i < len(meetings) else '',
                        'position': positions[i] if i < len(positions) else ''
                    }
                    performances.append(perf)
    
    return {
        'athlete_name': athlete_name,
        'athlete_id': athlete_id,
        'url': athlete_url,
        'performances': performances,
        'total_results': len(performances)
    }


def main():
    print(__doc__)
    
    # Example athlete IDs for testing
    test_athletes = {
        'John Kelley (example 1)': '1ee315ac-0a39-4d4a-82a8-beabecaf8cd9',  # Simon Choppin 
        'Ben Heller (example 2)': '3d5c4b1a-d532-4269-80e6-ebb58a5f00e7',  # V60 athlete
    }
    
    # Get URL/ID from command line or use example
    if len(sys.argv) > 1:
        athlete_input = sys.argv[1]
        print(f"\nScraping athlete: {athlete_input}\n")
    else:
        print("\nNo athlete URL/ID provided.")
        print("Example usage:")
        print('  python athlete_scraper_with_search.py "https://earlyaccess.myathletics.uk/Home/Athlete/[ID]"')
        print('  python athlete_scraper_with_search.py "[ATHLETE-ID]"')
        print("\nTo search for an athlete:")
        print("  1. Visit: https://earlyaccess.myathletics.uk/Home/AthleteSearch")
        print("  2. Search manually (e.g., 'John Kelley' or 'Ben Heller')")
        print("  3. Click on the athlete you want")
        print("  4. Copy the URL or ID from the address bar")
        print("  5. Run this script with that URL/ID")
        return
    
    try:
        data = scrape_athlete_performances(athlete_input)
        
        print(f"Athlete: {data['athlete_name']}")
        print(f"ID: {data['athlete_id']}")
        print(f"Total 5km/10km results found: {data['total_results']}\n")
        
        if data['performances']:
            df = pd.DataFrame(data['performances'])
            
            # Sort by time
            if 'time_seconds' in df.columns:
                df = df.sort_values('time_seconds')
            
            # Save to CSV
            output_file = f"athlete_{data['athlete_id']}_results.csv"
            df.to_csv(output_file, index=False)
            print(f"✓ Results saved to: {output_file}\n")
            
            # Print personal bests
            print("PERSONAL BESTS:")
            for distance in ['5km', '10km']:
                distance_data = df[df['distance'] == distance]
                if not distance_data.empty:
                    best = distance_data.iloc[0]
                    print(f"  {distance}: {best['time']} ({best['date']} at {best['venue']})")
                else:
                    print(f"  {distance}: No results found")
        else:
            print("No 5km or 10km results found.")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
