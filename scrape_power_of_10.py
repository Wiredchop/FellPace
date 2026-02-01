"""Scrape athlete performance data from Power of 10 / MyAthletics UK"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from typing import List, Dict
import re
import json
import sys


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


def scrape_athlete_performances(athlete_url: str) -> Dict:
    """
    Scrape athlete performance data, focusing on 5km and 10km times
    
    Args:
        athlete_url: URL of the athlete's profile page
        
    Returns:
        Dictionary containing athlete info and race results
    """
    # Fetch the page
    response = requests.get(athlete_url)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Extract athlete name from the page
    athlete_name = "Unknown"
    
    # Look for the athlete name in script or text
    scripts = soup.find_all('script')
    for script in scripts:
        script_text = script.string
        if script_text and 'athleteRef' in script_text:
            # Try to extract name from nearby elements
            pass
    
    # Try to find name in h2 or strong tags
    for tag in soup.find_all(['h2', 'h3', 'strong']):
        text = tag.get_text(strip=True)
        if text and len(text) > 3 and not any(x in text.lower() for x in ['club', 'lead', 'coach', 'event', 'performance']):
            athlete_name = text
            break
    
    # Extract performances from embedded JavaScript data
    performances = []
    
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
            
            # Determine if this is 5km, 10km, or other distance
            distance = None
            if 'parkrun' in event_name.lower():
                distance = '5km'  # parkrun is always 5km
            elif '5k' in event_name.lower() or '5000' in event_name.lower() or '5 k' in event_name.lower():
                distance = '5km'
            elif '10k' in event_name.lower() or '10000' in event_name.lower() or '10 k' in event_name.lower():
                distance = '10km'
            elif 'half' in event_name.lower() or '21k' in event_name.lower() or '21.1' in event_name.lower():
                distance = 'Half Marathon'
            elif 'marathon' in event_name.lower() and 'half' not in event_name.lower():
                distance = 'Marathon'
            elif '15k' in event_name.lower() or '15000' in event_name.lower():
                distance = '15km'
            elif '20k' in event_name.lower() or '20000' in event_name.lower():
                distance = '20km'
            elif 'mile' in event_name.lower() and '10' not in event_name.lower():
                distance = 'Mile'
            elif '10 mile' in event_name.lower() or '10mile' in event_name.lower():
                distance = '10 Mile'
            
            # Only process 5km and 10km events
            if distance not in ['5km', '10km']:
                continue
            
            # Try to extract all performances data (dataRpValues = all results)
            rp_values_pattern = f"var dataRpValues{idx} = \\[([\\d,\\s]+)\\]"
            rp_values_match = re.search(rp_values_pattern, script_text)
            
            if not rp_values_match:
                continue
                
            # Parse the values
            values_str = rp_values_match.group(1)
            values = [int(v.strip()) for v in values_str.split(',') if v.strip()]
            
            # Parse locations
            locations = []
            rp_locations_pattern = f"var dataRpLocations{idx} = \\[(.*?)\\];"
            rp_locations_match = re.search(rp_locations_pattern, script_text, re.DOTALL)
            if rp_locations_match:
                loc_str = rp_locations_match.group(1)
                locations = [l.strip().strip("'\"") for l in re.findall(r"'([^']*)'", loc_str)]
            
            # Parse dates
            dates = []
            rp_dates_pattern = f"var dataRpMeetDates{idx} = \\[(.*?)\\];"
            rp_dates_match = re.search(rp_dates_pattern, script_text, re.DOTALL)
            if rp_dates_match:
                date_str = rp_dates_match.group(1)
                dates = [d.strip().strip("'\"") for d in re.findall(r"'([^']*)'", date_str)]
            
            # Parse meetings
            meetings = []
            rp_meetings_pattern = f"var dataRpMeetings{idx} = \\[(.*?)\\];"
            rp_meetings_match = re.search(rp_meetings_pattern, script_text, re.DOTALL)
            if rp_meetings_match:
                meet_str = rp_meetings_match.group(1)
                meetings = [m.strip().strip("'\"") for m in re.findall(r"'([^']*)'", meet_str)]
            
            # Parse positions
            positions = []
            rp_positions_pattern = f"var dataRpPositions{idx} = \\[(.*?)\\];"
            rp_positions_match = re.search(rp_positions_pattern, script_text, re.DOTALL)
            if rp_positions_match:
                pos_str = rp_positions_match.group(1)
                positions = [p.strip().strip("'\"") for p in re.findall(r"'([^']*)'", pos_str)]
            
            # Combine all data
            for i in range(len(values)):
                if values[i]:  # Skip empty values
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
    
    # Also check tables for additional data
    tables = soup.find_all('table')
    for table in tables:
        rows = table.find_all('tr')
        
        for row in rows:
            cols = row.find_all('td')
            if len(cols) >= 2:
                col_texts = [col.get_text(strip=True) for col in cols]
                
                # Check if this is a results table with Event and Time columns
                if cols[0].get_text(strip=True).lower() in ['5k', '10k', '5km', '10km', '5000m', '10000m']:
                    distance = '5km' if '5' in cols[0].get_text(strip=True) else '10km'
                    
                    # Extract time
                    time_pattern = re.compile(r'\d{1,2}:\d{2}(?::\d{2})?')
                    for text in col_texts[1:]:
                        time_match = time_pattern.search(text)
                        if time_match:
                            performances.append({
                                'event': 'Road Race',
                                'distance': distance,
                                'time': time_match.group(),
                                'venue': '',
                                'date': '',
                                'race_name': '',
                                'position': ''
                            })
                            break
    
    return {
        'athlete_name': athlete_name,
        'url': athlete_url,
        'performances': performances,
        'total_results': len(performances)
    }


def main():
    # Get URL from command line argument or use default
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        url = "https://earlyaccess.myathletics.uk/Home/Athlete/1ee315ac-0a39-4d4a-82a8-beabecaf8cd9"
    
    print(f"Scraping data from: {url}\n")
    
    try:
        data = scrape_athlete_performances(url)
        
        print(f"Athlete: {data['athlete_name']}")
        print(f"Total 5km/10km results found: {data['total_results']}\n")
        
        if data['performances']:
            print("5km and 10km Results:")
            print("=" * 100)
            
            for i, perf in enumerate(data['performances'], 1):
                print(f"\n{i}. {perf['event']} ({perf['distance']})")
                print(f"   Time: {perf['time']}")
                if perf.get('date'):
                    print(f"   Date: {perf['date']}")
                if perf.get('venue'):
                    print(f"   Venue: {perf['venue']}")
                if perf.get('race_name'):
                    print(f"   Race: {perf['race_name']}")
                if perf.get('position'):
                    print(f"   Position: {perf['position']}")
            
            print("\n" + "=" * 100)
            
            # Convert to DataFrame for easy viewing
            df = pd.DataFrame(data['performances'])
            
            # Sort by time (fastest first)
            if 'time_seconds' in df.columns:
                df = df.sort_values('time_seconds')
            
            print("\nBest Times Summary:")
            print(df[['distance', 'time', 'date', 'venue']].to_string(index=False))
            
            # Save to CSV
            output_file = 'athlete_5k_10k_results.csv'
            df.to_csv(output_file, index=False)
            print(f"\n✓ Results saved to: {output_file}")
            
            # Print personal bests
            print("\n" + "=" * 100)
            print("PERSONAL BESTS:")
            for distance in ['5km', '10km']:
                distance_data = df[df['distance'] == distance]
                if not distance_data.empty:
                    best = distance_data.iloc[0]
                    print(f"{distance}: {best['time']} ({best['date']} at {best['venue']})")
                else:
                    print(f"{distance}: No results found")
            
        else:
            print("No 5km or 10km results found.")
            print("\nThe page may use dynamic JavaScript loading.")
            print("You might need to use Selenium or check if the data loads via API.")
            
    except Exception as e:
        print(f"Error scraping data: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
