#!/usr/bin/env python
"""
Standalone script to extract race results from Chip Timing website and save to CSV.
The site stores participant data in JSON format embedded in the page.
Usage: python extract_results_from_shef_half.py <url> [output_filename]
"""

import sys
import json
import re
from pathlib import Path
import pandas as pd
import requests


def extract_results_to_csv(url: str, output_filename: str = None):
    """Extract results from Chip Timing URL and save to CSV."""
    print('Getting data from URL')
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    response = requests.get(url, headers=headers)
    
    # Find all JSON-like arrays in the page
    # Look for [...] patterns containing objects with common runner fields
    json_arrays = re.findall(r'\[\s*\{[^}]*?"(bib_number|forename|surname|gun_time|chip_time)"[^}]*?\}[^\]]*?\]', response.text)
    
    if not json_arrays:
        print("Could not find runner data in the page")
        return
    
    print(f"Found potential data arrays")
    
    # Try to find and extract the actual JSON
    # Search for the full array
    match = re.search(r'(\[\s*\{\s*"participant_id"[^\]]*\]\s*)', response.text)
    
    if not match:
        print("Could not extract full JSON array")
        return
    
    json_str = match.group(1)
    
    # Clean up the JSON string - handle any escaping issues
    try:
        participants = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        print(f"Trying alternative parsing...")
        # Try to repair common JSON issues
        try:
            participants = json.loads(json_str.replace("'", '"'))
        except:
            print(f"Could not parse the data")
            return
    
    if not participants:
        print("No data extracted from JSON")
        return
    
    # Convert to DataFrame
    data = pd.DataFrame(participants)
    
    print(f"Extracted {len(data)} participants")
    print(f"Columns: {list(data.columns)[:10]}...")  # Show first 10 columns
    
    # Determine output filename
    if output_filename is None:
        from datetime import datetime
        output_filename = f"extracted_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    # Ensure it has .csv extension
    if not output_filename.endswith('.csv'):
        output_filename += '.csv'
    
    output_path = Path('./csv') / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data.to_csv(output_path, index=False)
    print(f"Results extracted and saved to {output_path}")
    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_results_from_shef_half.py <url> [output_filename]")
        sys.exit(1)
    
    url = sys.argv[1]
    output_filename = sys.argv[2] if len(sys.argv) > 2 else None
    
    extract_results_to_csv(url, output_filename)
