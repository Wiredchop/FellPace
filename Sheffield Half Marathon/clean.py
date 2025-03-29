from pathlib import Path
import csv
import re



def should_ignore_line(line):
    """Check if the line matches the 'X of Y' pattern"""
    pattern = r'Timing & Results Service by HS Sports Ltd Tel: 01260 275708'
    return bool(re.search(pattern, line))

def process_csv(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        ignore_count = 0

        for row in reader:
            if ignore_count > 0:
                ignore_count -= 1
                continue
            
            if should_ignore_line(', '.join(row)):
                print(row)
                ignore_count = 3  # Ignore the next 6 lines
                continue
            
            writer.writerow(row)

# Use the function with your CSV files
file = Path("C:\\Users\wired\\Downloads\\Sheffield Half Marathon\\Results-Sheffield-Half-Marathon-2018.csv")
output = Path("C:\\Users\\wired\Downloads\\Sheffield Half Marathon\\ShefHalf-2018.csv")
process_csv(file, output)
