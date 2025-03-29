import re

def find_multiple_commas(text):
    pattern = r',{2,}'
    matches = re.findall(pattern, text)
    return matches

def find_lines_with_single_comma(text):
    pattern = r'^,$\n'
    matches = re.findall(pattern, text, re.MULTILINE)
    return matches

# Example usage
if __name__ == "__main__":
    text = "a,,b,,,c,,,,d\n,\n,,\n,,,\n,"
    print(find_multiple_commas(text))  # Output: [',,', ',,,', ',,,,']
    print(find_lines_with_single_comma(text))  # Output: [',\n']
