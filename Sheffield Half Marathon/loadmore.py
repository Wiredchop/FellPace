from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
import csv

url = 'https://chiptiming.co.uk/events/asda-foundation-sheffield-half-marathon-2021/'
driver = webdriver.Firefox()
driver.get(url)

while True:
    try:
        # Wait for the load more button to be clickable
        load_more_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, 'a.load'))
        )
        load_more_button.click()
        time.sleep(0.1)  # Wait for new results to load
    except Exception as e:
        print("No more load more button found or an error occurred:", e)
        break

# After loading all data, save table content
table_html = driver.page_source
soup = BeautifulSoup(table_html, 'html.parser')
table = soup.find('table', class_='results')

# Write table to CSV
rows = table.find_all('tr')
with open('output.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    for row in rows:
        cols = row.find_all(['td', 'th'])
        writer.writerow([col.get_text(strip=True) for col in cols])

driver.quit()
