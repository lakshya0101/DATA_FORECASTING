# import requests

# url = "https://idsp.mohfw.gov.in/index4.php?lang=1&level=0&linkid=406&lid=3689"

# response = requests.get(url)

# print("Status Code:", response.status_code)

# with open("page_source.html", "w", encoding="utf-8") as f:
#     f.write(response.text)

# print("HTML saved successfully")

import re
import csv
import requests

URL = "https://idsp.mohfw.gov.in/index4.php?lang=1&level=0&linkid=406&lid=3689"

response = requests.get(URL)
html = response.text

target_years = ["2023", "2024", "2025", "2026"]

rows = []

for year in target_years:

    # Locate year block
    year_start = html.find(f"<strong>{year}</strong>")

    if year_start == -1:
        print(f"{year} not found")
        continue

    # Find next year's block
    next_positions = []

    for y in ["2022", "2023", "2024", "2025", "2026"]:
        pos = html.find(f"<strong>{y}</strong>", year_start + 1)

        if pos != -1:
            next_positions.append(pos)

    year_end = min(next_positions) if next_positions else len(html)

    year_html = html[year_start:year_end]

    links = re.findall(
        r'<a[^>]*href="([^"]+)"[^>]*>([^<]+)</a>',
        year_html,
        re.IGNORECASE
    )

    for link, week in links:

        week = week.strip()

        rows.append([
            year,
            week,
            link
        ])

with open(
    "metadata/pdf_links.csv",
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "year",
        "week",
        "url"
    ])

    writer.writerows(rows)

print(f"Saved {len(rows)} records")