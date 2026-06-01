# scripts/test_fetch.py

import requests

url = "https://idsp.mohfw.gov.in/index4.php?lang=1&level=0&linkid=406&lid=3689"

response = requests.get(url)

print("Status:", response.status_code)
print("Length:", len(response.text))