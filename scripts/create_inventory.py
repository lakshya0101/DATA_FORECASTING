import os
import csv

BASE = "data/pdfs"

with open(
    "metadata/pdf_master_inventory.csv",
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "year",
        "file_name",
        "size_kb"
    ])

    for year in sorted(os.listdir(BASE)):

        year_path = os.path.join(BASE, year)

        if not os.path.isdir(year_path):
            continue

        for file in sorted(os.listdir(year_path)):

            path = os.path.join(year_path, file)

            size_kb = round(
                os.path.getsize(path) / 1024,
                2
            )

            writer.writerow([
                year,
                file,
                size_kb
            ])

print("Inventory created.")