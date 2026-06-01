# import csv
# import os
# import requests
# from urllib.parse import urlparse
# import urllib3
# urllib3.disable_warnings()

# CSV_FILE = "metadata/pdf_links.csv"
# BASE_FOLDER = "data/pdfs"

# os.makedirs(BASE_FOLDER, exist_ok=True)

# downloaded = 0
# failed = 0

# with open(CSV_FILE, "r", encoding="utf-8") as file:
#     reader = csv.DictReader(file)

#     for row in reader:

#         year = row["year"]
#         week = row["week"]
#         url = row["url"]

#         year_folder = os.path.join(BASE_FOLDER, year)
#         os.makedirs(year_folder, exist_ok=True)

#         week_number = ''.join(filter(str.isdigit, week)).zfill(2)

#         filename = f"week_{week_number}.pdf"
#         filepath = os.path.join(year_folder, filename)

#         try:

#             # response = requests.get(url, timeout=30)
#             response = requests.get(
#                 url,
#                 timeout=30,
#                 verify=False
#             )

#             if response.status_code == 200:

#                 with open(filepath, "wb") as pdf:
#                     pdf.write(response.content)

#                 downloaded += 1
#                 print(f"[OK] {year} Week {week}")

#             else:

#                 failed += 1
#                 print(f"[FAILED] {year} Week {week}")

#         except Exception as e:

#             failed += 1
#             print(f"[ERROR] {year} Week {week} -> {e}")

# print("\n===================")
# print(f"Downloaded : {downloaded}")
# print(f"Failed     : {failed}")
# print("===================")

import csv
import os
import requests
import urllib3

# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

CSV_FILE = "metadata/pdf_links.csv"
BASE_FOLDER = "data/pdfs"
LOG_FILE = "logs/download_log.csv"

os.makedirs(BASE_FOLDER, exist_ok=True)
os.makedirs("logs", exist_ok=True)

downloaded = 0
failed = 0
skipped = 0

# Create log file
with open(LOG_FILE, "w", newline="", encoding="utf-8") as log:
    log_writer = csv.writer(log)
    log_writer.writerow(
        ["year", "week", "status", "file_name", "url"]
    )

    with open(CSV_FILE, "r", encoding="utf-8") as file:

        reader = csv.DictReader(file)

        for row in reader:

            year = row["year"].strip()
            week = row["week"].strip()
            url = row["url"].strip()

            # Create year folder
            year_folder = os.path.join(BASE_FOLDER, year)
            os.makedirs(year_folder, exist_ok=True)

            # Extract week number
            week_number = "".join(
                filter(str.isdigit, week)
            ).zfill(2)

            filename = f"week_{week_number}.pdf"
            filepath = os.path.join(
                year_folder,
                filename
            )

            # Skip already downloaded PDFs
            if os.path.exists(filepath):

                file_size = os.path.getsize(filepath)

                if file_size > 1000:
                    skipped += 1

                    print(
                        f"[SKIP] {year} Week {week}"
                    )

                    log_writer.writerow([
                        year,
                        week,
                        "already_exists",
                        filename,
                        url
                    ])

                    continue

            try:

                # --------------------------------------------------
                # GOOGLE DRIVE HANDLING
                # --------------------------------------------------

                if "drive.google.com" in url:

                    try:
                        file_id = (
                            url.split("/d/")[1]
                            .split("/")[0]
                        )

                        url = (
                            "https://drive.google.com/"
                            f"uc?export=download&id={file_id}"
                        )

                    except Exception:
                        raise Exception(
                            "Could not extract Google Drive File ID"
                        )

                # --------------------------------------------------
                # DOWNLOAD
                # --------------------------------------------------

                response = requests.get(
                    url,
                    timeout=60,
                    verify=False,
                    allow_redirects=True
                )

                if response.status_code == 200:

                    with open(
                        filepath,
                        "wb"
                    ) as pdf:

                        pdf.write(
                            response.content
                        )

                    downloaded += 1

                    print(
                        f"[OK] {year} Week {week}"
                    )

                    log_writer.writerow([
                        year,
                        week,
                        "success",
                        filename,
                        url
                    ])

                else:

                    failed += 1

                    print(
                        f"[FAILED] {year} "
                        f"Week {week} "
                        f"Status={response.status_code}"
                    )

                    log_writer.writerow([
                        year,
                        week,
                        "failed",
                        filename,
                        url
                    ])

            except Exception as e:

                failed += 1

                print(
                    f"[ERROR] {year} "
                    f"Week {week} -> {e}"
                )

                log_writer.writerow([
                    year,
                    week,
                    f"error: {str(e)}",
                    filename,
                    url
                ])

print("\n" + "=" * 40)
print(f"Downloaded : {downloaded}")
print(f"Skipped    : {skipped}")
print(f"Failed     : {failed}")
print("=" * 40)
print(f"Log saved  : {LOG_FILE}")