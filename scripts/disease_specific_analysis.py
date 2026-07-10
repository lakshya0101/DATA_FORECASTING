# Dengue
# Malaria
# Chikungunya
# Food Poisoning
# ADD
# Leptospirosis
# Scrub Typhus

import pandas as pd
import os

# =====================================
# FILES
# =====================================

DISEASE_FILE = "data/final_disease_with_week.xlsx"
WEATHER_FILE = "data/india_temp_with_station_locations.xlsx"

OUTPUT_DIR = "disease_specific_analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================
# LOAD DATA
# =====================================

disease = pd.read_excel(DISEASE_FILE)
weather = pd.read_excel(WEATHER_FILE)

disease.columns = disease.columns.str.strip().str.lower()
weather.columns = weather.columns.str.strip().str.lower()

# =====================================
# WEATHER WEEKLY
# =====================================

weather["datetime"] = pd.to_datetime(
    weather["datetime"]
)

weather["year"] = (
    weather["datetime"]
    .dt.isocalendar()
    .year
)

weather["week"] = (
    weather["datetime"]
    .dt.isocalendar()
    .week
)

weather_weekly = (
    weather
    .groupby(
        ["year", "week"]
    )
    .agg({
        "temp": "mean",
        "humidity": "mean",
        "precip": "sum"
    })
    .reset_index()
)

# =====================================
# TARGET DISEASES
# =====================================

target_diseases = [
    "Dengue",
    "Malaria",
    "Chikungunya",
    "Food Poisoning",
    "ADD",
    "Leptospirosis",
    "Scrub Typhus"
]

summary_rows = []

for disease_name in target_diseases:

    print(f"\nAnalyzing {disease_name}")

    disease_subset = disease[
        disease["disease"] == disease_name
    ]

    if len(disease_subset) == 0:
        print("No records found")
        continue

    disease_weekly = (
        disease_subset
        .groupby(
            ["year", "week"]
        )
        .agg({
            "cases": "sum"
        })
        .reset_index()
    )

    merged = pd.merge(
        disease_weekly,
        weather_weekly,
        on=["year", "week"],
        how="inner"
    )

    if len(merged) < 10:
        continue

    # Direct correlations

    temp_corr = merged["cases"].corr(
        merged["temp"]
    )

    humidity_corr = merged["cases"].corr(
        merged["humidity"]
    )

    precip_corr = merged["cases"].corr(
        merged["precip"]
    )

    best_lag = None
    best_corr = None

    for lag in [2, 4, 6, 8]:

        merged[f"precip_lag_{lag}"] = (
            merged["precip"]
            .shift(lag)
        )

        corr = merged["cases"].corr(
            merged[f"precip_lag_{lag}"]
        )

        # if pd.notna(corr):

        #     if abs(corr) > abs(best_corr):
        #         best_corr = corr
        #         best_lag = lag

        if pd.notna(corr):
            if best_corr is None or abs(corr) > abs(best_corr):
                best_corr = corr
                best_lag = lag



    summary_rows.append({

        "Disease": disease_name,

        "Temp_Correlation":
            temp_corr,

        "Humidity_Correlation":
            humidity_corr,

        "Rainfall_Correlation":
            precip_corr,

        "Best_Rainfall_Lag_Weeks":
            best_lag,

        "Best_Lag_Correlation":
            best_corr
    })

summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_excel(
    f"{OUTPUT_DIR}/disease_weather_summary.xlsx",
    index=False
)

print("\nSaved disease_weather_summary.xlsx") 