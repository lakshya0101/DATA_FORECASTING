# # df["disease"].value_counts().head(20)
# # df.groupby("year")["cases"].sum()
# # df.groupby("state")["cases"].sum()
# # df.groupby("disease")["deaths"].sum()


# # # ~~~~~~~~~~~~~~~~~~~~~~~~~

# # weather.groupby(
# #     weather["datetime"].dt.month
# # )["temp"].mean()

# # weather.groupby(
# #     weather["datetime"].dt.month
# # )["precip"].sum()

# # weather.groupby(
# #     weather["datetime"].dt.month
# # )["humidity"].mean()


# # weather["datetime"] = pd.to_datetime(weather["datetime"])

# # weather["year"] = weather["datetime"].dt.year

# # weather["week"] = weather["datetime"].dt.isocalendar().week


# # weather_weekly = (
# #     weather
# #     .groupby(["year","week"])
# #     .agg({
# #         "temp":"mean",
# #         "humidity":"mean",
# #         "precip":"sum"
# #     })
# #     .reset_index()
# # )

# # disease_weekly = (
# #     disease
# #     .groupby(["year","week"])
# #     .agg({
# #         "cases":"sum",
# #         "deaths":"sum"
# #     })
# #     .reset_index()
# # )


# # merged = pd.merge(
# #     disease_weekly,
# #     weather_weekly,
# #     on=["year","week"],
# #     how="inner"
# # )

# # #CORRELATION ANALYSIS

# # merged[
# #     [
# #         "cases",
# #         "temp",
# #         "humidity",
# #         "precip"
# #     ]
# # ].corr()


# # #LAG ANALYSIS

# # merged["precip_lag4"] = merged["precip"].shift(4)

# # merged["precip_lag8"] = merged["precip"].shift(8)

# # merged["cases"].corr(
# #     merged["precip_lag4"]
# # )



# import pandas as pd
# import os

# # ==========================================
# # FILE PATHS
# # ==========================================

# DISEASE_FILE = "data/final_disease_normalized_v2.xlsx"

# WEATHER_FILE = "data/india_temp_with_station_locations.xlsx"

# OUTPUT_DIR = "eda_outputs"

# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # ==========================================
# # LOAD DATA
# # ==========================================

# print("Loading datasets...")

# # disease = pd.read_excel(DISEASE_FILE)

# # weather = pd.read_excel(WEATHER_FILE)

# # print("Disease Shape:", disease.shape)
# # print("Weather Shape:", weather.shape)

# disease = pd.read_excel(DISEASE_FILE)

# weather = pd.read_excel(WEATHER_FILE)

# # Standardize column names
# disease.columns = (
#     disease.columns
#     .str.strip()
#     .str.lower()
# )

# weather.columns = (
#     weather.columns
#     .str.strip()
#     .str.lower()
# )

# print("Disease Shape:", disease.shape)
# print("Weather Shape:", weather.shape)

# print("\nDisease Columns:")
# print(disease.columns.tolist())

# print("\nWeather Columns:")
# print(weather.columns.tolist())

# # ==========================================
# # DISEASE EDA
# # ==========================================

# print("\nRunning Disease EDA...")

# # Top 20 diseases
# top_diseases = (
#     disease["disease"]
#     .value_counts()
#     .head(20)
#     .reset_index()
# )

# top_diseases.columns = [
#     "Disease",
#     "Count"
# ]

# # Cases by year
# cases_by_year = (
#     disease
#     .groupby("year")["cases"]
#     .sum()
#     .reset_index()
# )

# # Cases by state
# cases_by_state = (
#     disease
#     .groupby("state")["cases"]
#     .sum()
#     .sort_values(ascending=False)
#     .reset_index()
# )

# # Deaths by disease
# deaths_by_disease = (
#     disease
#     .groupby("disease")["deaths"]
#     .sum()
#     .sort_values(ascending=False)
#     .reset_index()
# )

# # Save Disease EDA

# with pd.ExcelWriter(
#     f"{OUTPUT_DIR}/disease_eda.xlsx"
# ) as writer:

#     top_diseases.to_excel(
#         writer,
#         sheet_name="Top Diseases",
#         index=False
#     )

#     cases_by_year.to_excel(
#         writer,
#         sheet_name="Cases By Year",
#         index=False
#     )

#     cases_by_state.to_excel(
#         writer,
#         sheet_name="Cases By State",
#         index=False
#     )

#     deaths_by_disease.to_excel(
#         writer,
#         sheet_name="Deaths By Disease",
#         index=False
#     )

# print("Disease EDA saved.")

# # ==========================================
# # WEATHER EDA
# # ==========================================

# print("\nRunning Weather EDA...")

# weather["datetime"] = pd.to_datetime(
#     weather["datetime"]
# )

# # Monthly temperature
# monthly_temp = (
#     weather
#     .groupby(
#         weather["datetime"].dt.month
#     )["temp"]
#     .mean()
#     .reset_index()
# )

# monthly_temp.columns = [
#     "Month",
#     "Avg Temperature"
# ]

# # Monthly rainfall
# monthly_precip = (
#     weather
#     .groupby(
#         weather["datetime"].dt.month
#     )["precip"]
#     .sum()
#     .reset_index()
# )

# monthly_precip.columns = [
#     "Month",
#     "Total Rainfall"
# ]

# # Monthly humidity
# monthly_humidity = (
#     weather
#     .groupby(
#         weather["datetime"].dt.month
#     )["humidity"]
#     .mean()
#     .reset_index()
# )

# monthly_humidity.columns = [
#     "Month",
#     "Avg Humidity"
# ]

# with pd.ExcelWriter(
#     f"{OUTPUT_DIR}/weather_eda.xlsx"
# ) as writer:

#     monthly_temp.to_excel(
#         writer,
#         sheet_name="Monthly Temp",
#         index=False
#     )

#     monthly_precip.to_excel(
#         writer,
#         sheet_name="Monthly Rainfall",
#         index=False
#     )

#     monthly_humidity.to_excel(
#         writer,
#         sheet_name="Monthly Humidity",
#         index=False
#     )

# print("Weather EDA saved.")

# # ==========================================
# # WEEKLY WEATHER AGGREGATION
# # ==========================================

# print("\nCreating weekly weather dataset...")

# weather["year"] = (
#     weather["datetime"].dt.year
# )

# weather["week"] = (
#     weather["datetime"]
#     .dt.isocalendar()
#     .week
# )

# weather_weekly = (
#     weather
#     .groupby(["year", "week"])
#     .agg({
#         "temp": "mean",
#         "humidity": "mean",
#         "precip": "sum"
#     })
#     .reset_index()
# )

# # ==========================================
# # WEEKLY DISEASE AGGREGATION
# # ==========================================

# print("Creating weekly disease dataset...")

# disease_weekly = (
#     disease
#     .groupby(["year", "week"])
#     .agg({
#         "cases": "sum",
#         "deaths": "sum"
#     })
#     .reset_index()
# )

# # ==========================================
# # MERGE DATASETS
# # ==========================================

# print("Merging datasets...")

# merged = pd.merge(
#     disease_weekly,
#     weather_weekly,
#     on=["year", "week"],
#     how="inner"
# )

# print("Merged Shape:", merged.shape)

# merged.to_excel(
#     f"{OUTPUT_DIR}/merged_weather_disease.xlsx",
#     index=False
# )

# # ==========================================
# # CORRELATION ANALYSIS
# # ==========================================

# print("\nRunning correlation analysis...")

# correlation_matrix = merged[
#     [
#         "cases",
#         "deaths",
#         "temp",
#         "humidity",
#         "precip"
#     ]
# ].corr()

# correlation_matrix.to_excel(
#     f"{OUTPUT_DIR}/correlation_matrix.xlsx"
# )

# print(correlation_matrix)

# # ==========================================
# # LAG ANALYSIS
# # ==========================================

# print("\nRunning lag analysis...")

# merged["precip_lag2"] = (
#     merged["precip"].shift(2)
# )

# merged["precip_lag4"] = (
#     merged["precip"].shift(4)
# )

# merged["precip_lag8"] = (
#     merged["precip"].shift(8)
# )

# lag_results = pd.DataFrame({

#     "Lag": [
#         "2 Weeks",
#         "4 Weeks",
#         "8 Weeks"
#     ],

#     "Correlation": [

#         merged["cases"].corr(
#             merged["precip_lag2"]
#         ),

#         merged["cases"].corr(
#             merged["precip_lag4"]
#         ),

#         merged["cases"].corr(
#             merged["precip_lag8"]
#         )
#     ]
# })

# lag_results.to_excel(
#     f"{OUTPUT_DIR}/lag_analysis.xlsx",
#     index=False
# )

# print("\nLag Analysis:")
# print(lag_results)

# # ==========================================
# # SAVE FINAL MERGED FILE
# # ==========================================

# merged.to_excel(
#     f"{OUTPUT_DIR}/final_merged_dataset.xlsx",
#     index=False
# )

# print("\n================================")
# print("EDA COMPLETE")
# print("Outputs saved in:")
# print(OUTPUT_DIR)
# print("================================")


######

import pandas as pd
import os

# ==========================================
# FILE PATHS
# ==========================================

DISEASE_FILE = "data/final_disease_normalized_v2.xlsx"
WEATHER_FILE = "data/india_temp_with_station_locations.xlsx"

OUTPUT_DIR = "eda_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# LOAD DATA
# ==========================================

print("Loading datasets...")

disease = pd.read_excel(DISEASE_FILE)
weather = pd.read_excel(WEATHER_FILE)

# Standardize column names
disease.columns = disease.columns.str.strip().str.lower()
weather.columns = weather.columns.str.strip().str.lower()

print("Disease Shape:", disease.shape)
print("Weather Shape:", weather.shape)

print("\nDisease Columns:")
print(disease.columns.tolist())

print("\nWeather Columns:")
print(weather.columns.tolist())

# ==========================================
# DISEASE EDA
# ==========================================

print("\nRunning Disease EDA...")

top_diseases = (
    disease["disease"]
    .value_counts()
    .head(20)
    .reset_index()
)

top_diseases.columns = ["Disease", "Count"]

cases_by_year = (
    disease.groupby("year")["cases"]
    .sum()
    .reset_index()
)

cases_by_state = (
    disease.groupby("state")["cases"]
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)

deaths_by_disease = (
    disease.groupby("disease")["deaths"]
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)

with pd.ExcelWriter(
    f"{OUTPUT_DIR}/disease_eda.xlsx"
) as writer:

    top_diseases.to_excel(
        writer,
        sheet_name="Top Diseases",
        index=False
    )

    cases_by_year.to_excel(
        writer,
        sheet_name="Cases By Year",
        index=False
    )

    cases_by_state.to_excel(
        writer,
        sheet_name="Cases By State",
        index=False
    )

    deaths_by_disease.to_excel(
        writer,
        sheet_name="Deaths By Disease",
        index=False
    )

print("Disease EDA saved.")

# ==========================================
# WEATHER EDA
# ==========================================

print("\nRunning Weather EDA...")

weather["datetime"] = pd.to_datetime(
    weather["datetime"]
)

monthly_temp = (
    weather.groupby(
        weather["datetime"].dt.month
    )["temp"]
    .mean()
    .reset_index()
)

monthly_temp.columns = [
    "Month",
    "Avg Temperature"
]

monthly_precip = (
    weather.groupby(
        weather["datetime"].dt.month
    )["precip"]
    .sum()
    .reset_index()
)

monthly_precip.columns = [
    "Month",
    "Total Rainfall"
]

monthly_humidity = (
    weather.groupby(
        weather["datetime"].dt.month
    )["humidity"]
    .mean()
    .reset_index()
)

monthly_humidity.columns = [
    "Month",
    "Avg Humidity"
]

with pd.ExcelWriter(
    f"{OUTPUT_DIR}/weather_eda.xlsx"
) as writer:

    monthly_temp.to_excel(
        writer,
        sheet_name="Monthly Temp",
        index=False
    )

    monthly_precip.to_excel(
        writer,
        sheet_name="Monthly Rainfall",
        index=False
    )

    monthly_humidity.to_excel(
        writer,
        sheet_name="Monthly Humidity",
        index=False
    )

print("Weather EDA saved.")

# ==========================================
# YEARLY WEATHER AGGREGATION
# ==========================================

print("\nCreating yearly weather dataset...")

weather["year"] = weather["datetime"].dt.year

weather_yearly = (
    weather
    .groupby("year")
    .agg({
        "temp": "mean",
        "humidity": "mean",
        "precip": "sum"
    })
    .reset_index()
)

# ==========================================
# YEARLY DISEASE AGGREGATION
# ==========================================

print("Creating yearly disease dataset...")

disease_yearly = (
    disease
    .groupby("year")
    .agg({
        "cases": "sum",
        "deaths": "sum"
    })
    .reset_index()
)

# ==========================================
# MERGE
# ==========================================

print("Merging yearly datasets...")

merged = pd.merge(
    disease_yearly,
    weather_yearly,
    on="year",
    how="inner"
)

print("Merged Shape:", merged.shape)

merged.to_excel(
    f"{OUTPUT_DIR}/merged_yearly_weather_disease.xlsx",
    index=False
)

# ==========================================
# CORRELATION ANALYSIS
# ==========================================

print("\nRunning correlation analysis...")

correlation_matrix = merged[
    [
        "cases",
        "deaths",
        "temp",
        "humidity",
        "precip"
    ]
].corr()

print("\nCorrelation Matrix:")
print(correlation_matrix)

correlation_matrix.to_excel(
    f"{OUTPUT_DIR}/correlation_matrix.xlsx"
)

# ==========================================
# SAVE OUTPUTS
# ==========================================

merged.to_excel(
    f"{OUTPUT_DIR}/final_merged_dataset.xlsx",
    index=False
)

print("\n===================================")
print("EDA COMPLETED SUCCESSFULLY")
print("Outputs saved in:")
print(OUTPUT_DIR)
print("===================================")