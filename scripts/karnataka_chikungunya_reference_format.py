import pandas as pd

print("Loading datasets...")

# =====================================
# LOAD DISEASE DATA
# =====================================

disease = pd.read_excel(
    "data/final_disease_with_week.xlsx"
)

disease.columns = (
    disease.columns
    .str.strip()
    .str.lower()
)

# =====================================
# LOAD WEATHER DATA
# =====================================

weather = pd.read_excel(
    "data/Weather_State_Monthly.xlsx"
)

weather.columns = (
    weather.columns
    .str.strip()
    .str.lower()
)

# =====================================
# DATE CONVERSION
# =====================================

disease["final_date"] = pd.to_datetime(
    disease["final_date"],
    errors="coerce"
)

disease["month"] = (
    disease["final_date"]
    .dt.month
)

# =====================================
# FILTER
# =====================================

filtered = disease[
    (disease["state"] == "Karnataka") &
    (disease["disease"] == "Chikungunya")
].copy()

print(
    "\nOutbreak Records Found:",
    len(filtered)
)

# =====================================
# MONTHLY SUM OF WEEKLY CASES
# =====================================

# monthly_cases = (
#     filtered
#     .groupby(
#         ["year", "month"],
#         as_index=False
#     )
#     .agg(
#         {
#             "cases": "sum"
#         }
#     )
# )

# =====================================
# MONTHLY SUM OF WEEKLY CASES
# =====================================

monthly_cases = (
    filtered
    .groupby(
        ["year", "month"],
        as_index=False
    )["cases"]
    .sum()
)

# =====================================
# CREATE COMPLETE MONTH GRID
# =====================================

years = [2023, 2024, 2025]

all_months = pd.MultiIndex.from_product(
    [
        years,
        range(1, 13)
    ],
    names=[
        "year",
        "month"
    ]
)

all_months = (
    pd.DataFrame(index=all_months)
    .reset_index()
)

monthly_cases = pd.merge(
    all_months,
    monthly_cases,
    on=["year", "month"],
    how="left"
)

monthly_cases["cases"] = (
    monthly_cases["cases"]
    .fillna(0)
)

# =====================================
# WEATHER FILTER
# =====================================

weather_filtered = weather[
    weather["state"] == "Karnataka"
].copy()

# =====================================
# MERGE
# =====================================

merged = pd.merge(
    monthly_cases,
    weather_filtered,
    left_on=["year", "month"],
    right_on=["year", "month"],
    how="left"
)

# =====================================
# DISPLAY YEAR FORMAT
# =====================================

year_map = {
    2023: "Year 1 (2023)",
    2024: "Year 2 (2024)",
    2025: "Year 3 (2025)",
    2026: "Year 4 (2026)"
}

merged["display_year"] = (
    merged["year"]
    .map(year_map)
)

# =====================================
# MONTH NAME
# =====================================

month_map = {
    1:"January",
    2:"February",
    3:"March",
    4:"April",
    5:"May",
    6:"June",
    7:"July",
    8:"August",
    9:"September",
    10:"October",
    11:"November",
    12:"December"
}

merged["month_name"] = (
    merged["month"]
    .map(month_map)
)

# =====================================
# FINAL TABLE
# =====================================

final_table = merged[
    [
        "display_year",
        "month_name",
        "month",
        "cases",
        "temperature",
        "humidity",
        "rainfall"
    ]
].copy()

final_table.columns = [
    "Year",
    "Month",
    "Month_No",
    "Sum_Weekly_Cases",
    "Temperature",
    "Humidity",
    "Rainfall"
]

# =====================================
# CORRELATION
# =====================================

temp_corr = (
    final_table["Sum_Weekly_Cases"]
    .corr(
        final_table["Temperature"]
    )
)

humidity_corr = (
    final_table["Sum_Weekly_Cases"]
    .corr(
        final_table["Humidity"]
    )
)

rain_corr = (
    final_table["Sum_Weekly_Cases"]
    .corr(
        final_table["Rainfall"]
    )
)

corr_table = pd.DataFrame({
    "Weather_Factor": [
        "Temperature",
        "Humidity",
        "Rainfall"
    ],
    "Correlation": [
        temp_corr,
        humidity_corr,
        rain_corr
    ]
})

# =====================================
# SAVE EXCEL
# =====================================

output_file = (
    "data/karnataka_chikungunya_reference_format.xlsx"
)

with pd.ExcelWriter(
    output_file,
    engine="openpyxl"
) as writer:

    final_table.to_excel(
        writer,
        sheet_name="Monthly_Data",
        index=False
    )

    corr_table.to_excel(
        writer,
        sheet_name="Correlation",
        index=False
    )

print("\n===================================")
print("KARNATAKA - CHIKUNGUNYA ANALYSIS")
print("===================================")

print("\nMonthly Data")
print(final_table)

print("\nCorrelation Results")
print(corr_table)

print(
    f"\nSaved: {output_file}"
)