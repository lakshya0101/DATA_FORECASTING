import pandas as pd

print("Loading Karnataka-Chikungunya dataset...")

# =====================================
# LOAD FILE CREATED EARLIER
# =====================================

df = pd.read_excel(
    "data/karnataka_chikungunya_reference_format.xlsx",
    sheet_name="Monthly_Data"
)

print("Rows:", len(df))

# =====================================
# CREATE LAGS
# =====================================

for lag in [1, 2, 3]:

    df[f"rainfall_lag_{lag}"] = (
        df["Rainfall"].shift(lag)
    )

    df[f"humidity_lag_{lag}"] = (
        df["Humidity"].shift(lag)
    )

    df[f"temperature_lag_{lag}"] = (
        df["Temperature"].shift(lag)
    )

# =====================================
# CALCULATE CORRELATIONS
# =====================================

results = []

for factor in [
    "rainfall",
    "humidity",
    "temperature"
]:

    for lag in [1, 2, 3]:

        corr = df["Sum_Weekly_Cases"].corr(
            df[f"{factor}_lag_{lag}"]
        )

        results.append({
            "Weather_Factor": factor.title(),
            "Lag_Months": lag,
            "Correlation": corr
        })

result_df = pd.DataFrame(results)

# =====================================
# BEST LAG PER FACTOR
# =====================================

best_results = []

for factor in [
    "Rainfall",
    "Humidity",
    "Temperature"
]:

    temp = result_df[
        result_df["Weather_Factor"] == factor
    ].copy()

    best_row = temp.loc[
        temp["Correlation"].abs().idxmax()
    ]

    best_results.append({
        "Weather_Factor": factor,
        "Best_Lag": best_row["Lag_Months"],
        "Best_Correlation": best_row["Correlation"]
    })

best_df = pd.DataFrame(best_results)

# =====================================
# SAVE
# =====================================

with pd.ExcelWriter(
    "data/karnataka_chikungunya_lag_analysis.xlsx",
    engine="openpyxl"
) as writer:

    result_df.to_excel(
        writer,
        sheet_name="All_Lags",
        index=False
    )

    best_df.to_excel(
        writer,
        sheet_name="Best_Lags",
        index=False
    )

# =====================================
# PRINT
# =====================================

print("\nAll Lag Correlations\n")
print(result_df)

print("\nBest Lag Per Weather Factor\n")
print(best_df)

print(
    "\nSaved: data/karnataka_chikungunya_lag_analysis.xlsx"
)