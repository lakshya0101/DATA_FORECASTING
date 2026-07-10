import pandas as pd

print("Loading merged dataset...")

df = pd.read_excel(
    "data/monthly_state_merged.xlsx"
)

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
)

print("Shape:", df.shape)

# ====================================
# CORRELATION
# ====================================

corr = df[
    [
        "cases",
        "deaths",
        "temperature",
        "rainfall",
        "humidity"
    ]
].corr()

print("\nCorrelation Matrix\n")
print(corr)

corr.to_excel(
    "data/monthly_correlation_matrix.xlsx"
)

# ====================================
# LAG ANALYSIS
# ====================================

df = df.sort_values(
    [
        "state",
        "year",
        "month"
    ]
)

results = []

for lag in [1,2,3]:

    temp = df.copy()

    temp["rainfall_lag"] = (
        temp
        .groupby("state")
        ["rainfall"]
        .shift(lag)
    )

    corr_value = (
        temp["cases"]
        .corr(
            temp["rainfall_lag"]
        )
    )

    results.append(
        {
            "Lag_Months": lag,
            "Correlation": corr_value
        }
    )

lag_df = pd.DataFrame(results)

print("\nRainfall Lag Analysis\n")
print(lag_df)

lag_df.to_excel(
    "data/monthly_lag_analysis.xlsx",
    index=False
)

# ====================================
# MONTHLY TRENDS
# ====================================

monthly = (
    df
    .groupby("month")
    .agg(
        {
            "cases":"sum",
            "deaths":"sum",
            "rainfall":"mean",
            "temperature":"mean",
            "humidity":"mean"
        }
    )
)

monthly.to_excel(
    "data/monthly_trends.xlsx"
)

print("\nSaved monthly trends")

print("\nAnalysis Complete")