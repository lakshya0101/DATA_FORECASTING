import pandas as pd

print("Loading disease dataset...")

df = pd.read_excel(
    "data/final_disease_monthly.xlsx"
)

# ---------------------
# Aggregate Monthly
# ---------------------

monthly = (
    df
    .groupby(
        [
            "state",
            "year",
            "month"
        ]
    )
    .agg(
        {
            "cases":"sum",
            "deaths":"sum"
        }
    )
    .reset_index()
)

print(monthly.head())

print(
    "\nRows:",
    len(monthly)
)

monthly.to_excel(
    "data/disease_monthly_summary.xlsx",
    index=False
)

print(
    "\nSaved: disease_monthly_summary.xlsx"
)