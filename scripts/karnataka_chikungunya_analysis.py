import pandas as pd

print("Loading master dataset...")

df = pd.read_excel(
    "data/state_disease_weather_master.xlsx"
)

# -------------------------
# Filter
# -------------------------

filtered = df[
    (df["state"] == "Karnataka") &
    (df["disease"] == "Chikungunya")
].copy()

print("\nRows Found:", len(filtered))

if len(filtered) == 0:
    print("No data found")
    exit()

# -------------------------
# Sort
# -------------------------

filtered = filtered.sort_values(
    ["year", "month"]
)

# -------------------------
# Correlations
# -------------------------

temp_corr = filtered["cases"].corr(
    filtered["temperature"]
)

humidity_corr = filtered["cases"].corr(
    filtered["humidity"]
)

rain_corr = filtered["cases"].corr(
    filtered["rainfall"]
)

# -------------------------
# Output Sheet
# -------------------------

output = filtered[
    [
        "year",
        "month",
        "cases",
        "temperature",
        "humidity",
        "rainfall"
    ]
]

with pd.ExcelWriter(
    "data/karnataka_chikungunya_analysis.xlsx",
    engine="openpyxl"
) as writer:

    output.to_excel(
        writer,
        sheet_name="Data",
        index=False
    )

    corr_df = pd.DataFrame({
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

    corr_df.to_excel(
        writer,
        sheet_name="Correlation",
        index=False
    )

print("\nCorrelation Results")

print(
    pd.DataFrame({
        "Weather_Factor":[
            "Temperature",
            "Humidity",
            "Rainfall"
        ],
        "Correlation":[
            temp_corr,
            humidity_corr,
            rain_corr
        ]
    })
)

print(
    "\nSaved: data/karnataka_chikungunya_analysis.xlsx"
)