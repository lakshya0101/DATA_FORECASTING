import pandas as pd

print("Loading master dataset...")

df = pd.read_excel(
    "data/state_disease_weather_master.xlsx"
)

print("Shape:", df.shape)

results = []

groups = df.groupby(
    ["state", "disease"]
)

for (state, disease), group in groups:

    if len(group) < 6:
        continue

    rain_corr = group["cases"].corr(
        group["rainfall"]
    )

    temp_corr = group["cases"].corr(
        group["temperature"]
    )

    humidity_corr = group["cases"].corr(
        group["humidity"]
    )

    results.append(
        {
            "State": state,
            "Disease": disease,
            "Records": len(group),
            "Rainfall_Corr": rain_corr,
            "Temperature_Corr": temp_corr,
            "Humidity_Corr": humidity_corr
        }
    )

corr_df = pd.DataFrame(results)

corr_df = corr_df.sort_values(
    by="Rainfall_Corr",
    ascending=False
)

corr_df.to_excel(
    "data/state_disease_correlations.xlsx",
    index=False
)

print("\nTop 20 Rainfall Relationships\n")

print(
    corr_df[
        [
            "State",
            "Disease",
            "Rainfall_Corr"
        ]
    ]
    .head(20)
)

print(
    "\nSaved: data/state_disease_correlations.xlsx"
)