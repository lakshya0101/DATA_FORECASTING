import pandas as pd

print("Loading files...")

files = [
    (
        "data/state_disease_temperature_lag_analysis_v5.xlsx",
        "Temperature"
    ),
    (
        "data/state_disease_humidity_lag_analysis_v5.xlsx",
        "Humidity"
    ),
    (
        "data/state_disease_rainfall_lag_analysis_v5.xlsx",
        "Rainfall"
    )
]

results = []

for file, factor in files:

    print(f"Processing {factor}")

    df = pd.read_excel(file)

    df.columns = (
        df.columns
        .str.strip()
    )

    significant = df[
        df["Best_Correlation"]
        .abs()
        >= 0.55
    ].copy()

    significant["Weather_Factor"] = factor

    results.append(significant)

final_df = pd.concat(
    results,
    ignore_index=True
)

final_df = final_df[
    [
        "State",
        "Disease",
        "Weather_Factor",
        "Lag1",
        "Lag2",
        "Lag3",
        "Best_Lag",
        "Best_Correlation"
    ]
]

final_df = final_df.sort_values(
    "Best_Correlation",
    key=lambda x: x.abs(),
    ascending=False
)

final_df["Relationship_Strength"] = (
    final_df["Best_Correlation"]
    .abs()
    .apply(
        lambda x:
        "Very Strong"
        if x >= 0.75
        else "Moderately Strong"
    )
)

output_file = (
    "data/Significant_Weather_Disease_Relationships.xlsx"
)

final_df.to_excel(
    output_file,
    index=False
)

print(
    f"\nSaved: {output_file}"
)

print(
    f"Total Significant Relationships: "
    f"{len(final_df)}"
)