import pandas as pd

print("Loading master dataset...")

df = pd.read_excel(
    "data/state_disease_weather_master.xlsx"
)

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
)

# important_diseases = [
#     "Dengue",
#     "Malaria",
#     "Chikungunya",
#     "Leptospirosis",
#     "Scrub Typhus",
#     "Japanese Encephalitis",
#     "Kyasanur Forest Disease",
#     "Zika Virus",
#     "West Nile Fever"
# ]
all_diseases = sorted(
    df["disease"]
    .dropna()
    .unique()
)

results = []

# for disease_name in important_diseases:
for disease_name in all_diseases:
    disease_df = df[
        df["disease"]
        .astype(str)
        .str.contains(
            disease_name,
            case=False,
            na=False
        )
    ]

    if len(disease_df) < 10:
        continue

    states = disease_df["state"].unique()

    for state in states:

        temp = disease_df[
            disease_df["state"] == state
        ].copy()

        if len(temp) < 12:
            continue

        temp = temp.sort_values(
            ["year", "month"]
        )

        temp["rain_lag_1"] = (
            temp["rainfall"].shift(1)
        )

        temp["rain_lag_2"] = (
            temp["rainfall"].shift(2)
        )

        temp["rain_lag_3"] = (
            temp["rainfall"].shift(3)
        )

        print(
            disease_name,
            state,
            len(temp)
        )

        lag1 = temp["cases"].corr(
            temp["rain_lag_1"]
        )

        lag2 = temp["cases"].corr(
            temp["rain_lag_2"]
        )

        lag3 = temp["cases"].corr(
            temp["rain_lag_3"]
        )

        lag_dict = {
            1: lag1,
            2: lag2,
            3: lag3
        }

        best_lag = max(
            lag_dict,
            key=lambda k:
            abs(
                lag_dict[k]
            )
            if pd.notna(
                lag_dict[k]
            )
            else -1
        )

        best_corr = lag_dict[
            best_lag
        ]

        if pd.isna(lag1) and pd.isna(lag2) and pd.isna(lag3):
            continue

        results.append(
            {
                "State": state,
                "Disease": disease_name,
                "Records": len(temp),
                "Lag_1": lag1,
                "Lag_2": lag2,
                "Lag_3": lag3,
                "Best_Lag": best_lag,
                "Best_Correlation": best_corr
            }
        )

result_df = pd.DataFrame(
    results
)

result_df = result_df.sort_values(
    "Best_Correlation",
    ascending=False
)

result_df.to_excel(
    "data/state_disease_lag_analysis_full.xlsx",
    index=False
)

print("\nTop 20 Lag Relationships\n")

print(
    result_df[
        [
            "State",
            "Disease",
            "Best_Lag",
            "Best_Correlation"
        ]
    ]
    .head(20)
)

print(
    "\nSaved: data/state_disease_lag_analysis_full.xlsx"
)