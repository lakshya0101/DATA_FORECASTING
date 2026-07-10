import pandas as pd
import numpy as np

print("Loading master dataset...")

df = pd.read_excel(
    "data/state_disease_weather_master.xlsx"
)

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
)

results = []

groups = df.groupby(
    ["state", "disease"]
)

for (state, disease), temp in groups:

    temp = temp.copy()

    # Minimum records required
    if len(temp) < 6:
        continue

    temp = temp.sort_values(
        ["year", "month"]
    )

    # Humidity lag features

    temp["hum_lag_1"] = (
        temp["humidity"].shift(1)
    )

    temp["hum_lag_2"] = (
        temp["humidity"].shift(2)
    )

    temp["hum_lag_3"] = (
        temp["humidity"].shift(3)
    )

    lag1 = temp["cases"].corr(
        temp["hum_lag_1"]
    )

    lag2 = temp["cases"].corr(
        temp["hum_lag_2"]
    )

    lag3 = temp["cases"].corr(
        temp["hum_lag_3"]
    )

    if (
        pd.isna(lag1)
        and pd.isna(lag2)
        and pd.isna(lag3)
    ):
        continue

    lag_dict = {
        1: lag1,
        2: lag2,
        3: lag3
    }

    valid_lags = {
        k: v
        for k, v in lag_dict.items()
        if pd.notna(v)
    }

    if len(valid_lags) == 0:
        continue

    best_lag = max(
        valid_lags,
        key=lambda k: abs(valid_lags[k])
    )

    best_corr = valid_lags[
        best_lag
    ]

    results.append(
        {
            "State": state,
            "Disease": disease,
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
    "data/state_disease_humidity_lag_analysis.xlsx",
    index=False
)

print("\nTop 20 Humidity Lag Relationships\n")

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
    "\nSaved: data/state_disease_humidity_lag_analysis.xlsx"
)
import pandas as pd
import numpy as np

print("Loading master dataset...")

df = pd.read_excel(
    "data/state_disease_weather_master.xlsx"
)

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
)

results = []

groups = df.groupby(
    ["state", "disease"]
)

for (state, disease), temp in groups:

    temp = temp.copy()

    # Minimum records required
    if len(temp) < 12:
        continue

    temp = temp.sort_values(
        ["year", "month"]
    )

    # Humidity lag features

    temp["hum_lag_1"] = (
        temp["humidity"].shift(1)
    )

    temp["hum_lag_2"] = (
        temp["humidity"].shift(2)
    )

    temp["hum_lag_3"] = (
        temp["humidity"].shift(3)
    )

    lag1 = temp["cases"].corr(
        temp["hum_lag_1"]
    )

    lag2 = temp["cases"].corr(
        temp["hum_lag_2"]
    )

    lag3 = temp["cases"].corr(
        temp["hum_lag_3"]
    )

    if (
        pd.isna(lag1)
        and pd.isna(lag2)
        and pd.isna(lag3)
    ):
        continue

    lag_dict = {
        1: lag1,
        2: lag2,
        3: lag3
    }

    valid_lags = {
        k: v
        for k, v in lag_dict.items()
        if pd.notna(v)
    }

    if len(valid_lags) == 0:
        continue

    best_lag = max(
        valid_lags,
        key=lambda k: abs(valid_lags[k])
    )

    best_corr = valid_lags[
        best_lag
    ]

    results.append(
        {
            "State": state,
            "Disease": disease,
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
    "data/state_disease_humidity_lag_analysis.xlsx",
    index=False
)

print("\nTop 20 Humidity Lag Relationships\n")

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
    "\nSaved: data/state_disease_humidity_lag_analysis.xlsx"
)