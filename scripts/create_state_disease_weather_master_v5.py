import pandas as pd

print("Loading files...")

# ==========================
# WEATHER DATA
# ==========================

weather_df = pd.read_excel(
    "data/Weather_State_Monthly.xlsx"
)

weather_df.columns = (
    weather_df.columns
    .str.strip()
    .str.lower()
)

# ==========================
# CURRENT MASTER DATA
# ==========================

master_df = pd.read_excel(
    "data/state_disease_weather_master.xlsx"
)

master_df.columns = (
    master_df.columns
    .str.strip()
    .str.lower()
)

# ==========================
# ALL STATES
# ==========================

states = sorted(
    weather_df["state"]
    .dropna()
    .unique()
)

# ==========================
# ALL DISEASES
# ==========================

diseases = sorted(
    master_df["disease"]
    .dropna()
    .unique()
)

print(
    f"States: {len(states)}"
)

print(
    f"Diseases: {len(diseases)}"
)

# ==========================
# CREATE FULL GRID
# ==========================

records = []

for state in states:

    print(
        f"Building {state}"
    )

    state_weather = weather_df[
        weather_df["state"] == state
    ]

    for disease in diseases:

        for _, row in state_weather.iterrows():

            records.append([
                state,
                disease,
                row["year"],
                row["month"],
                row["temperature"],
                row["rainfall"],
                row["humidity"]
            ])

full_grid = pd.DataFrame(
    records,
    columns=[
        "state",
        "disease",
        "year",
        "month",
        "temperature",
        "rainfall",
        "humidity"
    ]
)

print(
    f"Grid Rows: {len(full_grid)}"
)

# ==========================
# MERGE CASES
# ==========================

cases_df = master_df[
    [
        "state",
        "disease",
        "year",
        "month",
        "cases"
    ]
].copy()

v5_df = full_grid.merge(
    cases_df,
    how="left",
    on=[
        "state",
        "disease",
        "year",
        "month"
    ]
)

# ==========================
# FILL MISSING CASES
# ==========================

v5_df["cases"] = (
    v5_df["cases"]
    .fillna(0)
    .astype(int)
)

# ==========================
# SORT
# ==========================

v5_df = v5_df.sort_values(
    [
        "state",
        "disease",
        "year",
        "month"
    ]
)

# ==========================
# SAVE
# ==========================

output_file = (
    "data/state_disease_weather_master_v5.xlsx"
)

v5_df.to_excel(
    output_file,
    index=False
)

print(
    f"\nSaved: {output_file}"
)

print(
    f"Total Rows: {len(v5_df)}"
)