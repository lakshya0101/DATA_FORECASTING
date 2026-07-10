import pandas as pd

# -----------------------------
# LOAD EXCEL
# -----------------------------

df = pd.read_excel(
    "data/merged_outbreaks.xlsx"
)

print("Original Shape:", df.shape)

# -----------------------------
# REMOVE UNWANTED COLUMNS
# -----------------------------

df.drop(
    columns=[
        "start_date",
        "report_date",
        "status",
        "page"
    ],
    inplace=True,
    errors="ignore"
)

# -----------------------------
# STANDARDIZE COLUMN NAMES
# -----------------------------

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
)

# -----------------------------
# DISEASE NORMALIZATION
# -----------------------------

disease_map = {

    "Dengue Fever": "Dengue",
    "Dengue / DHF": "Dengue",
    "Dengue Fever/DHF": "Dengue",

    "Acute Diarrhoeal Disease": "ADD",
    "Acute Diarrheal Disease": "ADD",
    "ADD": "ADD",

    "Chicken Pox": "Chickenpox",

    "Measles/Rubella": "Measles-Rubella",
    "MR": "Measles-Rubella",

    "COVID 19": "COVID-19",

    "Influenza Like Illness": "ILI"
}

# Clean disease column
df["disease"] = (
    df["disease"]
    .astype(str)
    .str.strip()
)

# Apply mapping
df["disease"] = (
    df["disease"]
    .replace(disease_map)
)

# -----------------------------
# STATE NORMALIZATION
# -----------------------------

state_map = {

    "NCT Delhi": "Delhi",
    "Orissa": "Odisha",
    "Uttaranchal": "Uttarakhand"
}

df["state"] = (
    df["state"]
    .astype(str)
    .str.strip()
)

df["state"] = (
    df["state"]
    .replace(state_map)
)

# -----------------------------
# REMOVE DUPLICATES
# -----------------------------

df.drop_duplicates(inplace=True)

# -----------------------------
# HANDLE MISSING VALUES
# -----------------------------

df = df[df["disease"].notna()]
df = df[df["state"].notna()]

# -----------------------------
# SAVE CLEAN FILE
# -----------------------------

output_path = (
    "data/final_normalized_outbreaks.xlsx"
)

df.to_excel(
    output_path,
    index=False
)

print("\nNormalization Complete.")
print("Final Shape:", df.shape)
print(f"Saved File: {output_path}")