# import pandas as pd

# # -----------------------------
# # LOAD DATA
# # -----------------------------

# df = pd.read_excel("data/final_normalized_outbreaks.xlsx")

# print("Before normalization:", df["disease"].nunique())

# # -----------------------------
# # DISEASE MAPPING
# # -----------------------------

# disease_map = {

#     "Acute Diarrhea Disease": "ADD",
#     "Acute diarrheal Disease": "ADD",
#     "Acute Diarrheal Diseases": "ADD",
#     "Acute Diarrheal Disease (Klebsiella)": "ADD",
#     "Acute Diarrheal Disease (Shigellosis)": "ADD",
#     "Acute Diarrhoeal Disease ( Klebsiella)": "ADD",

#     "Acute Gastritis": "Gastroenteritis",
#     "Acute Gastroente ritis": "Gastroenteritis",
#     "Acute Gastroenter itis": "Gastroenteritis",
#     "Acute Gastroenteri tis": "Gastroenteritis",
#     "Acute Gastroenterit is": "Gastroenteritis",
#     "Acute Gastroenteriti": "Gastroenteritis",
#     "Acute Gastroenteriti s": "Gastroenteritis",
#     "Acute Gastroenteritis": "Gastroenteritis",
#     "Acute Gastroenterit is (Norovirus)": "Gastroenteritis",

#     "Chicken pox": "Chickenpox",
#     "Chickenp ox": "Chickenpox",
#     "Chickenpo x": "Chickenpox",
#     "Chikecke npox": "Chickenpox",
#     "Chikenpox": "Chickenpox",

#     "Chikungun ya": "Chikungunya",
#     "Chikunguny a": "Chikungunya",

#     "Food Poisioning": "Food Poisoning",
#     "Food poisoning": "Food Poisoning",
#     "Suspecte d Food Poisoning": "Food Poisoning",
#     "Suspected Food Poisoning": "Food Poisoning",
#     "Suspected Food poisoning": "Food Poisoning",
#     "Food Poisoning (Norovirus)": "Food Poisoning",
#     "Food Poisoning (Mushroom Poisoning)": "Food Poisoning",
#     "Food Poisoning (Shigella Sonnie)": "Food Poisoning",
#     "Suspected Food Poisoning (Klebsiella)": "Food Poisoning",
#     "Suspected Food Poisoning (Salmonella)": "Food Poisoning",

#     "Acute Hepatitis": "Hepatitis",
#     "Acute Hepatitis A": "Hepatitis A",
#     "Acute Hepatitis B": "Hepatitis B",
#     "Acute Hepatitis E": "Hepatitis E",
#     "Acute Hepatitis-A": "Hepatitis A",
#     "Hepatitis-A": "Hepatitis A",
#     "Suspected Hepatitis": "Hepatitis",

#     "Hand Foot Mouth Disease (HFMD)": "HFMD",
#     "Hand Foot Mouth Diseases.": "HFMD",
#     "Hand Foot and Mouth Disease": "HFMD",
#     "Hand, Foot and Mouth Disease": "HFMD",
#     "Hand, Foot, and Mouth Disease (HFMD)": "HFMD",

#     "ARI- Influenza Like Illness(ILI)": "ILI",
#     "Influenza": "ILI",
#     "Influenza H3N2": "ILI",

#     "Leptospiro sis": "Leptospirosis",
#     "Leptospirosi s": "Leptospirosis",

#     "Monkey Pox": "Mpox",
#     "Monkey pox": "Mpox",
#     "Mpox (Clade I)": "Mpox",
#     "Mpox (Clade II)": "Mpox",
#     "Mpox (Clade l)": "Mpox",
#     "Mpox (Clade- I)": "Mpox",

#     "Fever of Unknown Cause": "FUO",
#     "Fever of Unknown Origin": "FUO",
#     "Fever of unknown origin": "FUO",
#     "Pyrexia of Unknown Origin": "FUO",
#     "Pyrexia of unknown origin": "FUO",

#     "Scrub typhus": "Scrub Typhus",

#     "Typhoid Fever": "Typhoid",
#     "Suspected Typhoid": "Typhoid",
#     "Surpected Typhoid": "Typhoid",

#     "Measles & Rubella": "Measles-Rubella",
#     "Measles and Rubella": "Measles-Rubella",

#     "Dog Bite": "Animal Bite - Dog Bite"
# }

# # Apply mapping
# df["disease"] = df["disease"].replace(disease_map)

# # Mixed infections
# mixed_map = {
#     "Dengue & Chikungunya": "Mixed Infection",
#     "Dengue and Chikungunya": "Mixed Infection",
#     "Leptospirosis & Dengue": "Mixed Infection",
#     "Dengue & Scrub Typhus": "Mixed Infection",
#     "Hepatitis A & E": "Mixed Infection",
#     "Hepatitis A/E": "Mixed Infection",
#     "Hepatitis A&E": "Mixed Infection",
#     "Hepatitis A and E": "Mixed Infection",
#     "Typhoid & Hepatitis E": "Mixed Infection"
# }

# df["disease"] = df["disease"].replace(mixed_map)

# print("After normalization:", df["disease"].nunique())

# # Save
# output_file = "data/final_disease_normalized.xlsx"

# df.to_excel(output_file, index=False)

# print(f"Saved: {output_file}")

# # Disease frequency table
# summary = df["disease"].value_counts().reset_index()
# summary.columns = ["Disease", "Count"]

# summary.to_excel(
#     "data/disease_summary.xlsx",
#     index=False
# )

# print("Disease summary saved.")


import pandas as pd

# ==========================================
# LOAD DATA
# ==========================================

INPUT_FILE = "data/final_normalized_outbreaks.xlsx"

df = pd.read_excel(INPUT_FILE)

print("Before normalization:", df["disease"].nunique())

# ==========================================
# PHASE 1 NORMALIZATION
# ==========================================

disease_map = {

    # ADD
    "Acute Diarrhea Disease": "ADD",
    "Acute diarrheal Disease": "ADD",
    "Acute Diarrheal Diseases": "ADD",
    "Acute Diarrheal Disease (Klebsiella)": "ADD",
    "Acute Diarrheal Disease (Shigellosis)": "ADD",
    "Acute Diarrhoeal Disease ( Klebsiella)": "ADD",

    # Gastroenteritis
    "Acute Gastritis": "Gastroenteritis",
    "Acute Gastroente ritis": "Gastroenteritis",
    "Acute Gastroenter itis": "Gastroenteritis",
    "Acute Gastroenteri tis": "Gastroenteritis",
    "Acute Gastroenterit is": "Gastroenteritis",
    "Acute Gastroenteriti": "Gastroenteritis",
    "Acute Gastroenteriti s": "Gastroenteritis",
    "Acute Gastroenteritis": "Gastroenteritis",
    "Acute Gastroenterit is (Norovirus)": "Gastroenteritis",

    # Chickenpox
    "Chicken pox": "Chickenpox",
    "Chickenp ox": "Chickenpox",
    "Chickenpo x": "Chickenpox",
    "Chikecke npox": "Chickenpox",
    "Chikenpox": "Chickenpox",

    # Chikungunya
    "Chikungun ya": "Chikungunya",
    "Chikunguny a": "Chikungunya",

    # Food Poisoning
    "Food Poisioning": "Food Poisoning",
    "Food poisoning": "Food Poisoning",
    "Suspecte d Food Poisoning": "Food Poisoning",
    "Suspected Food Poisoning": "Food Poisoning",
    "Suspected Food poisoning": "Food Poisoning",
    "Food Poisoning (Norovirus)": "Food Poisoning",
    "Food Poisoning (Mushroom Poisoning)": "Food Poisoning",
    "Food Poisoning (Shigella Sonnie)": "Food Poisoning",
    "Suspected Food Poisoning (Klebsiella)": "Food Poisoning",
    "Suspected Food Poisoning (Salmonella)": "Food Poisoning",

    # Hepatitis
    "Acute Hepatitis": "Hepatitis",
    "Acute Hepatitis A": "Hepatitis A",
    "Acute Hepatitis B": "Hepatitis B",
    "Acute Hepatitis E": "Hepatitis E",
    "Acute Hepatitis-A": "Hepatitis A",
    "Hepatitis-A": "Hepatitis A",
    "Suspected Hepatitis": "Hepatitis",

    # HFMD
    "Hand Foot Mouth Disease (HFMD)": "HFMD",
    "Hand Foot Mouth Diseases.": "HFMD",
    "Hand Foot and Mouth Disease": "HFMD",
    "Hand, Foot and Mouth Disease": "HFMD",
    "Hand, Foot, and Mouth Disease (HFMD)": "HFMD",

    # Influenza
    "ARI- Influenza Like Illness(ILI)": "ILI",
    "Influenza": "ILI",
    "Influenza H3N2": "ILI",

    # Leptospirosis
    "Leptospiro sis": "Leptospirosis",
    "Leptospirosi s": "Leptospirosis",

    # Mpox
    "Monkey Pox": "Mpox",
    "Monkey pox": "Mpox",
    "Mpox (Clade I)": "Mpox",
    "Mpox (Clade II)": "Mpox",
    "Mpox (Clade l)": "Mpox",
    "Mpox (Clade- I)": "Mpox",

    # Fever
    "Fever of Unknown Cause": "FUO",
    "Fever of Unknown Origin": "FUO",
    "Fever of unknown origin": "FUO",
    "Pyrexia of Unknown Origin": "FUO",
    "Pyrexia of unknown origin": "FUO",

    # Scrub Typhus
    "Scrub typhus": "Scrub Typhus",

    # Typhoid
    "Typhoid Fever": "Typhoid",
    "Suspected Typhoid": "Typhoid",
    "Surpected Typhoid": "Typhoid",

    # Measles Rubella
    "Measles & Rubella": "Measles-Rubella",
    "Measles and Rubella": "Measles-Rubella",

    # Animal Bite
    "Dog Bite": "Animal Bite - Dog Bite"
}

df["disease"] = df["disease"].replace(disease_map)

print("After Phase 1:", df["disease"].nunique())

# ==========================================
# PHASE 2 NORMALIZATION
# ==========================================

# Malaria grouping
df["disease"] = df["disease"].replace(
    regex={
        r"^Malaria.*": "Malaria",
        r"^Mixed Malaria.*": "Malaria",
        r"^Complicate.*Malaria.*": "Malaria"
    }
)

# Hepatitis grouping
df["disease"] = df["disease"].replace(
    regex={
        r"^Hepatitis.*": "Hepatitis",
        r"^Viral Hepatitis.*": "Hepatitis"
    }
)

# Fever grouping
df["disease"] = df["disease"].replace({
    "Fever": "Fever Syndrome",
    "Only Fever < 7 days": "Fever Syndrome",
    "Un Differentiate d Fever": "Fever Syndrome",
    "Fever with Rash": "Fever Syndrome",
    "Fever With Rash": "Fever Syndrome",
    "Fever with Altered sensorium": "Fever Syndrome",
    "FUO": "Fever Syndrome"
})

# Animal Bite grouping
df["disease"] = df["disease"].replace({
    "Animal Bite Dog Bite": "Animal Bite",
    "Animal Bite - Dog Bite": "Animal Bite",
    "Animal Bite- Snake Bite": "Animal Bite",
    "Human Rabies": "Rabies"
})

# Suspected diseases
df["disease"] = df["disease"].replace({
    "Suspected Dengue": "Dengue",
    "Suspected Dengue Fever": "Dengue",
    "Suspected HFMD": "HFMD",
    "Suspected Measles": "Measles",
    "Suspected Mumps": "Mumps",
    "Suspected Scrub Typhus": "Scrub Typhus",
    "Suspected Japanese Encephalitis": "Japanese Encephalitis",
    "Suspected Anthrax": "Anthrax"
})

# Mixed infections
mixed_keywords = [
    "&",
    " and "
]

df.loc[
    df["disease"].astype(str).str.contains(
        "|".join(mixed_keywords),
        case=False,
        na=False
    ),
    "disease"
] = "Mixed Infection"

print("After Phase 2:", df["disease"].nunique())

# ==========================================
# SAVE FINAL DATASET
# ==========================================

OUTPUT_FILE = "data/final_disease_normalized_v2.xlsx"

df.to_excel(
    OUTPUT_FILE,
    index=False
)

# ==========================================
# SAVE DISEASE SUMMARY
# ==========================================

summary = (
    df["disease"]
    .value_counts()
    .reset_index()
)

summary.columns = [
    "Disease",
    "Count"
]

summary.to_excel(
    "data/disease_summary_v2.xlsx",
    index=False
)

print("\n===================================")
print("Normalization Complete")
print("Final unique diseases:", df["disease"].nunique())
print("Saved:", OUTPUT_FILE)
print("Saved: data/disease_summary_v2.xlsx")
print("===================================")