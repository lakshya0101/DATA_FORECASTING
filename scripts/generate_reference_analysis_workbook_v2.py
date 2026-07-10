import pandas as pd
from openpyxl import load_workbook



print("Loading dataset...")

df = pd.read_excel("data/state_disease_weather_master.xlsx")
weather_df = pd.read_excel(
    "data/Weather_State_Monthly.xlsx"
)

weather_df.columns = (
    weather_df.columns
    .str.strip()
    .str.lower()
)

df.columns = df.columns.str.strip().str.lower()

month_map = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}

year_map = {
    2023: "Year 1 (2023)",
    2024: "Year 2 (2024)",
    2025: "Year 3 (2025)",
    2026: "Year 4 (2026)",
}

import numpy as np

def safe_corr(series1, series2):
    """
    Safely calculate correlation.
    Returns None when insufficient data exists.
    """

    temp = pd.concat(
        [series1, series2],
        axis=1
    ).dropna()

    # Need at least 2 points
    if len(temp) < 2:
        return None

    # Constant series causes correlation issues
    if temp.iloc[:, 0].nunique() <= 1:
        return None

    if temp.iloc[:, 1].nunique() <= 1:
        return None

    try:
        return temp.iloc[:, 0].corr(
            temp.iloc[:, 1]
        )

    except Exception:
        return None

states = sorted(df["state"].dropna().unique())

output_file = "data/State_Disease_Reference_Analysis_v5.xlsx"

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:

    for state in states:

        print(f"Processing {state}")

        state_df = df[df["state"] == state].copy()

        diseases = sorted(state_df["disease"].dropna().unique())

        sheet_rows = []

        # =====================
        # STATE HEADER
        # =====================

        sheet_rows.append([f"STATE : {state}"])

        sheet_rows.append([])

        # =====================
        # DISEASE SUMMARY
        # =====================

        sheet_rows.append(["Disease Summary"])

        sheet_rows.append(["Disease", "Records"])

        disease_summary = (
            state_df.groupby("disease")
            .size()
            .reset_index(name="Records")
            .sort_values("Records", ascending=False)
        )

        for _, row in disease_summary.iterrows():

            sheet_rows.append([row["disease"], row["Records"]])

        sheet_rows.append([])
        sheet_rows.append([])
        sheet_rows.append([])

        # =====================
        # DISEASE BLOCKS
        # =====================

        for disease in diseases:

            print(f"State={state}, Disease={disease}")

            disease_df = state_df[state_df["disease"] == disease].copy()

            if len(disease_df) == 0:
                continue

            sheet_rows.append([f"DISEASE : {disease}"])

            sheet_rows.append([])

            # years = sorted(disease_df["year"].dropna().unique())

            # all_months = pd.MultiIndex.from_product(
            #     [years, range(1, 13)], names=["year", "month"]
            # )

            # all_months = pd.DataFrame(index=all_months).reset_index()

            # monthly_cases = disease_df.groupby(["year", "month"], as_index=False).agg(
            #     {
            #         "cases": "sum",
            #         "temperature": "mean",
            #         "humidity": "mean",
            #         "rainfall": "mean",
            #     }
            # )

            # monthly = pd.merge(
            #     all_months, monthly_cases, on=["year", "month"], how="left"
            # )

            # monthly["cases"] = monthly["cases"].fillna(0)

            # =====================
            # COMPLETE 36 MONTH WEATHER DATA
            # =====================

            state_weather = weather_df[
                weather_df["state"] == state
            ].copy()

            monthly_cases = (
                disease_df.groupby(
                    ["year", "month"],
                    as_index=False
                )["cases"]
                .sum()
            )

            monthly = state_weather.merge(
                monthly_cases,
                how="left",
                on=["year", "month"]
            )

            monthly["cases"] = (
                monthly["cases"]
                .fillna(0)
                .astype(int)
            )

            monthly["month_name"] = monthly["month"].map(month_map)

            monthly["display_year"] = monthly["year"].map(year_map)

            # monthly = monthly.dropna(
            #     subset=["temperature", "humidity", "rainfall"], how="all"
            # )

            sheet_rows.append(
                [
                    "Year",
                    "Month",
                    "Month_No",
                    "Cases",
                    "Temperature",
                    "Humidity",
                    "Rainfall",
                ]
            )

            for _, row in monthly.iterrows():

                sheet_rows.append(
                    [
                        row["display_year"],
                        row["month_name"],
                        row["month"],
                        row["cases"],
                        row["temperature"],
                        row["humidity"],
                        row["rainfall"],
                    ]
                )

            # =====================
            # CORRELATION ANALYSIS
            # =====================

            corr_df = monthly.copy()

            corr_df = corr_df.dropna(subset=["temperature", "humidity", "rainfall"])

            if len(corr_df) < 4:
                sheet_rows.append([])
                sheet_rows.append(["Insufficient data for correlation analysis"])

                sheet_rows.append([])
                sheet_rows.append([])
                sheet_rows.append([])

                continue

            sheet_rows.append([])
            sheet_rows.append(["Correlation Analysis"])

            sheet_rows.append(["Weather Factor", "Correlation"])

            temp_corr = safe_corr(
                corr_df["cases"],
                corr_df["temperature"]
            )

            humidity_corr = safe_corr(
                corr_df["cases"],
                corr_df["humidity"]
            )

            rainfall_corr = safe_corr(
                corr_df["cases"],
                corr_df["rainfall"]
            )

            sheet_rows.append(
                ["Temperature", round(temp_corr, 4) if pd.notna(temp_corr) else None]
            )

            sheet_rows.append(
                [
                    "Humidity",
                    round(humidity_corr, 4) if pd.notna(humidity_corr) else None,
                ]
            )

            sheet_rows.append(
                [
                    "Rainfall",
                    round(rainfall_corr, 4) if pd.notna(rainfall_corr) else None,
                ]
            )

            # =====================
            # LAG ANALYSIS
            # =====================

            sheet_rows.append([])
            sheet_rows.append(["Lag Analysis"])

            sheet_rows.append(["Weather Factor", "Lag_Months", "Correlation"])

            lag_results = []

            for factor in ["rainfall", "humidity", "temperature"]:

                for lag in [1, 2, 3]:

                    lag_df = corr_df.copy()

                    lag_df[f"{factor}_lag"] = lag_df[factor].shift(lag)

                    corr_value = safe_corr(
                        lag_df["cases"],
                        lag_df[f"{factor}_lag"]
                    )

                    lag_results.append(
                        {
                            "Factor": factor.title(),
                            "Lag": lag,
                            "Correlation": corr_value,
                        }
                    )

                    sheet_rows.append(
                        [
                            factor.title(),
                            lag,
                            round(corr_value, 4) if pd.notna(corr_value) else None,
                        ]
                    )

        #     # =====================
        #     # BEST RELATIONSHIP
        #     # =====================

        #     sheet_rows.append([])
        #     sheet_rows.append([
        #         "Best Relationship"
        #     ])

        #     sheet_rows.append([
        #         "Dominant Factor",
        #         "Best Lag",
        #         "Best Correlation"
        #     ])

        #     lag_df_final = pd.DataFrame(
        #         lag_results
        #     )

        #     lag_df_final = lag_df_final.dropna()

        #     if len(lag_df_final) > 0:

        #         best_row = (
        #             lag_df_final.loc[
        #                 lag_df_final[
        #                     "Correlation"
        #                 ]
        #                 .abs()
        #                 .idxmax()
        #             ]
        #         )

        #         sheet_rows.append([
        #             best_row["Factor"],
        #             best_row["Lag"],
        #             round(
        #                 best_row["Correlation"],
        #                 4
        #             )
        #         ])

        #     sheet_rows.append([])
        #     sheet_rows.append([])
        #     sheet_rows.append([])

        # final_sheet = pd.DataFrame(
        #     sheet_rows
        # )

        # final_sheet.to_excel(
        #     writer,
        #     sheet_name=state[:31],
        #     index=False,
        #     header=False
        # )
        # =====================
        # BEST RELATIONSHIP
        # =====================

        sheet_rows.append([])
        sheet_rows.append(["Best Relationship"])

        sheet_rows.append(["Dominant Factor", "Best Lag", "Best Correlation"])

        lag_df_final = pd.DataFrame(lag_results).dropna().reset_index(drop=True)

        if not lag_df_final.empty:

            best_idx = lag_df_final["Correlation"].abs().idxmax()

            best_row = lag_df_final.loc[best_idx]

            sheet_rows.append(
                [best_row["Factor"], best_row["Lag"], round(best_row["Correlation"], 4)]
            )
        else:

            sheet_rows.append(["No Valid Relationship", None, None])

            sheet_rows.append([])
            sheet_rows.append([])
            sheet_rows.append([])

            # sheet_rows.append([
            #     best_row["Factor"],
            #     best_row["Lag"],
            #     round(
            #         best_row["Correlation"],
            #         4
            #     )
            # ])

    # =====================
    # SAVE STATE SHEET
    # =====================

        final_sheet = pd.DataFrame(sheet_rows)

        final_sheet.to_excel(writer, sheet_name=str(state)[:31], index=False, header=False)

# =====================
# AUTO COLUMN WIDTH
# =====================

wb = load_workbook(output_file)

for ws in wb.worksheets:

    for column in ws.columns:

        max_length = 0

        col_letter = column[0].column_letter

        for cell in column:

            try:

                if cell.value is not None:

                    max_length = max(max_length, len(str(cell.value)))

            except:
                pass

        ws.column_dimensions[col_letter].width = max_length + 3

wb.save(output_file)

print(f"\nSaved: {output_file}")
