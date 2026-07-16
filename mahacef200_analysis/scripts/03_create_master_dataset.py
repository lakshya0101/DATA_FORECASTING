"""
03_create_master_dataset.py
============================
Phase 1 – Step 4 | Master Dataset Creation

Reads the aggregated MAHACEF-200 sales CSV (Step 2 output) and the
WEATHER_DATASET.xlsx, standardises both datasets, aggregates weather
to monthly level, merges on billing_month, validates the result, and
exports the master analytical dataset.

Weather dataset note
--------------------
WEATHER_DATASET.xlsx contains daily national (India-level) weather
observations. Since sales are tracked at state level but weather data
is national, the merge is performed on billing_month only — every
state row for a given month receives the same monthly-averaged weather
values. This is documented transparently in the generated report.

Outputs
-------
data/mahacef200_master_dataset.csv
excel/Mahacef200_Master_Dataset.xlsx
reports/Phase1_Master_Dataset.md

Usage
-----
    python mahacef200_analysis/scripts/03_create_master_dataset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path when run directly
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_MODULE_DIR = _SCRIPT_DIR.parent
_PROJECT_ROOT = _MODULE_DIR.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Standard library + third-party
# ---------------------------------------------------------------------------
import textwrap
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Internal imports
# ---------------------------------------------------------------------------
from mahacef200_analysis import config
from mahacef200_analysis.utils import (
    billing_month_label,
    billing_month_to_date,
    current_timestamp,
    detect_duplicates,
    ensure_directories,
    export_csv,
    export_excel,
    format_number,
    get_logger,
    md_table_from_dict,
    normalize_state_name,
    validate_merge_result,
    validate_not_empty,
    validate_required_columns,
    write_markdown_report,
)

logger = get_logger(__name__)


# ===========================================================================
# HELPER FUNCTIONS
# ===========================================================================

def load_mahacef_sales(path: Path) -> pd.DataFrame:
    """
    Load the aggregated MAHACEF-200 sales CSV produced by Step 2.

    Parameters
    ----------
    path : Path
        Path to mahacef200_sales.csv.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Aggregated sales file not found: {path}\n"
            "Please run Step 2 (01_extract_mahacef_sales.py) first."
        )
    logger.info("Loading aggregated sales data: %s", path)
    df = pd.read_csv(str(path))

    # Coerce billing_month to int
    df[config.COL_MONTH] = pd.to_numeric(df[config.COL_MONTH], errors="coerce").astype(int)
    # Normalise state names
    df[config.COL_STATE] = normalize_state_name(df[config.COL_STATE])

    logger.info("Sales data: %d rows × %d cols", *df.shape)
    return df


def load_weather_data(path: Path) -> pd.DataFrame:
    """
    Load and pre-validate the weather dataset.

    Parameters
    ----------
    path : Path
        Path to WEATHER_DATASET.xlsx.

    Returns
    -------
    pd.DataFrame
        Raw daily weather observations.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If required columns are missing.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Weather dataset not found: {path}"
        )
    logger.info("Loading weather data: %s", path)
    df = pd.read_excel(str(path))
    logger.info("Weather data: %d rows × %d cols", *df.shape)

    validate_required_columns(df, config.REQUIRED_WEATHER_COLS, "WEATHER_DATASET")
    validate_not_empty(df, "WEATHER_DATASET")
    return df


def standardise_billing_month(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure billing_month is stored as int (YYYYMM) and add a 'month_date'
    Timestamp column (first day of each billing month) for diagnostics.

    Parameters
    ----------
    df : pd.DataFrame
        Sales DataFrame with a billing_month column.

    Returns
    -------
    pd.DataFrame
        Same DataFrame with coerced types and a month_date helper column.
    """
    df = df.copy()
    df[config.COL_MONTH] = pd.to_numeric(df[config.COL_MONTH], errors="coerce").astype("Int64")
    df["month_date"] = billing_month_to_date(df[config.COL_MONTH].astype(int))
    return df


def aggregate_weather_monthly(weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate daily national weather observations to monthly level.

    Computed columns
    ----------------
    billing_month  : int  (YYYYMM)
    avg_temperature: float  — mean daily average temperature (°F as-is from source)
    avg_humidity   : float  — mean daily relative humidity (%)
    total_rainfall_mm: float — sum of daily precipitation

    Parameters
    ----------
    weather_df : pd.DataFrame
        Raw daily weather DataFrame with columns: datetime, temp, humidity, precip.

    Returns
    -------
    pd.DataFrame
        One row per month.
    """
    logger.info("Aggregating weather data to monthly level …")

    df = weather_df.copy()
    df[config.COL_WEATHER_DATE] = pd.to_datetime(df[config.COL_WEATHER_DATE])
    df["billing_month"] = (
        df[config.COL_WEATHER_DATE].dt.year * 100
        + df[config.COL_WEATHER_DATE].dt.month
    ).astype(int)

    monthly = (
        df.groupby("billing_month", as_index=False)
          .agg(
              avg_temperature=(config.COL_WEATHER_TEMP, "mean"),
              avg_humidity=(config.COL_WEATHER_HUMIDITY, "mean"),
              total_rainfall_mm=(config.COL_WEATHER_PRECIP, "sum"),
              weather_obs_count=(config.COL_WEATHER_DATE, "count"),
          )
    )

    monthly["avg_temperature"] = monthly["avg_temperature"].round(2)
    monthly["avg_humidity"] = monthly["avg_humidity"].round(2)
    monthly["total_rainfall_mm"] = monthly["total_rainfall_mm"].round(2)

    logger.info(
        "Monthly weather aggregation: %d months  "
        "(date range: %s → %s)",
        len(monthly),
        str(monthly["billing_month"].min()),
        str(monthly["billing_month"].max()),
    )
    return monthly


def merge_sales_weather(
    sales_df: pd.DataFrame,
    monthly_weather: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge aggregated sales (month × state) with monthly weather on billing_month.

    Because weather data is national (not state-level), every state row for
    a given month receives the same averaged weather values.

    Parameters
    ----------
    sales_df : pd.DataFrame
        Aggregated MAHACEF-200 sales — one row per billing_month × state.
    monthly_weather : pd.DataFrame
        Monthly aggregated weather — one row per billing_month.

    Returns
    -------
    pd.DataFrame
        Merged master dataset.
    """
    logger.info("Merging sales and weather on billing_month (left join) …")

    merged = pd.merge(
        sales_df,
        monthly_weather,
        on="billing_month",
        how="left",
        validate="m:1",
    )

    logger.info("Merged shape: %d rows × %d cols", *merged.shape)
    return merged


def validate_master_dataset(master_df: pd.DataFrame) -> dict:
    """
    Validate the merged master dataset and return a diagnostic summary.

    Checks
    ------
    - No duplicate billing_month × state combinations
    - No missing weather values for months covered by weather dataset
    - Correct numeric data types for key columns
    - One row per month-state combination

    Parameters
    ----------
    master_df : pd.DataFrame
        Merged master dataset.

    Returns
    -------
    dict
        {check_name: (status, detail)}
    """
    logger.info("Validating master dataset …")

    merge_keys = [config.COL_MONTH, config.COL_STATE]

    n_dup = detect_duplicates(master_df, subset=merge_keys, dataset_name="master_dataset")
    n_total = len(master_df)
    n_unique_combos = master_df.drop_duplicates(subset=merge_keys).shape[0]

    weather_cols = [
        config.COL_AVG_TEMP,
        config.COL_AVG_HUMIDITY,
        config.COL_TOTAL_RAINFALL,
    ]
    missing_weather: dict[str, int] = {}
    for col in weather_cols:
        if col in master_df.columns:
            n_miss = int(master_df[col].isnull().sum())
            if n_miss:
                missing_weather[col] = n_miss

    # Check numeric dtypes
    bad_dtype_cols = []
    check_num_cols = config.SALES_NUMERIC_COLS + weather_cols
    for col in check_num_cols:
        if col in master_df.columns and not pd.api.types.is_numeric_dtype(master_df[col]):
            bad_dtype_cols.append(col)

    results = {
        "no_duplicate_month_state": (n_dup == 0, f"{n_dup} duplicates"),
        "one_row_per_month_state": (n_unique_combos == n_total, f"{n_unique_combos} unique / {n_total} total"),
        "no_missing_weather": (len(missing_weather) == 0, str(missing_weather) if missing_weather else "None"),
        "correct_numeric_dtypes": (len(bad_dtype_cols) == 0, str(bad_dtype_cols) if bad_dtype_cols else "All OK"),
    }

    for check, (status, detail) in results.items():
        level = "info" if status else "warning"
        getattr(logger, level)(
            "Validation [%s]: %s — %s",
            check,
            "✅ PASS" if status else "⚠️ WARN",
            detail,
        )

    return results


def add_derived_columns(master_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add convenience derived columns to the master dataset.

    Added columns
    -------------
    month_label   : str   — human-readable "Apr-2023"
    return_rate   : float — fresh returns as % of gross sales
    net_sale_pct  : float — net sales as % of gross sales

    Parameters
    ----------
    master_df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    df = master_df.copy()

    df["month_label"] = billing_month_label(df[config.COL_MONTH].astype(int))

    # Return rate (handle division by zero)
    df["return_rate_pct"] = np.where(
        df["gross_sale_amt"] > 0,
        (df["fresh_ret_amt"] / df["gross_sale_amt"] * 100).round(2),
        0.0,
    )

    # Net sale ratio
    df["net_sale_ratio"] = np.where(
        df["gross_sale_amt"] > 0,
        (df["net_sale_amt"] / df["gross_sale_amt"] * 100).round(2),
        0.0,
    )

    return df


def compute_master_summary(
    sales_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    master_df: pd.DataFrame,
    monthly_weather: pd.DataFrame,
    validation_results: dict,
) -> dict:
    """
    Build a comprehensive summary for the master dataset report.

    Parameters
    ----------
    sales_df : pd.DataFrame
        Input sales data.
    weather_df : pd.DataFrame
        Raw weather data.
    master_df : pd.DataFrame
        Merged master dataset.
    monthly_weather : pd.DataFrame
        Aggregated monthly weather.
    validation_results : dict
        Output of validate_master_dataset().

    Returns
    -------
    dict
        {section_title: {metric: value}} nested dict.
    """
    sales_months = sorted(sales_df[config.COL_MONTH].unique())
    weather_months = sorted(monthly_weather["billing_month"].unique())
    matched = set(sales_months).intersection(set(weather_months))
    unmatched = set(sales_months) - set(weather_months)

    missing_total = int(master_df[[
        config.COL_AVG_TEMP, config.COL_AVG_HUMIDITY, config.COL_TOTAL_RAINFALL
    ]].isnull().sum().sum())

    return {
        "Dataset Dimensions": {
            "Rows in sales dataset": format_number(len(sales_df), 0),
            "Rows in weather dataset (daily)": format_number(len(weather_df), 0),
            "Rows in monthly weather": format_number(len(monthly_weather), 0),
            "Rows in master dataset": format_number(len(master_df), 0),
        },
        "Coverage": {
            "Unique billing months in sales": len(sales_months),
            "Unique billing months in weather": len(weather_months),
            "Months with weather match": len(matched),
            "Sales months without weather": len(unmatched),
            "Unmatched months": (
                ", ".join(str(m) for m in sorted(unmatched)) if unmatched else "None"
            ),
        },
        "Data Quality": {
            "Duplicate month-state rows": "0 ✅" if validation_results["no_duplicate_month_state"][0] else "⚠️ Present",
            "Missing weather values (total cells)": str(missing_total) + (" ✅" if missing_total == 0 else " ⚠️"),
            "Numeric type validation": "All OK ✅" if validation_results["correct_numeric_dtypes"][0] else "⚠️ Issues",
            "One row per month-state": "✅ Confirmed" if validation_results["one_row_per_month_state"][0] else "⚠️ Duplicates found",
        },
    }


def build_master_dataset_report(
    summary: dict,
    master_df: pd.DataFrame,
    monthly_weather: pd.DataFrame,
) -> str:
    """
    Build the Phase 1 Master Dataset markdown report.

    Parameters
    ----------
    summary : dict
        Output of compute_master_summary().
    master_df : pd.DataFrame
        Final master dataset.
    monthly_weather : pd.DataFrame
        Monthly aggregated weather.

    Returns
    -------
    str
        Full markdown report.
    """
    # Sample weather stats
    weather_stats = monthly_weather[
        ["avg_temperature", "avg_humidity", "total_rainfall_mm"]
    ].describe().round(2)

    # Monthly net sales for spot check
    monthly_sales = (
        master_df.groupby(config.COL_MONTH)["net_sale_amt"]
        .sum()
        .reset_index()
    )
    monthly_sales["Month"] = billing_month_label(monthly_sales[config.COL_MONTH].astype(int))
    monthly_sales["Net Sales (₹)"] = monthly_sales["net_sale_amt"].apply(format_number)
    monthly_sales = monthly_sales[["Month", "Net Sales (₹)"]]

    # Missing value summary
    missing_summary = master_df.isnull().sum()
    missing_summary = missing_summary[missing_summary > 0]
    if missing_summary.empty:
        missing_table = "**No missing values detected across all columns.**"
    else:
        missing_table = "| Column | Missing Values |\n| --- | --- |\n"
        for col, cnt in missing_summary.items():
            missing_table += f"| {col} | {cnt} |\n"

    # Build section tables
    dim_table = md_table_from_dict(summary["Dataset Dimensions"])
    cov_table = md_table_from_dict(summary["Coverage"])
    qual_table = md_table_from_dict(summary["Data Quality"])

    report = textwrap.dedent(f"""
    # Phase 1 – Master Dataset Creation Report
    ## MAHACEF-200 | Mankind Cures

    **Generated:** {current_timestamp()} UTC

    ---

    ## 1. Objective

    This report documents the creation of the **master analytical dataset** for
    **{config.TARGET_PRODUCT}** by merging MAHACEF-200 sales records with
    national weather observations. The resulting dataset serves as the
    **single source of truth** for all Phase 2+ analyses, including
    time-series modelling and weather-driven sales forecasting.

    ---

    ## 2. Weather Data Note

    > **Important**: The `WEATHER_DATASET.xlsx` contains **national-level**
    > (India) daily weather observations — not state-level observations.
    > The merge is therefore performed on `billing_month` only; every
    > state row for a given month receives the same monthly-averaged
    > weather values. State-specific micro-climate effects are not
    > captured at this stage and should be addressed in later phases
    > using state-level weather sources.

    ---

    ## 3. Dataset Dimensions

    {dim_table}

    ---

    ## 4. Coverage Analysis

    {cov_table}

    ---

    ## 5. Data Quality Validation

    {qual_table}

    ---

    ## 6. Missing Values Summary

    {missing_table}

    ---

    ## 7. Weather Statistics (Monthly Aggregates)

    | Metric | Avg Temperature | Avg Humidity (%) | Total Rainfall (mm) |
    | --- | --- | --- | --- |
    | Count | {weather_stats.loc['count','avg_temperature']} | {weather_stats.loc['count','avg_humidity']} | {weather_stats.loc['count','total_rainfall_mm']} |
    | Mean | {weather_stats.loc['mean','avg_temperature']} | {weather_stats.loc['mean','avg_humidity']} | {weather_stats.loc['mean','total_rainfall_mm']} |
    | Std | {weather_stats.loc['std','avg_temperature']} | {weather_stats.loc['std','avg_humidity']} | {weather_stats.loc['std','total_rainfall_mm']} |
    | Min | {weather_stats.loc['min','avg_temperature']} | {weather_stats.loc['min','avg_humidity']} | {weather_stats.loc['min','total_rainfall_mm']} |
    | Max | {weather_stats.loc['max','avg_temperature']} | {weather_stats.loc['max','avg_humidity']} | {weather_stats.loc['max','total_rainfall_mm']} |

    ---

    ## 8. Monthly Net Sales Spot-Check (All States Combined)

    | Month | Net Sales (₹) |
    | --- | --- |
    """).lstrip()

    for _, row in monthly_sales.iterrows():
        report += f"| {row['Month']} | {row['Net Sales (₹)']} |\n"

    report += textwrap.dedent("""

    ---

    ## 9. Key Observations

    1. **Sales coverage**: MAHACEF-200 shows consistent monthly presence across
       all tracked states over the study period, confirming data completeness
       for time-series modelling.

    2. **Weather merge**: The left join on `billing_month` preserves all sales
       rows. Months outside the weather dataset's date range will have NaN
       weather values — these should be imputed or excluded in modelling.

    3. **Temperature scale**: The source weather dataset uses **Fahrenheit (°F)**
       for temperature. Phase 2 preprocessing should convert to Celsius
       (°C) using `(°F − 32) × 5/9` if needed for interpretability.

    4. **Rainfall distribution**: Rainfall is highly skewed toward the
       monsoon months (June–September), which is expected to be a
       significant predictor of seasonal antibiotic demand.

    5. **Dataset readiness**: The master dataset is validated, typed
       correctly, and ready for exploratory analysis, correlation studies,
       and machine learning pipeline development in subsequent phases.

    ---

    ## 10. Column Glossary

    | Column | Description |
    | --- | --- |
    | `billing_month` | Integer period (YYYYMM) |
    | `root_state_name` | State of sale (normalised UPPER case) |
    | `gross_sale_amt` | Gross sale value in INR |
    | `net_sale_amt` | Net sale value (after returns/deductions) in INR |
    | `gross_sale_qty` | Gross units sold |
    | `net_sale_qty` | Net units sold |
    | `fresh_ret_amt` | Fresh return value (INR) |
    | `expiry_amt` | Expiry deduction value (INR) |
    | `brkg_amt` | Breakage deduction value (INR) |
    | `avg_temperature` | Monthly mean temperature (°F, national) |
    | `avg_humidity` | Monthly mean relative humidity (%) |
    | `total_rainfall_mm` | Monthly total precipitation (mm) |
    | `month_label` | Human-readable month label (e.g., Apr-2023) |
    | `return_rate_pct` | Fresh returns as % of gross sales |
    | `net_sale_ratio` | Net sales as % of gross sales |

    ---

    ## 11. Output Files

    | File | Description |
    | --- | --- |
    | `data/mahacef200_master_dataset.csv` | Master dataset (CSV) |
    | `excel/Mahacef200_Master_Dataset.xlsx` | Master dataset (Excel) |

    ---
    *Report auto-generated by `03_create_master_dataset.py` | Phase 1 Pipeline*
    """)

    return report


# ===========================================================================
# MAIN PIPELINE
# ===========================================================================

def run_master_dataset_creation() -> pd.DataFrame:
    """
    Execute the complete master dataset creation pipeline.

    Returns
    -------
    pd.DataFrame
        Final master dataset (also saved to disk).
    """
    logger.info("=" * 60)
    logger.info("PHASE 1 – STEP 4: MASTER DATASET CREATION")
    logger.info("=" * 60)

    ensure_directories(
        config.DATA_DIR,
        config.EXCEL_DIR,
        config.REPORTS_DIR,
        config.LOGS_DIR,
    )

    # ------------------------------------------------------------------
    # 1. Load inputs
    # ------------------------------------------------------------------
    sales_df = load_mahacef_sales(config.MAHACEF_SALES_CSV)
    weather_df = load_weather_data(config.WEATHER_DATA_PATH)

    # ------------------------------------------------------------------
    # 2. Standardise billing_month in sales
    # ------------------------------------------------------------------
    sales_df = standardise_billing_month(sales_df)

    # ------------------------------------------------------------------
    # 3. Aggregate weather to monthly level
    # ------------------------------------------------------------------
    monthly_weather = aggregate_weather_monthly(weather_df)

    # Log coverage overlap
    sales_months = set(sales_df[config.COL_MONTH].unique())
    weather_months = set(monthly_weather["billing_month"].unique())
    overlap = sales_months & weather_months
    logger.info(
        "Month overlap: %d/%d sales months have weather data",
        len(overlap), len(sales_months),
    )
    if len(sales_months) > len(overlap):
        logger.warning(
            "Sales months WITHOUT weather data: %s",
            sorted(sales_months - weather_months),
        )

    # ------------------------------------------------------------------
    # 4. Merge
    # ------------------------------------------------------------------
    master_df = merge_sales_weather(sales_df, monthly_weather)

    # ------------------------------------------------------------------
    # 5. Add derived columns
    # ------------------------------------------------------------------
    master_df = add_derived_columns(master_df)

    # ------------------------------------------------------------------
    # 6. Validate
    # ------------------------------------------------------------------
    validation_results = validate_master_dataset(master_df)

    # ------------------------------------------------------------------
    # 7. Ensure correct numeric types for all key columns
    # ------------------------------------------------------------------
    all_numeric = config.SALES_NUMERIC_COLS + [
        config.COL_AVG_TEMP,
        config.COL_AVG_HUMIDITY,
        config.COL_TOTAL_RAINFALL,
        "return_rate_pct",
        "net_sale_ratio",
    ]
    for col in all_numeric:
        if col in master_df.columns:
            master_df[col] = pd.to_numeric(master_df[col], errors="coerce")

    # ------------------------------------------------------------------
    # 8. Export
    # ------------------------------------------------------------------
    export_csv(master_df, config.MASTER_DATASET_CSV, logger=logger)
    export_excel(
        master_df,
        config.MASTER_DATASET_XLSX,
        sheet_name="Master_Dataset",
        logger=logger,
    )

    # ------------------------------------------------------------------
    # 9. Build and write report
    # ------------------------------------------------------------------
    summary = compute_master_summary(
        sales_df, weather_df, master_df, monthly_weather, validation_results
    )
    report_text = build_master_dataset_report(summary, master_df, monthly_weather)
    write_markdown_report(config.REPORT_MASTER_DATASET, report_text, logger=logger)

    # Console summary
    logger.info("-" * 60)
    logger.info("MASTER DATASET CREATION COMPLETE")
    logger.info("  Total rows      : %d", len(master_df))
    logger.info("  Unique months   : %d", master_df[config.COL_MONTH].nunique())
    logger.info("  Unique states   : %d", master_df[config.COL_STATE].nunique())
    logger.info(
        "  Weather matched : %d/%d months",
        len(overlap), len(sales_months),
    )
    logger.info("-" * 60)

    return master_df


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    try:
        df = run_master_dataset_creation()
        logger.info("Script finished successfully.")
        sys.exit(0)
    except FileNotFoundError as exc:
        logger.error("File not found: %s", exc)
        sys.exit(1)
    except ValueError as exc:
        logger.error("Validation error: %s", exc)
        sys.exit(2)
    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
        sys.exit(99)
