"""
01_extract_mahacef_sales.py
===========================
Phase 1 – Step 2 | Product Extraction

Reads the master sales dataset (Sale_Details.xlsx), filters for
MAHACEF-200, validates the result, aggregates monthly sales by
billing_month × root_state_name, and exports structured outputs.

Outputs
-------
data/mahacef200_sales.csv
excel/Mahacef200_Sales.xlsx
reports/Phase1_Product_Extraction.md

Usage
-----
    python -m mahacef200_analysis.scripts.01_extract_mahacef_sales
    # or from project root:
    python mahacef200_analysis/scripts/01_extract_mahacef_sales.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path when script is run directly
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent          # …/scripts/
_MODULE_DIR = _SCRIPT_DIR.parent                        # …/mahacef200_analysis/
_PROJECT_ROOT = _MODULE_DIR.parent                      # …/IDSP_Disease_Project/

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Standard library + third-party imports
# ---------------------------------------------------------------------------
import textwrap
from typing import Tuple

import numpy as np
import pandas as pd

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
    validate_not_empty,
    validate_numeric_columns,
    validate_required_columns,
    write_markdown_report,
)

# ---------------------------------------------------------------------------
# Module logger
# ---------------------------------------------------------------------------
logger = get_logger(__name__)


# ===========================================================================
# HELPER FUNCTIONS
# ===========================================================================

def load_sales_data(path: Path) -> pd.DataFrame:
    """
    Read the raw sales Excel file and return a DataFrame.

    Parameters
    ----------
    path : Path
        Absolute path to Sale_Details.xlsx.

    Returns
    -------
    pd.DataFrame
        Raw sales DataFrame.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at *path*.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Sales dataset not found at: {path}\n"
            "Please ensure Sale_Details.xlsx is in the data/ directory."
        )
    logger.info("Loading sales data from: %s", path)
    df = pd.read_excel(str(path))
    logger.info("Loaded %d rows × %d columns.", *df.shape)
    return df


def filter_target_product(df: pd.DataFrame, product: str) -> pd.DataFrame:
    """
    Filter rows where item_name matches *product* (case-insensitive, trimmed).

    Parameters
    ----------
    df : pd.DataFrame
        Raw sales DataFrame.
    product : str
        Target product name, e.g. "MAHACEF-200".

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.

    Raises
    ------
    ValueError
        If no rows match the target product.
    """
    logger.info("Filtering for product: '%s'", product)
    mask = df[config.COL_ITEM_NAME].astype(str).str.strip().str.upper() == product.upper()
    filtered = df[mask].copy()
    logger.info("Rows matching '%s': %d", product, len(filtered))

    validate_not_empty(filtered, dataset_name=f"filter[{product}]")
    return filtered


def validate_sales_dataframe(df: pd.DataFrame) -> None:
    """
    Run all structural validations on the filtered sales DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Filtered MAHACEF-200 sales DataFrame.

    Raises
    ------
    ValueError
        On missing required columns or empty DataFrame.
    """
    validate_required_columns(df, config.REQUIRED_SALES_COLS, "MAHACEF-200 Sales")
    validate_numeric_columns(df, config.SALES_NUMERIC_COLS, "MAHACEF-200 Sales")

    # Detect full-row duplicates
    n_dups = detect_duplicates(df, dataset_name="MAHACEF-200 Sales")
    if n_dups > 0:
        logger.warning("Dropping %d full duplicate rows.", n_dups)
        df.drop_duplicates(inplace=True)


def aggregate_monthly_sales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate numeric sales columns by billing_month × root_state_name.

    Parameters
    ----------
    df : pd.DataFrame
        Filtered (and de-duplicated) MAHACEF-200 sales DataFrame.

    Returns
    -------
    pd.DataFrame
        Aggregated DataFrame sorted by billing_month, then root_state_name.
    """
    logger.info("Aggregating monthly sales by billing_month × root_state_name …")

    group_keys = [config.COL_MONTH, config.COL_STATE]
    agg_dict = {col: "sum" for col in config.SALES_NUMERIC_COLS}

    aggregated = (
        df.groupby(group_keys, as_index=False)
          .agg(agg_dict)
          .sort_values(group_keys)
          .reset_index(drop=True)
    )

    # Add human-readable month label
    aggregated["month_label"] = billing_month_label(aggregated[config.COL_MONTH])

    logger.info(
        "Aggregated shape: %d rows × %d cols  "
        "(unique months=%d, unique states=%d)",
        *aggregated.shape,
        aggregated[config.COL_MONTH].nunique(),
        aggregated[config.COL_STATE].nunique(),
    )
    return aggregated


def compute_summary_metrics(df: pd.DataFrame) -> dict:
    """
    Compute high-level summary metrics from the aggregated sales DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Aggregated MAHACEF-200 sales DataFrame.

    Returns
    -------
    dict
        Dictionary of metric name → formatted value.
    """
    total_records = len(df)
    unique_months = sorted(df[config.COL_MONTH].unique())
    n_months = len(unique_months)
    n_states = df[config.COL_STATE].nunique()

    month_labels = billing_month_label(pd.Series(unique_months))
    month_range = f"{month_labels.iloc[0]} → {month_labels.iloc[-1]}"

    total_gross_sales = df["gross_sale_amt"].sum()
    total_net_sales = df["net_sale_amt"].sum()
    total_gross_qty = df["gross_sale_qty"].sum()
    total_net_qty = df["net_sale_qty"].sum()
    total_fresh_returns = df["fresh_ret_amt"].sum()
    total_expiry = df["expiry_amt"].sum()

    return {
        "Total Records (month-state rows)": format_number(total_records, 0),
        "Unique Billing Months": str(n_months),
        "Month Range": month_range,
        "Unique States": str(n_states),
        "Total Gross Sales (₹)": format_number(total_gross_sales),
        "Total Net Sales (₹)": format_number(total_net_sales),
        "Total Gross Quantity (units)": format_number(total_gross_qty),
        "Total Net Quantity (units)": format_number(total_net_qty),
        "Total Fresh Returns (₹)": format_number(total_fresh_returns),
        "Total Expiry Deductions (₹)": format_number(total_expiry),
    }


def build_extraction_report(
    summary: dict,
    df: pd.DataFrame,
) -> str:
    """
    Build a markdown report for Phase 1 Product Extraction.

    Parameters
    ----------
    summary : dict
        Summary metrics from compute_summary_metrics().
    df : pd.DataFrame
        Aggregated MAHACEF-200 sales DataFrame.

    Returns
    -------
    str
        Full markdown report text.
    """
    # Top 5 states by net sales (for the report)
    top_states = (
        df.groupby(config.COL_STATE)["net_sale_amt"]
          .sum()
          .sort_values(ascending=False)
          .head(5)
          .reset_index()
    )
    top_states.columns = ["State", "Net Sales (₹)"]
    top_states["Net Sales (₹)"] = top_states["Net Sales (₹)"].apply(format_number)

    # Monthly net sales trend
    monthly_trend = (
        df.groupby(config.COL_MONTH)["net_sale_amt"]
          .sum()
          .reset_index()
    )
    monthly_trend["Month"] = billing_month_label(monthly_trend[config.COL_MONTH])
    monthly_trend["Net Sales (₹)"] = monthly_trend["net_sale_amt"].apply(format_number)
    monthly_trend = monthly_trend[["Month", "Net Sales (₹)"]]

    report = textwrap.dedent(f"""
    # Phase 1 – Product Extraction Report
    ## MAHACEF-200 | Mankind Cures

    **Generated:** {current_timestamp()} UTC

    ---

    ## 1. Objective

    This report documents the extraction and initial validation of sales data for
    **{config.TARGET_PRODUCT}** from the master `Sale_Details.xlsx` dataset as
    part of Phase 1 (Project Foundation & Data Preparation) of the
    *Weather-Driven Pharmaceutical Sales Forecasting* study.

    ---

    ## 2. Data Source

    | Attribute | Value |
    | --- | --- |
    | Source File | `Sale_Details.xlsx` |
    | Filter Criterion | `item_name == '{config.TARGET_PRODUCT}'` (case-insensitive) |
    | Aggregation Keys | `billing_month`, `root_state_name` |

    ---

    ## 3. Summary Metrics

    {md_table_from_dict(summary)}

    ---

    ## 4. Top 5 States by Net Sales

    | State | Net Sales (₹) |
    | --- | --- |
    """).lstrip()

    for _, row in top_states.iterrows():
        report += f"| {row['State']} | {row['Net Sales (₹)']} |\n"

    report += textwrap.dedent(f"""

    ---

    ## 5. Monthly Net Sales Trend

    | Month | Net Sales (₹) |
    | --- | --- |
    """)

    for _, row in monthly_trend.iterrows():
        report += f"| {row['Month']} | {row['Net Sales (₹)']} |\n"

    report += textwrap.dedent("""

    ---

    ## 6. Validation Results

    | Check | Status |
    | --- | --- |
    | Product exists in dataset | ✅ PASS |
    | Required columns present | ✅ PASS |
    | No unresolvable duplicate rows | ✅ PASS |
    | Numeric column integrity | ✅ PASS |

    ---

    ## 7. Output Files

    | File | Description |
    | --- | --- |
    | `data/mahacef200_sales.csv` | Aggregated monthly-state sales (CSV) |
    | `excel/Mahacef200_Sales.xlsx` | Aggregated monthly-state sales (Excel) |

    ---

    ## 8. Next Steps

    - **Step 3**: State Contribution Analysis — rank and classify states by
      region and contribution percentage.
    - **Step 4**: Master Dataset Creation — merge sales data with
      weather observations for modelling.

    ---
    *Report auto-generated by `01_extract_mahacef_sales.py` | Phase 1 Pipeline*
    """)

    return report


# ===========================================================================
# MAIN PIPELINE
# ===========================================================================

def run_extraction() -> pd.DataFrame:
    """
    Execute the complete product extraction pipeline.

    Returns
    -------
    pd.DataFrame
        Aggregated MAHACEF-200 sales DataFrame (also saved to disk).
    """
    logger.info("=" * 60)
    logger.info("PHASE 1 – STEP 2: PRODUCT EXTRACTION")
    logger.info("Target product: %s", config.TARGET_PRODUCT)
    logger.info("=" * 60)

    # Ensure all output directories exist
    ensure_directories(
        config.DATA_DIR,
        config.EXCEL_DIR,
        config.REPORTS_DIR,
        config.LOGS_DIR,
    )

    # ------------------------------------------------------------------
    # 1. Load raw sales data
    # ------------------------------------------------------------------
    raw_df = load_sales_data(config.SALES_DATA_PATH)

    # ------------------------------------------------------------------
    # 2. Filter for MAHACEF-200
    # ------------------------------------------------------------------
    filtered_df = filter_target_product(raw_df, config.TARGET_PRODUCT)

    # ------------------------------------------------------------------
    # 3. Validate filtered data
    # ------------------------------------------------------------------
    validate_sales_dataframe(filtered_df)

    # ------------------------------------------------------------------
    # 4. Coerce numeric columns
    # ------------------------------------------------------------------
    for col in config.SALES_NUMERIC_COLS:
        filtered_df[col] = pd.to_numeric(filtered_df[col], errors="coerce").fillna(0.0)

    # Normalise state names
    filtered_df[config.COL_STATE] = normalize_state_name(filtered_df[config.COL_STATE])

    # ------------------------------------------------------------------
    # 5. Aggregate monthly sales
    # ------------------------------------------------------------------
    aggregated_df = aggregate_monthly_sales(filtered_df)

    # ------------------------------------------------------------------
    # 6. Export CSV
    # ------------------------------------------------------------------
    export_csv(aggregated_df, config.MAHACEF_SALES_CSV, logger=logger)

    # ------------------------------------------------------------------
    # 7. Export Excel
    # ------------------------------------------------------------------
    export_excel(
        aggregated_df,
        config.MAHACEF_SALES_XLSX,
        sheet_name="Mahacef200_Sales",
        logger=logger,
    )

    # ------------------------------------------------------------------
    # 8. Compute summary and write report
    # ------------------------------------------------------------------
    summary = compute_summary_metrics(aggregated_df)
    report_text = build_extraction_report(summary, aggregated_df)
    write_markdown_report(config.REPORT_PRODUCT_EXTRACTION, report_text, logger=logger)

    # Console summary
    logger.info("-" * 60)
    logger.info("EXTRACTION COMPLETE")
    for k, v in summary.items():
        logger.info("  %-40s %s", k, v)
    logger.info("-" * 60)

    return aggregated_df


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    try:
        df = run_extraction()
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
