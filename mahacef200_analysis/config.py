"""
config.py
=========
Central configuration module for the MAHACEF-200 Pharmaceutical Sales
Forecasting project (Phase 1).

All paths, constants, and logging settings are defined here so that every
downstream script remains configuration-driven and environment-agnostic.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# PROJECT ROOT
# ---------------------------------------------------------------------------
# This file lives at  mahacef200_analysis/config.py.
# The project root is one level up.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# MAHACEF-200 MODULE ROOT
# ---------------------------------------------------------------------------
MODULE_ROOT: Path = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# TARGET PRODUCT
# ---------------------------------------------------------------------------
TARGET_PRODUCT: str = "MAHACEF-200"

# ---------------------------------------------------------------------------
# INPUT DATASET PATHS
# ---------------------------------------------------------------------------
SALES_DATA_PATH: Path = PROJECT_ROOT / "data" / "Sale_Details.xlsx"
WEATHER_DATA_PATH: Path = PROJECT_ROOT / "data" / "WEATHER_DATASET.xlsx"

# ---------------------------------------------------------------------------
# OUTPUT DIRECTORY PATHS
# ---------------------------------------------------------------------------
DATA_DIR: Path = MODULE_ROOT / "data"
EXCEL_DIR: Path = MODULE_ROOT / "excel"
GRAPHS_DIR: Path = MODULE_ROOT / "graphs"
REPORTS_DIR: Path = MODULE_ROOT / "reports"
LOGS_DIR: Path = MODULE_ROOT / "logs"

# ---------------------------------------------------------------------------
# OUTPUT FILE PATHS
# ---------------------------------------------------------------------------
# Step 2 – Product Extraction
MAHACEF_SALES_CSV: Path = DATA_DIR / "mahacef200_sales.csv"
MAHACEF_SALES_XLSX: Path = EXCEL_DIR / "Mahacef200_Sales.xlsx"
REPORT_PRODUCT_EXTRACTION: Path = REPORTS_DIR / "Phase1_Product_Extraction.md"

# Step 3 – State Contribution Analysis
STATEWISE_SALES_CSV: Path = DATA_DIR / "mahacef200_statewise_sales.csv"
STATEWISE_SALES_XLSX: Path = EXCEL_DIR / "Mahacef200_Statewise_Sales.xlsx"
REPORT_STATE_CONTRIBUTION: Path = REPORTS_DIR / "Phase1_State_Contribution.md"

GRAPH_STATEWISE_NET_SALES: Path = GRAPHS_DIR / "01_statewise_net_sales.png"
GRAPH_TOP10_STATES: Path = GRAPHS_DIR / "02_top10_states.png"
GRAPH_REGIONAL_DIST: Path = GRAPHS_DIR / "03_regional_sales_distribution.png"

# Step 4 – Master Dataset Creation
MASTER_DATASET_CSV: Path = DATA_DIR / "mahacef200_master_dataset.csv"
MASTER_DATASET_XLSX: Path = EXCEL_DIR / "Mahacef200_Master_Dataset.xlsx"
REPORT_MASTER_DATASET: Path = REPORTS_DIR / "Phase1_Master_Dataset.md"

# ---------------------------------------------------------------------------
# PHASE 1.5 – WEATHER QUALITY & IMPUTATION
# ---------------------------------------------------------------------------
STATE_TIMESERIES_DIR: Path = DATA_DIR / "state_timeseries"
MASTER_CLEAN_CSV: Path = DATA_DIR / "mahacef200_master_dataset_clean.csv"
MASTER_CLEAN_XLSX: Path = EXCEL_DIR / "Mahacef200_Master_Dataset_Clean.xlsx"
REPORT_WEATHER_QUALITY: Path = REPORTS_DIR / "Phase1_5_Weather_Quality.md"
GRAPH_WEATHER_COVERAGE: Path = GRAPHS_DIR / "04_weather_coverage_heatmap.png"
GRAPH_IMPUTATION_COMPARISON: Path = GRAPHS_DIR / "05_imputation_comparison.png"
GRAPH_WEATHER_SEASONAL: Path = GRAPHS_DIR / "06_weather_seasonal_patterns.png"

# ---------------------------------------------------------------------------
# VERSIONING & METADATA
# ---------------------------------------------------------------------------
PROJECT_VERSION: str = "1.0.0"
CLEAN_DATASET_NAME: str = "mahacef200_master_dataset_clean.csv"

# ---------------------------------------------------------------------------
# PHASE 2 – SALES TREND ANALYSIS
# ---------------------------------------------------------------------------
PHASE2_GRAPHS_DIR: Path = GRAPHS_DIR / "phase2_sales"
PHASE2_MONTHLY_SALES_CSV: Path = DATA_DIR / "phase2_monthly_sales.csv"
PHASE2_SALES_XLSX: Path = EXCEL_DIR / "Phase2_Sales_Trend.xlsx"
REPORT_SALES_TREND: Path = REPORTS_DIR / "Phase2_Sales_Trend.md"

# ---------------------------------------------------------------------------
# PHASE 3 – WEATHER TREND ANALYSIS
# ---------------------------------------------------------------------------
PHASE3_GRAPHS_DIR: Path = GRAPHS_DIR / "phase3_weather"
PHASE3_WEATHER_MONTHLY_CSV: Path = DATA_DIR / "phase3_weather_monthly.csv"
PHASE3_WEATHER_XLSX: Path = EXCEL_DIR / "Phase3_Weather_Trend.xlsx"
REPORT_WEATHER_TREND: Path = REPORTS_DIR / "Phase3_Weather_Trend.md"

# Monsoon calendar months (Jun–Sep, India)
MONSOON_MONTHS: list[int] = [6, 7, 8, 9]

# Anomaly detection z-score threshold
ANOMALY_ZSCORE_THRESHOLD: float = 2.0

# ---------------------------------------------------------------------------
# PHASE 4 – WEATHER vs SALES COMPARISON
# ---------------------------------------------------------------------------
PHASE4_GRAPHS_DIR: Path = GRAPHS_DIR / "phase4_weather_vs_sales"
PHASE4_COMPARISON_CSV: Path = DATA_DIR / "phase4_weather_vs_sales.csv"
PHASE4_COMPARISON_XLSX: Path = EXCEL_DIR / "Phase4_Weather_vs_Sales.xlsx"
REPORT_WEATHER_VS_SALES: Path = REPORTS_DIR / "Phase4_Weather_vs_Sales.md"

# Rolling window for correlation preview (Phase 4 teaser)
ROLL_CORR_WINDOW: int = 6

# Number of top states shown in state-level panels
TOP_STATES_DISPLAY: int = 6

# ---------------------------------------------------------------------------
# PHASE 5 – CORRELATION ANALYSIS
# ---------------------------------------------------------------------------
PHASE5_GRAPHS_DIR: Path = GRAPHS_DIR / "phase5_correlation"
PHASE5_CORRELATION_CSV: Path = DATA_DIR / "phase5_correlation_results.csv"
PHASE5_CORRELATION_XLSX: Path = EXCEL_DIR / "Phase5_Correlation.xlsx"
REPORT_CORRELATION: Path = REPORTS_DIR / "Phase5_Correlation.md"

# Significance annotation thresholds
SIG_LEVELS: list[tuple[float, str]] = [
    (0.001, "***"),
    (0.01,  "**"),
    (0.05,  "*"),
    (1.0,   "ns"),
]

# ---------------------------------------------------------------------------
# PHASE 6 – STATISTICAL TESTING & DISTRIBUTION ANALYSIS
# ---------------------------------------------------------------------------
PHASE6_GRAPHS_DIR: Path = GRAPHS_DIR / "phase6_statistical_testing"
PHASE6_RESULTS_CSV: Path = DATA_DIR / "phase6_statistical_tests.csv"
PHASE6_RESULTS_XLSX: Path = EXCEL_DIR / "Phase6_Statistical_Testing.xlsx"
REPORT_STATISTICAL_TESTING: Path = REPORTS_DIR / "Phase6_Statistical_Testing.md"

# Season groupings (calendar month numbers)
SEASONS: dict = {
    "Winter":       [12, 1, 2],
    "Pre-Monsoon":  [3, 4, 5],
    "Monsoon":      [6, 7, 8, 9],
    "Post-Monsoon": [10, 11],
}
SEASON_COLOURS: dict = {
    "Winter":       "#1565C0",
    "Pre-Monsoon":  "#F57F17",
    "Monsoon":      "#1B5E20",
    "Post-Monsoon": "#6A1B9A",
}

# Cohen's d magnitude thresholds  (Cohen 1988)
COHEN_D_THRESHOLDS: list[tuple[float, str]] = [
    (0.2, "Negligible"), (0.5, "Small"), (0.8, "Medium"), (float("inf"), "Large")
]
# η² thresholds
ETA_SQ_THRESHOLDS: list[tuple[float, str]] = [
    (0.01, "Negligible"), (0.06, "Small"), (0.14, "Medium"), (float("inf"), "Large")
]

# ---------------------------------------------------------------------------
# PHASE 7 – REGRESSION ANALYSIS
# ---------------------------------------------------------------------------
PHASE7_GRAPHS_DIR: Path = GRAPHS_DIR / "phase7_regression"
PHASE7_RESULTS_CSV: Path = DATA_DIR / "phase7_regression_results.csv"
PHASE7_RESULTS_XLSX: Path = EXCEL_DIR / "Phase7_Regression.xlsx"
REPORT_REGRESSION: Path = REPORTS_DIR / "Phase7_Regression.md"

# Optimal lag structure derived from Phase 5 correlation analysis
LAG_RAINFALL:    int = 1   # strongest: r=+0.717 ***
LAG_HUMIDITY:    int = 1   # strongest: r=+0.437 **
LAG_TEMPERATURE: int = 3   # strongest: r=+0.391 *

# Diagnostics thresholds
VIF_THRESHOLD:    float = 10.0          # multicollinearity concern
VIF_WARNING:      float = 5.0           # VIF moderate warning
COOKS_D_MULT:     float = 4.0           # Cook's D threshold = COOKS_D_MULT / n
DW_LOWER:         float = 1.5           # Durbin-Watson: below = positive autocorr
DW_UPPER:         float = 2.5           # Durbin-Watson: above = negative autocorr

# ---------------------------------------------------------------------------
# PHASE 8 – MACHINE LEARNING
# ---------------------------------------------------------------------------
PHASE8_GRAPHS_DIR: Path = GRAPHS_DIR / "phase8_machine_learning"
PHASE8_RESULTS_CSV: Path = DATA_DIR / "phase8_ml_results.csv"
PHASE8_RESULTS_XLSX: Path = EXCEL_DIR / "Phase8_MachineLearning.xlsx"
REPORT_ML: Path = REPORTS_DIR / "Phase8_MachineLearning.md"

# Cross-validation strategy
CV_N_SPLITS:      int = 5   # outer TimeSeriesSplit for final evaluation
CV_N_SPLITS_TUNE: int = 3   # inner TimeSeriesSplit for hyperparameter tuning

# Phase 7 regression baseline (Model 3, n=36)
BASELINE_R2:      float = 0.6912
BASELINE_ADJ_R2:  float = 0.6140
BASELINE_RMSE_M:  float = 7.8914   # ₹M
BASELINE_LABEL:   str   = "OLS Regression (Model 3)"

# High-sensitivity states identified in Phase 5 (Bonferroni-significant)
REGIONAL_STATES: list[str] = ["maharashtra", "goa"]

# ---------------------------------------------------------------------------
# PHASE 9 – FORECASTING
# ---------------------------------------------------------------------------
PHASE9_GRAPHS_DIR: Path = GRAPHS_DIR / "phase9_forecasting"
PHASE9_RESULTS_CSV: Path = DATA_DIR / "phase9_forecasts.csv"
PHASE9_RESULTS_XLSX: Path = EXCEL_DIR / "Phase9_Forecasts.xlsx"
REPORT_FORECAST: Path = REPORTS_DIR / "Phase9_Forecasting.md"

FORECAST_HORIZON: int = 6   # 6 months forward forecast (Jul–Dec 2026)
TEST_HORIZON:     int = 6   # 6 months test set (Jan–Jun 2026)








# ---------------------------------------------------------------------------
# ANALYSIS-WIDE CONSTANTS (shared across Phases 2–9)
# ---------------------------------------------------------------------------
# Rolling window sizes (months)
ROLLING_WINDOWS: list[int] = [3, 6]

# STL decomposition seasonal period (12 = annual)
STL_PERIOD: int = 12

# Spike detection: flag months > mean + SPIKE_STD_FACTOR × std
SPIKE_STD_FACTOR: float = 1.5

# Statistical significance level
ALPHA: float = 0.05

# Lag range for correlation and lag analysis
LAG_RANGE: list[int] = [0, 1, 2, 3]

# Top-N states for focused analyses
TOP_N_STATES_ANALYSIS: int = 5

# Forecast horizon (months)
FORECAST_HORIZON: int = 6

# Train/test split ratio for ML
ML_TRAIN_RATIO: float = 0.80

# Cross-validation folds
CV_FOLDS: int = 5

# ---------------------------------------------------------------------------
# SALES DATASET COLUMN CONSTANTS
# ---------------------------------------------------------------------------
COL_ITEM_NAME: str = "item_name"
COL_STATE: str = "root_state_name"
COL_MONTH: str = "billing_month"
COL_MATNR: str = "matnr"

SALES_NUMERIC_COLS: list[str] = [
    "gross_sale_amt",
    "gross_sale_qty",
    "fresh_ret_amt",
    "fresh_ret_qty",
    "expiry_amt",
    "expiry_qty",
    "brkg_amt",
    "brkg_qty",
    "net_sale_amt",
    "net_sale_qty",
]

REQUIRED_SALES_COLS: list[str] = [COL_ITEM_NAME, COL_STATE, COL_MONTH] + SALES_NUMERIC_COLS

# ---------------------------------------------------------------------------
# WEATHER DATASET COLUMN CONSTANTS
# ---------------------------------------------------------------------------
COL_WEATHER_NAME: str = "name"
COL_WEATHER_DATE: str = "datetime"
COL_WEATHER_TEMP: str = "temp"
COL_WEATHER_HUMIDITY: str = "humidity"
COL_WEATHER_PRECIP: str = "precip"

REQUIRED_WEATHER_COLS: list[str] = [
    COL_WEATHER_NAME,
    COL_WEATHER_DATE,
    COL_WEATHER_TEMP,
    COL_WEATHER_HUMIDITY,
    COL_WEATHER_PRECIP,
]

# Aggregated weather column names (used in master dataset)
COL_AVG_TEMP: str = "avg_temperature"
COL_AVG_HUMIDITY: str = "avg_humidity"
COL_TOTAL_RAINFALL: str = "total_rainfall_mm"

# Phase 1.5 imputation method identifiers
IMPUTE_METHOD_CLIMATOLOGY: str = "monthly_climatology"
IMPUTE_METHOD_INTERPOLATION: str = "linear_interpolation"
IMPUTE_METHOD_FORWARD_FILL: str = "forward_fill"
IMPUTE_METHOD_BACKWARD_FILL: str = "backward_fill"

# Imputation strategy per weather variable
# monthly_climatology = use mean of same calendar month from available data
IMPUTATION_STRATEGY: dict[str, str] = {
    COL_AVG_TEMP: IMPUTE_METHOD_CLIMATOLOGY,
    COL_AVG_HUMIDITY: IMPUTE_METHOD_CLIMATOLOGY,
    COL_TOTAL_RAINFALL: IMPUTE_METHOD_CLIMATOLOGY,
}

# Temperature conversion flag (source is °F, set True to add Celsius column)
ADD_TEMP_CELSIUS: bool = True
COL_AVG_TEMP_C: str = "avg_temperature_c"

# ---------------------------------------------------------------------------
# STATE → REGION MAPPING
# ---------------------------------------------------------------------------
STATE_REGION_MAP: dict[str, str] = {
    # North
    "DELHI": "North",
    "HARYANA": "North",
    "HIMACHAL PRADESH": "North",
    "JAMMU": "North",
    "KASHMIR": "North",
    "PUNJAB": "North",
    "RAJASTHAN": "North",
    "U.P.": "North",
    "UTTARAKHAND": "North",
    # South
    "A.P.": "South",
    "KARNATAKA": "South",
    "KERALA": "South",
    "TAMIL NADU": "South",
    "TELANGANA": "South",
    "GOA": "South",
    # East
    "BIHAR": "East",
    "JHARKHAND": "East",
    "ORISSA": "East",
    "WEST BENGAL": "East",
    # West
    "GUJARAT": "West",
    "MAHARASHTRA": "West",
    # Central
    "CHATTISGARH": "Central",
    "M.P.": "Central",
    # North-East
    "ASSAM": "North-East",
    "MANIPUR": "North-East",
    "MEGHALAYA": "North-East",
    "MIZORAM": "North-East",
    "NAGALAND": "North-East",
    "SIKKIM": "North-East",
    "TRIPURA": "North-East",
    "ARUNACHAL PRADESH": "North-East",
}

# ---------------------------------------------------------------------------
# ANALYSIS CONSTANTS
# ---------------------------------------------------------------------------
TOP_N_STATES: int = 10          # Used in bar charts / top-10 analysis
FIGURE_DPI: int = 150           # PNG export resolution
FIGURE_SIZE_WIDE: tuple[int, int] = (16, 8)
FIGURE_SIZE_SQUARE: tuple[int, int] = (10, 10)
FIGURE_SIZE_MEDIUM: tuple[int, int] = (12, 7)

# ---------------------------------------------------------------------------
# LOGGING CONFIGURATION
# ---------------------------------------------------------------------------
LOG_LEVEL: int = logging.DEBUG
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
LOG_FILE: Path = LOGS_DIR / "phase1_pipeline.log"


def configure_logging(name: str = "mahacef200") -> logging.Logger:
    """
    Configure and return a named logger that writes to both
    the console (INFO level) and a rotating log file (DEBUG level).

    Parameters
    ----------
    name : str
        Logger name — typically the calling script's __name__.

    Returns
    -------
    logging.Logger
        Fully configured logger instance.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        # Avoid adding duplicate handlers on repeated imports
        return logger

    logger.setLevel(LOG_LEVEL)

    # Console handler – INFO and above
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))

    # File handler – DEBUG and above
    fh = logging.FileHandler(str(LOG_FILE), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger
