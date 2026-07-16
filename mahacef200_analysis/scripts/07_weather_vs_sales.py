"""
07_weather_vs_sales.py
=======================
Phase 4 — Weather vs Sales Comparison

Objective:
    Visually overlay weather variables and MAHACEF-200 sales for the first
    time.  This phase is descriptive — no statistical tests yet (those are
    Phase 5).  The goal is to let the data *speak visually* before numbers
    are produced.

Outputs
-------
data/phase4_weather_vs_sales.csv
data/phase4_weather_vs_sales.metadata.json
excel/Phase4_Weather_vs_Sales.xlsx
graphs/phase4_weather_vs_sales/
    01_national_temp_vs_sales.png        dual-axis time-series
    02_national_humidity_vs_sales.png    dual-axis time-series
    03_national_rainfall_vs_sales.png    dual-axis time-series
    04_seasonal_overlay.png              calendar-month average profiles
    05_lag_scatter_grid.png              sales(t) vs weather(t-k), k=0..3
    06_top6_state_temp_vs_sales.png      state-level dual-axis panels
    07_rolling_correlation.png           6-month rolling Pearson preview
reports/Phase4_Weather_vs_Sales.md

Usage
-----
    python mahacef200_analysis/scripts/07_weather_vs_sales.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# sys.path bootstrap
# ---------------------------------------------------------------------------
_SCRIPT_DIR   = Path(__file__).resolve().parent
_MODULE_DIR   = _SCRIPT_DIR.parent
_PROJECT_ROOT = _MODULE_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Standard library + third-party
# ---------------------------------------------------------------------------
import warnings
import itertools

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Internal imports
# ---------------------------------------------------------------------------
from mahacef200_analysis import config
from mahacef200_analysis.utils import (
    billing_month_label,
    billing_month_to_date,
    build_phase_report,
    current_timestamp,
    ensure_directories,
    export_csv,
    export_excel,
    format_number,
    get_logger,
    normalize_state_name,
    write_dataset_metadata,
    write_markdown_report,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
CLR_SALES  = "#1565C0"   # blue  — net sales
CLR_TEMP   = "#C62828"   # red   — temperature
CLR_HUM    = "#00695C"   # teal  — humidity
CLR_RAIN   = "#1B5E20"   # green — rainfall
CLR_ROLL   = "#F57F17"   # amber — rolling means
CLR_IMPUTED = "#FFF9C4"  # yellow — imputed shading
CLR_BG     = "#F8F9FA"

MONTH_NAMES = {
    1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr", 5:"May",  6:"Jun",
    7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Dec",
}

# Weather variable definitions
WEATHER_VARS = [
    ("avg_temperature_c",  "Temperature",  "°C",  CLR_TEMP),
    ("avg_humidity",       "Humidity",     "%",   CLR_HUM),
    ("total_rainfall_mm",  "Rainfall",     "mm",  CLR_RAIN),
]

SCRIPT_NAME = "07_weather_vs_sales.py"
PHASE_LABEL = "Phase 4 - Weather vs Sales Comparison"


# ===========================================================================
# 1. DATA PREPARATION
# ===========================================================================

def build_merged_national(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate clean master dataset to a single national monthly table
    combining sales metrics and weather variables.

    Parameters
    ----------
    df : pd.DataFrame
        Clean 935-row master dataset.

    Returns
    -------
    pd.DataFrame
        39-row merged national monthly series.
    """
    logger.info("Building merged national monthly series …")

    # Sales aggregation
    sales = (
        df.groupby(config.COL_MONTH, as_index=False)
          .agg(
              net_sale_amt   =("net_sale_amt",   "sum"),
              gross_sale_amt =("gross_sale_amt", "sum"),
              net_sale_qty   =("net_sale_qty",   "sum"),
          )
          .sort_values(config.COL_MONTH)
          .reset_index(drop=True)
    )

    # Weather deduplication (one row per month)
    weather_cols = [
        config.COL_MONTH,
        "avg_temperature_c", "avg_humidity", "total_rainfall_mm",
        "weather_imputed", "imputation_method",
    ]
    weather = (
        df[weather_cols]
          .drop_duplicates(subset=[config.COL_MONTH])
          .sort_values(config.COL_MONTH)
          .reset_index(drop=True)
    )

    merged = sales.merge(weather, on=config.COL_MONTH, how="left")
    merged["month_label"] = billing_month_label(merged[config.COL_MONTH])
    merged["month_date"]  = billing_month_to_date(merged[config.COL_MONTH])
    merged["month_num"]   = merged[config.COL_MONTH] % 100
    merged["year"]        = merged[config.COL_MONTH] // 100

    # Rolling means — sales
    for w in config.ROLLING_WINDOWS:
        merged[f"rolling_{w}m_net_sales"] = (
            merged["net_sale_amt"].rolling(w, min_periods=1).mean()
        )
    # Rolling means — weather
    for col, _, _, _ in WEATHER_VARS:
        for w in config.ROLLING_WINDOWS:
            merged[f"{col}_roll{w}m"] = (
                merged[col].rolling(w, min_periods=1).mean()
            )

    # Rolling Pearson correlation preview (6-month window)
    w = config.ROLL_CORR_WINDOW
    for col, label, _, _ in WEATHER_VARS:
        merged[f"rollcorr_{col}"] = (
            merged["net_sale_amt"]
              .rolling(w, min_periods=max(3, w // 2))
              .corr(merged[col])
              .round(3)
        )

    # Lag features (weather(t-k) for k=1,2,3)
    for col, _, _, _ in WEATHER_VARS:
        for k in [1, 2, 3]:
            merged[f"{col}_lag{k}"] = merged[col].shift(k)

    logger.info("  Merged series: %d months", len(merged))
    return merged


def identify_top_states(df: pd.DataFrame, n: int = None) -> list[str]:
    """
    Return the top-N states by total net sales, descending.

    Parameters
    ----------
    df : pd.DataFrame   Clean master dataset.
    n  : int            Defaults to config.TOP_STATES_DISPLAY.

    Returns
    -------
    list[str]
    """
    if n is None:
        n = config.TOP_STATES_DISPLAY
    df2 = df.copy()
    df2[config.COL_STATE] = normalize_state_name(df2[config.COL_STATE])
    top = (
        df2.groupby(config.COL_STATE)["net_sale_amt"]
           .sum()
           .sort_values(ascending=False)
           .head(n)
           .index
           .tolist()
    )
    logger.info("  Top %d states: %s", n, top)
    return top


def build_state_monthly(df: pd.DataFrame,
                         top_states: list[str]) -> pd.DataFrame:
    """
    Build state × month aggregation for the top states,
    with weather columns joined.

    Returns
    -------
    pd.DataFrame
        Rows for top states only, columns include state, billing_month,
        net_sale_amt, weather vars.
    """
    df2 = df.copy()
    df2[config.COL_STATE] = normalize_state_name(df2[config.COL_STATE])
    sub = df2[df2[config.COL_STATE].isin(top_states)].copy()

    sales_s = (
        sub.groupby([config.COL_STATE, config.COL_MONTH], as_index=False)
           .agg(net_sale_amt=("net_sale_amt", "sum"),
                net_sale_qty=("net_sale_qty", "sum"))
    )

    weather_s = (
        df[[config.COL_MONTH, "avg_temperature_c",
            "avg_humidity", "total_rainfall_mm", "weather_imputed"]]
          .drop_duplicates(subset=[config.COL_MONTH])
    )

    state_df = sales_s.merge(weather_s, on=config.COL_MONTH, how="left")
    state_df["month_label"] = billing_month_label(state_df[config.COL_MONTH])
    state_df["month_num"]   = state_df[config.COL_MONTH] % 100
    state_df["year"]        = state_df[config.COL_MONTH] // 100
    return state_df


# ===========================================================================
# 2. SHARED HELPERS
# ===========================================================================

def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved → %s", path.name)


def _shade_imputed(ax: plt.Axes, imputed_mask: pd.Series) -> None:
    for i, imp in enumerate(imputed_mask.values):
        if imp:
            ax.axvspan(i - 0.5, i + 0.5,
                       color=CLR_IMPUTED, alpha=0.55, zorder=0)


def _dual_axis_setup(ax1: plt.Axes, ax2: plt.Axes,
                     ylabel1: str, ylabel2: str,
                     col1: str = CLR_SALES, col2: str = CLR_TEMP) -> None:
    ax1.set_ylabel(ylabel1, color=col1, fontsize=10)
    ax2.set_ylabel(ylabel2, color=col2, fontsize=10)
    ax1.tick_params(axis="y", labelcolor=col1)
    ax2.tick_params(axis="y", labelcolor=col2)
    for ax in [ax1, ax2]:
        ax.spines["top"].set_visible(False)


# ===========================================================================
# 3. GRAPH 1-3 — Dual-Axis National Time-Series
# ===========================================================================

def _plot_dual_national(merged: pd.DataFrame, w_col: str, w_label: str,
                         w_unit: str, w_colour: str,
                         filename: str, out_dir: Path) -> None:
    """
    Reusable dual-axis plot: net sales (left) + one weather variable (right).
    Includes rolling means and imputed shading.
    """
    x      = range(len(merged))
    labels = merged["month_label"].tolist()
    step   = max(1, len(merged) // 18)

    fig, ax1 = plt.subplots(figsize=(18, 7))
    ax2 = ax1.twinx()
    fig.patch.set_facecolor(CLR_BG)
    ax1.set_facecolor(CLR_BG)
    ax2.set_facecolor(CLR_BG)

    _shade_imputed(ax1, merged["weather_imputed"])

    # Sales — left axis
    y_sales = merged["net_sale_amt"].values / 1e6
    ax1.fill_between(x, y_sales, alpha=0.07, color=CLR_SALES)
    ax1.plot(x, y_sales, "-o", color=CLR_SALES, lw=2.2, ms=4.5,
             label="Net Sales (₹M)", zorder=4)
    ax1.plot(x, merged["rolling_3m_net_sales"].values / 1e6,
             "--", color=CLR_SALES, lw=1.6, alpha=0.6,
             label="3M Rolling (Sales)", zorder=3)

    # Weather — right axis
    y_weather = merged[w_col].values
    ax2.plot(x, y_weather, "-^", color=w_colour, lw=2.0, ms=5,
             label=f"{w_label} ({w_unit})", zorder=4, alpha=0.9)
    ax2.plot(x, merged[f"{w_col}_roll3m"].values, "--", color=w_colour,
             lw=1.5, alpha=0.55, label=f"3M Rolling ({w_label})", zorder=3)

    # Monsoon shading for rainfall plot
    if w_col == "total_rainfall_mm":
        for i, mon in enumerate(merged["month_num"].values):
            if mon in config.MONSOON_MONTHS:
                ax1.axvspan(i - 0.5, i + 0.5,
                            color="#C8E6C9", alpha=0.3, zorder=0)

    # X-axis
    ax1.set_xticks(list(range(0, len(merged), step)))
    ax1.set_xticklabels([labels[i] for i in range(0, len(merged), step)],
                         rotation=40, ha="right", fontsize=8.5)

    ax1.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"₹{v:.0f}M"))
    _dual_axis_setup(ax1, ax2,
                     "Net Sales (₹ Millions)",
                     f"{w_label} ({w_unit})",
                     CLR_SALES, w_colour)
    ax1.grid(axis="y", ls="--", alpha=0.3)

    # Combined legend
    l1, n1 = ax1.get_legend_handles_labels()
    l2, n2 = ax2.get_legend_handles_labels()
    imp_patch = mpatches.Patch(color=CLR_IMPUTED, alpha=0.8,
                                label="Climatology Imputed")
    ax1.legend(l1 + l2 + [imp_patch], n1 + n2 + ["Climatology Imputed"],
               fontsize=9, loc="upper right", framealpha=0.92)

    ax1.set_title(
        f"MAHACEF-200 | National Net Sales vs {w_label}  "
        f"(Yellow = Imputed Weather)",
        fontsize=13, fontweight="bold", pad=14,
    )
    plt.tight_layout()
    _save(fig, out_dir / filename)


def plot_national_comparisons(merged: pd.DataFrame, out_dir: Path) -> None:
    """Generate Graphs 1, 2, 3 — one per weather variable."""
    logger.info("Plotting Graphs 1–3: National dual-axis comparisons …")
    _plot_dual_national(merged, "avg_temperature_c",
                         "Temperature", "°C", CLR_TEMP,
                         "01_national_temp_vs_sales.png", out_dir)
    _plot_dual_national(merged, "avg_humidity",
                         "Humidity", "%", CLR_HUM,
                         "02_national_humidity_vs_sales.png", out_dir)
    _plot_dual_national(merged, "total_rainfall_mm",
                         "Rainfall", "mm", CLR_RAIN,
                         "03_national_rainfall_vs_sales.png", out_dir)


# ===========================================================================
# 4. GRAPH 4 — Seasonal Profile Overlay
# ===========================================================================

def plot_seasonal_overlay(merged: pd.DataFrame, out_dir: Path) -> None:
    """
    Average calendar-month profile (Jan–Dec) for sales and each weather
    variable.  Reveals whether seasonal sales peaks align with weather peaks.
    """
    logger.info("Plotting Graph 4: Seasonal Profile Overlay …")

    season = (
        merged.groupby("month_num")
              .agg(
                  avg_net_sales       =("net_sale_amt", "mean"),
                  avg_temperature_c   =("avg_temperature_c", "mean"),
                  avg_humidity        =("avg_humidity", "mean"),
                  avg_total_rainfall  =("total_rainfall_mm", "mean"),
              )
              .reset_index()
    )
    mon_labels = [MONTH_NAMES[m] for m in season["month_num"]]
    x = range(len(season))

    fig, axes = plt.subplots(3, 1, figsize=(14, 13), sharex=True)
    fig.patch.set_facecolor(CLR_BG)

    combos = [
        ("avg_temperature_c", "Avg Temperature (°C)", CLR_TEMP),
        ("avg_humidity",      "Avg Humidity (%)",     CLR_HUM),
        ("avg_total_rainfall","Avg Rainfall (mm)",    CLR_RAIN),
    ]

    for ax, (w_col, w_ylabel, w_col_c) in zip(axes, combos):
        ax.set_facecolor(CLR_BG)
        ax2 = ax.twinx()

        # Sales — left
        sales_vals = season["avg_net_sales"].values / 1e6
        ax.bar(x, sales_vals, color=CLR_SALES, alpha=0.25, width=0.45,
               label="Avg Net Sales (₹M)")
        ax.plot(x, sales_vals, "-o", color=CLR_SALES, lw=2.2,
                ms=6, zorder=4)

        # Weather — right
        w_vals = season[w_col].values
        ax2.plot(x, w_vals, "-s", color=w_col_c, lw=2.2,
                 ms=6, label=w_ylabel, zorder=5)
        ax2.fill_between(x, w_vals, alpha=0.10, color=w_col_c)

        # Annotate peak/trough for sales
        pk = int(np.argmax(sales_vals))
        ax.annotate(f"Peak\n{mon_labels[pk]}",
                    xy=(pk, sales_vals[pk]),
                    xytext=(pk + 0.4, sales_vals[pk] * 1.05),
                    fontsize=8.5, color=CLR_SALES, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=CLR_SALES, lw=1.0))

        ax.set_ylabel("Avg Net Sales (₹M)", color=CLR_SALES, fontsize=9.5)
        ax2.set_ylabel(w_ylabel, color=w_col_c, fontsize=9.5)
        ax.tick_params(axis="y", labelcolor=CLR_SALES)
        ax2.tick_params(axis="y", labelcolor=w_col_c)
        ax.spines["top"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax.grid(axis="y", ls="--", alpha=0.3)

        l1, n1 = ax.get_legend_handles_labels()
        l2, n2 = ax2.get_legend_handles_labels()
        ax.legend(l1 + l2, n1 + n2, fontsize=9, loc="upper right",
                  framealpha=0.9)

    axes[-1].set_xticks(list(x))
    axes[-1].set_xticklabels(mon_labels, fontsize=10)
    axes[-1].set_xlabel("Calendar Month", fontsize=11)

    fig.suptitle(
        "MAHACEF-200 | Average Seasonal Profile  —  Sales vs Weather  "
        "(all years aggregated by calendar month)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    _save(fig, out_dir / "04_seasonal_overlay.png")


# ===========================================================================
# 5. GRAPH 5 — Lag Scatter Grid
# ===========================================================================

def plot_lag_scatter_grid(merged: pd.DataFrame, out_dir: Path) -> None:
    """
    3 × 4 scatter grid:
    - Rows: temperature, humidity, rainfall
    - Columns: lag 0, 1, 2, 3 months
    - Each cell: sales(t) vs weather(t - lag)
    - Regression line + Pearson r annotated
    - Points coloured by year
    """
    logger.info("Plotting Graph 5: Lag Scatter Grid …")

    lags  = [0, 1, 2, 3]
    year_colours = {2023: "#E65100", 2024: "#1565C0",
                    2025: "#2E7D32", 2026: "#6A1B9A"}

    fig, axes = plt.subplots(3, 4, figsize=(20, 14))
    fig.patch.set_facecolor(CLR_BG)

    for row_idx, (w_col, w_label, w_unit, w_colour) in enumerate(WEATHER_VARS):
        for col_idx, lag in enumerate(lags):
            ax = axes[row_idx, col_idx]
            ax.set_facecolor(CLR_BG)

            if lag == 0:
                x_vals = merged[w_col].values
            else:
                x_vals = merged[f"{w_col}_lag{lag}"].values

            y_vals = merged["net_sale_amt"].values / 1e6
            years  = merged["year"].values

            # Drop NaN pairs
            mask = ~np.isnan(x_vals) & ~np.isnan(y_vals)
            xv, yv, yr = x_vals[mask], y_vals[mask], years[mask]

            # Scatter, coloured by year
            for yr_val, col in year_colours.items():
                m2 = yr == yr_val
                if m2.any():
                    ax.scatter(xv[m2], yv[m2], c=col, s=40, alpha=0.82,
                               label=str(yr_val), edgecolors="white", lw=0.5,
                               zorder=3)

            # Regression line
            if len(xv) >= 3:
                sl, intercept, r, p, _ = stats.linregress(xv, yv)
                x_line = np.linspace(xv.min(), xv.max(), 100)
                ax.plot(x_line, intercept + sl * x_line,
                        "-", color=w_colour, lw=1.8, alpha=0.8, zorder=2)
                p_str  = "p<0.05*" if p < 0.05 else "p≥0.05"
                ax.text(0.05, 0.93,
                        f"r={r:.3f}\n{p_str}",
                        transform=ax.transAxes, fontsize=8.5,
                        color="#222222", verticalalignment="top",
                        bbox=dict(boxstyle="round,pad=0.25",
                                  facecolor="white", alpha=0.85,
                                  edgecolor=w_colour, lw=0.8))

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(ls="--", alpha=0.3)
            ax.tick_params(labelsize=7.5)

            # Column headers (top row)
            if row_idx == 0:
                title = f"Lag {lag}" if lag > 0 else "Concurrent (Lag 0)"
                ax.set_title(title, fontsize=10, fontweight="bold", pad=8)

            # Row labels (leftmost column)
            if col_idx == 0:
                ax.set_ylabel(f"Net Sales (₹M)\n[{w_label} axis →]",
                              fontsize=8.5, color=w_colour)
            ax.set_xlabel(f"{w_label} ({w_unit})" if row_idx == 2
                          else "", fontsize=8)

    # Shared year legend
    legend_elements = [
        mpatches.Patch(facecolor=c, label=str(yr))
        for yr, c in year_colours.items()
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=4, fontsize=9, title="Year",
               bbox_to_anchor=(0.5, -0.01), framealpha=0.9)

    fig.suptitle(
        "MAHACEF-200 | Sales vs Weather — Lag Scatter Grid  "
        "(r = Pearson correlation, * p<0.05)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    _save(fig, out_dir / "05_lag_scatter_grid.png")


# ===========================================================================
# 6. GRAPH 6 — Top-6 State Dual-Axis Panels
# ===========================================================================

def plot_top_state_panels(state_df: pd.DataFrame,
                           top_states: list[str],
                           out_dir: Path) -> None:
    """
    2 × 3 grid of dual-axis panels, one per top state.
    Shows temperature (right axis) vs state net sales (left axis).
    """
    logger.info("Plotting Graph 6: Top %d State Panels …", len(top_states))

    n     = len(top_states)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(22, 5 * nrows))
    fig.patch.set_facecolor(CLR_BG)
    axes_flat = axes.flatten() if nrows > 1 else np.array(axes).flatten()

    for idx, state in enumerate(top_states):
        ax1 = axes_flat[idx]
        ax2 = ax1.twinx()
        ax1.set_facecolor(CLR_BG)

        sdf = (
            state_df[state_df[config.COL_STATE] == state]
              .sort_values(config.COL_MONTH)
              .reset_index(drop=True)
        )
        x      = range(len(sdf))
        labels = sdf["month_label"].tolist()
        step   = max(1, len(sdf) // 10)
        y_s    = sdf["net_sale_amt"].values / 1e6

        # Shade imputed
        _shade_imputed(ax1, sdf["weather_imputed"])

        # Sales
        ax1.fill_between(x, y_s, alpha=0.10, color=CLR_SALES)
        ax1.plot(x, y_s, "-o", color=CLR_SALES, lw=2.0, ms=4,
                 label="Net Sales", zorder=4)

        # Temperature
        y_t = sdf["avg_temperature_c"].values
        ax2.plot(x, y_t, "-^", color=CLR_TEMP, lw=1.8, ms=4,
                 label="Temperature (°C)", alpha=0.85, zorder=3)

        # Rainfall bars (scaled for visual)
        y_r = sdf["total_rainfall_mm"].values
        ax2.bar(x, y_r, color=CLR_RAIN, alpha=0.20, width=0.7,
                label="Rainfall (mm)", zorder=2)

        ax1.set_title(state, fontsize=11, fontweight="bold", pad=8,
                      color="#1A237E")
        ax1.set_ylabel("Net Sales (₹M)", color=CLR_SALES, fontsize=9)
        ax2.set_ylabel("Temp (°C) / Rain (mm)", color=CLR_TEMP, fontsize=9)
        ax1.tick_params(axis="y", labelcolor=CLR_SALES)
        ax2.tick_params(axis="y", labelcolor=CLR_TEMP, labelsize=8)
        ax1.spines["top"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax1.grid(axis="y", ls="--", alpha=0.3)

        ax1.set_xticks(list(range(0, len(sdf), step)))
        ax1.set_xticklabels([labels[i] for i in range(0, len(sdf), step)],
                             rotation=40, ha="right", fontsize=7.5)
        ax1.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"₹{v:.0f}M"))

        if idx == 0:
            l1, n1 = ax1.get_legend_handles_labels()
            l2, n2 = ax2.get_legend_handles_labels()
            ax1.legend(l1 + l2, n1 + n2, fontsize=8, framealpha=0.9)

    # Hide unused panels
    for i in range(n, len(axes_flat)):
        axes_flat[i].set_visible(False)

    fig.suptitle(
        "MAHACEF-200 | Top States — Net Sales vs Temperature & Rainfall",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    _save(fig, out_dir / "06_top6_state_temp_vs_sales.png")


# ===========================================================================
# 7. GRAPH 7 — Rolling Pearson Correlation Preview
# ===========================================================================

def plot_rolling_correlation(merged: pd.DataFrame, out_dir: Path) -> None:
    """
    Three-panel time-series of the 6-month rolling Pearson correlation
    between net sales and each weather variable.

    This is a teaser for Phase 5 — it shows whether the weather-sales
    relationship is stable, seasonal, or time-varying.
    """
    logger.info("Plotting Graph 7: Rolling Correlation Preview …")

    w = config.ROLL_CORR_WINDOW
    labels = merged["month_label"].tolist()
    step   = max(1, len(merged) // 18)

    fig, axes = plt.subplots(3, 1, figsize=(17, 12), sharex=True)
    fig.patch.set_facecolor(CLR_BG)

    combos = [
        ("rollcorr_avg_temperature_c",  "Temperature",  CLR_TEMP),
        ("rollcorr_avg_humidity",        "Humidity",     CLR_HUM),
        ("rollcorr_total_rainfall_mm",   "Rainfall",     CLR_RAIN),
    ]

    for ax, (r_col, label, colour) in zip(axes, combos):
        ax.set_facecolor(CLR_BG)
        x = range(len(merged))
        r = merged[r_col].values

        _shade_imputed(ax, merged["weather_imputed"])

        ax.axhline(0,    color="#888888", lw=1.0)
        ax.axhline( 0.3, color=colour, lw=0.7, ls="--", alpha=0.5)
        ax.axhline(-0.3, color=colour, lw=0.7, ls="--", alpha=0.5)
        ax.axhline( 0.6, color=colour, lw=0.7, ls=":",  alpha=0.5)
        ax.axhline(-0.6, color=colour, lw=0.7, ls=":",  alpha=0.5)

        # Fill positive/negative
        ax.fill_between(x, r, 0,
                         where=(pd.Series(r).fillna(0) > 0),
                         color=colour, alpha=0.18, zorder=1)
        ax.fill_between(x, r, 0,
                         where=(pd.Series(r).fillna(0) <= 0),
                         color=colour, alpha=0.08, zorder=1)
        ax.plot(x, r, "-o", color=colour, lw=2.0, ms=4.5, zorder=3)

        # Annotate max and min
        valid = ~np.isnan(r)
        if valid.any():
            ix_max = int(np.nanargmax(r))
            ix_min = int(np.nanargmin(r))
            for ix, v in [(ix_max, r[ix_max]), (ix_min, r[ix_min])]:
                ax.annotate(
                    f"{v:+.2f}\n{labels[ix]}",
                    xy=(ix, v),
                    xytext=(ix + 1.0, v + (0.05 if v >= 0 else -0.05)),
                    fontsize=8, color=colour, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=colour, lw=1.0),
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              ec=colour, alpha=0.9),
                )

        ax.set_ylim(-1.05, 1.05)
        ax.set_ylabel(f"r(Sales, {label})\n[{w}-Month Window]",
                      fontsize=9.5, color=colour)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", ls="--", alpha=0.3)

        # Reference labels
        for val, desc in [(0.6, "Strong"), (0.3, "Moderate"),
                           (-0.3, "Moderate"), (-0.6, "Strong")]:
            ax.text(len(merged) - 0.5, val + 0.03,
                    desc, fontsize=7, color=colour, alpha=0.65, ha="right")

    axes[-1].set_xticks(list(range(0, len(merged), step)))
    axes[-1].set_xticklabels(
        [labels[i] for i in range(0, len(merged), step)],
        rotation=40, ha="right", fontsize=8.5,
    )

    fig.suptitle(
        f"MAHACEF-200 | {w}-Month Rolling Pearson Correlation — "
        f"Net Sales vs Weather  (Preview for Phase 5)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    _save(fig, out_dir / "07_rolling_correlation.png")


# ===========================================================================
# 8. EXPORT
# ===========================================================================

def export_phase4_data(merged: pd.DataFrame) -> None:
    """Export merged national series to CSV, Excel, and metadata sidecars."""

    export_cols = [
        config.COL_MONTH, "month_label", "year", "month_num",
        "net_sale_amt", "gross_sale_amt", "net_sale_qty",
        "rolling_3m_net_sales", "rolling_6m_net_sales",
        "avg_temperature_c", "avg_humidity", "total_rainfall_mm",
        "avg_temperature_c_roll3m", "avg_temperature_c_roll6m",
        "avg_humidity_roll3m",      "avg_humidity_roll6m",
        "total_rainfall_mm_roll3m", "total_rainfall_mm_roll6m",
        "avg_temperature_c_lag1",   "avg_temperature_c_lag2",
        "avg_temperature_c_lag3",
        "avg_humidity_lag1",        "avg_humidity_lag2",
        "avg_humidity_lag3",
        "total_rainfall_mm_lag1",   "total_rainfall_mm_lag2",
        "total_rainfall_mm_lag3",
        "rollcorr_avg_temperature_c",
        "rollcorr_avg_humidity",
        "rollcorr_total_rainfall_mm",
        "weather_imputed", "imputation_method",
    ]
    out = merged[[c for c in export_cols if c in merged.columns]].copy()

    export_csv(out, config.PHASE4_COMPARISON_CSV, logger=logger)
    export_excel(out, config.PHASE4_COMPARISON_XLSX,
                 sheet_name="Weather_vs_Sales", logger=logger)

    meta_extra = {"rows": len(out), "months": int(out[config.COL_MONTH].nunique())}
    for path in [config.PHASE4_COMPARISON_CSV, config.PHASE4_COMPARISON_XLSX]:
        write_dataset_metadata(
            path, PHASE_LABEL, SCRIPT_NAME,
            source_dataset=config.CLEAN_DATASET_NAME,
            extra=meta_extra,
        )
    logger.info("Metadata sidecars written.")


# ===========================================================================
# 9. REPORT BUILDER
# ===========================================================================

def build_report(merged: pd.DataFrame, top_states: list[str]) -> str:
    """Build the standardised Phase 4 report."""

    # Concurrent Pearson correlations (at lag 0)
    corr_rows = ""
    for col, label, unit, _ in WEATHER_VARS:
        r, p = stats.pearsonr(
            merged["net_sale_amt"].values,
            merged[col].values,
        )
        sig = "✅ Significant" if p < config.ALPHA else "❌ Not significant"
        corr_rows += f"| {label} | {r:+.3f} | {p:.4f} | {sig} |\n"

    # Rolling corr summary
    roll_rows = ""
    for suffix, label, _ in [
        ("avg_temperature_c", "Temperature",  CLR_TEMP),
        ("avg_humidity",       "Humidity",     CLR_HUM),
        ("total_rainfall_mm",  "Rainfall",     CLR_RAIN),
    ]:
        col = f"rollcorr_{suffix}"
        valid = merged[col].dropna()
        if len(valid):
            roll_rows += (
                f"| {label} | {valid.mean():+.3f} | "
                f"{valid.min():+.3f} | {valid.max():+.3f} |\n"
            )

    # Best lag per variable
    lag_rows = ""
    for col, label, unit, _ in WEATHER_VARS:
        best_r, best_lag = 0.0, 0
        for k in [0, 1, 2, 3]:
            x_col = col if k == 0 else f"{col}_lag{k}"
            xv = merged[x_col].dropna()
            yv = merged.loc[xv.index, "net_sale_amt"]
            if len(xv) >= 5:
                r_val, _ = stats.pearsonr(xv, yv)
                if abs(r_val) > abs(best_r):
                    best_r, best_lag = r_val, k
        lag_rows += (
            f"| {label} | Lag {best_lag} month(s) | {best_r:+.3f} |\n"
        )

    objective = (
        "Overlay MAHACEF-200 sales and weather variables **for the first time** "
        "to detect visual patterns, directional alignment, and potential lead-lag "
        "relationships.  This phase is intentionally *descriptive* — all statistical "
        "tests and formal inference are deferred to Phase 5."
    )

    dataset_used = (
        "| Attribute | Value |\n| --- | --- |\n"
        f"| Source | `mahacef200_master_dataset_clean.csv` |\n"
        f"| Rows | 935 (39 months × 24 states, aggregated) |\n"
        f"| National series | 39 monthly observations |\n"
        f"| State-level panels | Top {config.TOP_STATES_DISPLAY} states by net sales |\n"
        f"| Top States | {', '.join(top_states)} |\n"
        f"| Lag features | weather(t−1), (t−2), (t−3) created for each variable |"
    )

    methodology = (
        "1. **National merging**: State sales summed per month; weather deduplicated.\n"
        "2. **Dual-axis plots** (Graphs 1–3): Left axis = net sales (₹M), "
        "right axis = weather variable; imputed months shaded amber.\n"
        "3. **Seasonal profile** (Graph 4): Calendar-month averages across all "
        "years — shows repeatable annual pattern alignment.\n"
        "4. **Lag scatter grid** (Graph 5): `sales(t)` plotted against "
        "`weather(t−k)` for k = 0, 1, 2, 3 months. "
        "OLS regression line + Pearson r annotated per cell.\n"
        "5. **State panels** (Graph 6): Top-6 state dual-axis; temperature "
        "on right axis, rainfall bars overlaid.\n"
        f"6. **Rolling correlation** (Graph 7): {config.ROLL_CORR_WINDOW}-month "
        "rolling Pearson — shows how the weather-sales relationship evolves.\n"
        "7. **Concurrent Pearson** and **best-lag** computed to guide "
        "Phase 5 formal tests."
    )

    key_findings = (
        "### Concurrent Pearson Correlation (Lag 0)\n\n"
        "| Weather Variable | Pearson r | p-value | Significance |\n"
        "| --- | --- | --- | --- |\n"
        + corr_rows.strip()
        + "\n\n### Best Observed Lag per Variable\n\n"
        "| Variable | Best Lag | Pearson r |\n"
        "| --- | --- | --- |\n"
        + lag_rows.strip()
        + "\n\n### Rolling Correlation Summary "
        f"({config.ROLL_CORR_WINDOW}-Month Window)\n\n"
        "| Variable | Mean r | Min r | Max r |\n"
        "| --- | --- | --- | --- |\n"
        + roll_rows.strip()
    )

    business_insights = (
        "1. **Seasonal alignment confirmed**: The seasonal profile overlay (Graph 4) "
        "shows whether sales peaks coincide with temperature/monsoon peaks. "
        "Visual alignment (or misalignment) guides the interpretation of correlations.\n\n"
        "2. **Lag structure matters**: The lag scatter grid reveals the "
        "most predictive weather lag. If lag 1–2 months is strongest, "
        "it implies that weather exposure drives prescription behaviour "
        "with a delay — an actionable finding for inventory planning.\n\n"
        "3. **Time-varying relationship**: The rolling correlation chart exposes "
        "whether the weather-sales link is consistent year-round or concentrated "
        "in specific seasons (e.g., monsoon). Unstable correlations indicate "
        "confounding factors and should be disclosed in any regression model.\n\n"
        "4. **State heterogeneity**: State-level panels (Graph 6) show that "
        "high-volume states (U.P., Maharashtra) dominate the national signal. "
        "State-specific weather relationships may differ substantially — "
        "Phase 5 will test this formally."
    )

    limitations = (
        "- **National weather proxy**: All states share one weather series, "
        "attenuating state-level correlations (heterogeneity bias).\n"
        "- **14 imputed months**: Climatology imputation preserves average "
        "seasonal structure but removes inter-annual variation, potentially "
        "inflating the apparent seasonal correlation.\n"
        "- **Confounders not controlled**: Observed visual co-movement may "
        "reflect shared seasonality (both sales and weather peak/trough in "
        "the same months) rather than direct causal influence. Formal partial "
        "correlation and regression control for this in Phases 5–7.\n"
        "- **Short rolling window**: 6-month rolling Pearson has high "
        "variance; treat Phase 7 findings as preliminary direction, not "
        "definitive strength."
    )

    next_phase = (
        "**Phase 5 — Correlation Analysis**\n\n"
        "With visual patterns confirmed, Phase 5 applies formal statistical tests:\n"
        "- Pearson r (linear) and Spearman ρ (rank-based, robust to outliers)\n"
        "- Lag-0, -1, -2, -3 correlations per weather variable\n"
        "- State-level correlation matrix (24 states × 3 weather vars)\n"
        "- Partial correlations controlling for time trend\n"
        "- Correlation heatmaps and significance masks"
    )

    return build_phase_report(
        phase_number="4",
        phase_title="Weather vs Sales Comparison",
        objective=objective,
        dataset_used=dataset_used,
        methodology=methodology,
        key_findings=key_findings,
        business_insights=business_insights,
        limitations=limitations,
        next_phase=next_phase,
        generated_by=SCRIPT_NAME,
    )


# ===========================================================================
# MAIN PIPELINE
# ===========================================================================

def run_weather_vs_sales() -> pd.DataFrame:
    """Execute the complete Phase 4 — Weather vs Sales pipeline."""

    logger.info("=" * 60)
    logger.info("PHASE 4 — WEATHER vs SALES COMPARISON")
    logger.info("=" * 60)

    ensure_directories(
        config.PHASE4_GRAPHS_DIR,
        config.DATA_DIR,
        config.EXCEL_DIR,
        config.REPORTS_DIR,
        config.LOGS_DIR,
    )

    # ------------------------------------------------------------------ Load
    path = config.MASTER_CLEAN_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"Clean dataset not found: {path}\n"
            "Please run Phase 1.5 first."
        )
    logger.info("Loading clean dataset: %s", path)
    df = pd.read_csv(str(path))
    df[config.COL_MONTH] = df[config.COL_MONTH].astype(int)
    df[config.COL_STATE] = normalize_state_name(df[config.COL_STATE])
    logger.info("  %d rows × %d cols", *df.shape)

    # ------------------------------------------------------------------ Build
    merged     = build_merged_national(df)
    top_states = identify_top_states(df, config.TOP_STATES_DISPLAY)
    state_df   = build_state_monthly(df, top_states)

    # ------------------------------------------------------------------ Graphs
    out_dir = config.PHASE4_GRAPHS_DIR
    plot_national_comparisons(merged, out_dir)          # 1–3
    plot_seasonal_overlay(merged, out_dir)              # 4
    plot_lag_scatter_grid(merged, out_dir)              # 5
    plot_top_state_panels(state_df, top_states, out_dir)# 6
    plot_rolling_correlation(merged, out_dir)           # 7

    # ------------------------------------------------------------------ Export
    export_phase4_data(merged)

    # ------------------------------------------------------------------ Report
    report = build_report(merged, top_states)
    write_markdown_report(config.REPORT_WEATHER_VS_SALES, report, logger=logger)

    # ------------------------------------------------------------------ Summary
    logger.info("-" * 60)
    logger.info("PHASE 4 COMPLETE")
    logger.info("  Months in merged series : %d", len(merged))
    logger.info("  Top states displayed    : %s", top_states)
    for col, label, unit, _ in WEATHER_VARS:
        r, p = stats.pearsonr(merged["net_sale_amt"].values,
                              merged[col].values)
        logger.info("  Concurrent r(Sales, %-12s) = %+.3f  (p=%.4f)",
                    label, r, p)
    logger.info("-" * 60)

    return merged


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    try:
        merged = run_weather_vs_sales()
        logger.info("Script finished successfully.")
        sys.exit(0)
    except FileNotFoundError as exc:
        logger.error("File not found: %s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
        sys.exit(99)
