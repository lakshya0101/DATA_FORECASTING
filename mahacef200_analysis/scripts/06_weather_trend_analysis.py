"""
06_weather_trend_analysis.py
=============================
Phase 3 — Weather Trend Analysis

Objective:
    Understand the weather behaviour of India (national level) independently,
    before linking it to MAHACEF-200 sales. This phase is intentionally
    sales-free — weather patterns are characterised on their own terms so
    that Phase 4 comparisons are grounded in a solid weather baseline.

Outputs
-------
data/phase3_weather_monthly.csv
data/phase3_weather_monthly.metadata.json
excel/Phase3_Weather_Trend.xlsx
graphs/phase3_weather/
    01_temperature_trend.png
    02_humidity_trend.png
    03_rainfall_trend.png
    04_weather_stl_decomposition.png
    05_weather_calendar_heatmap.png
    06_monsoon_analysis.png
    07_weather_anomalies.png
reports/Phase3_Weather_Trend.md

Usage
-----
    python mahacef200_analysis/scripts/06_weather_trend_analysis.py
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
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.seasonal import STL

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
CLR_TEMP    = "#D32F2F"   # red     — temperature
CLR_HUM     = "#1565C0"   # blue    — humidity
CLR_RAIN    = "#1B5E20"   # green   — rainfall
CLR_ROLL3   = "#FF8A65"   # orange  — 3M rolling (temp context)
CLR_ROLL6   = "#42A5F5"   # sky     — 6M rolling
CLR_TREND   = "#6A1B9A"   # purple  — OLS trend
CLR_ANOMALY = "#E65100"   # burnt orange — anomaly markers
CLR_IMPUTED = "#FFF9C4"   # yellow  — imputed month shade
CLR_BG      = "#F8F9FA"   # off-white — figure bg

MONTH_NAMES = {
    1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr", 5:"May", 6:"Jun",
    7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Dec",
}

SCRIPT_NAME = "06_weather_trend_analysis.py"
PHASE_LABEL = "Phase 3 - Weather Trend Analysis"

# Variable definitions: (column, label, unit, colour)
WEATHER_VARS: list[tuple[str, str, str, str]] = [
    ("avg_temperature_c",  "Avg Temperature", "°C",  CLR_TEMP),
    ("avg_humidity",       "Avg Humidity",    "%",   CLR_HUM),
    ("total_rainfall_mm",  "Total Rainfall",  "mm",  CLR_RAIN),
]


# ===========================================================================
# 1. DATA LOADING — deduplicate to monthly weather series
# ===========================================================================

def load_weather_monthly_series() -> pd.DataFrame:
    """
    Load the clean master dataset and deduplicate to a single row per
    billing_month.  Since all 24 states share the same national weather
    values, we simply take the first occurrence of each month.

    Returns
    -------
    pd.DataFrame
        39-row monthly weather series with:
        billing_month, month_label, month_date, month_num, year,
        avg_temperature_c, avg_humidity, total_rainfall_mm,
        weather_imputed, imputation_method, weather_obs_count.
    """
    path = config.MASTER_CLEAN_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"Clean dataset not found: {path}\n"
            "Please run Phase 1.5 (04_weather_imputation.py) first."
        )
    logger.info("Loading clean dataset: %s", path)
    df = pd.read_csv(str(path))
    df[config.COL_MONTH] = df[config.COL_MONTH].astype(int)

    weather_cols = [
        config.COL_MONTH, "month_label", "month_date",
        "avg_temperature_c", "avg_humidity", "total_rainfall_mm",
        "weather_imputed", "imputation_method", "weather_obs_count",
    ]
    # Deduplicate — all states share identical weather per month
    wdf = (
        df[weather_cols]
          .drop_duplicates(subset=[config.COL_MONTH])
          .sort_values(config.COL_MONTH)
          .reset_index(drop=True)
    )
    wdf["month_num"] = wdf[config.COL_MONTH] % 100
    wdf["year"]      = wdf[config.COL_MONTH] // 100
    wdf["month_date"] = pd.to_datetime(wdf["month_date"])

    n_obs     = int((~wdf["weather_imputed"]).sum())
    n_imputed = int(wdf["weather_imputed"].sum())
    logger.info(
        "  Weather series: %d months (%d observed, %d climatology-imputed)",
        len(wdf), n_obs, n_imputed,
    )
    return wdf


# ===========================================================================
# 2. ROLLING METRICS
# ===========================================================================

def compute_weather_rolling(wdf: pd.DataFrame) -> pd.DataFrame:
    """
    Add 3-month and 6-month trailing rolling means for each weather variable.
    """
    logger.info("Computing 3M / 6M rolling means for weather variables …")
    for col, _, _, _ in WEATHER_VARS:
        for w in config.ROLLING_WINDOWS:
            wdf[f"{col}_roll{w}m"] = (
                wdf[col].rolling(w, min_periods=1).mean().round(3)
            )
    return wdf


# ===========================================================================
# 3. STL DECOMPOSITIONS
# ===========================================================================

def run_weather_stl(wdf: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Run STL decomposition (period=12) on each weather variable.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        wdf with stl_{col}_{component} columns, and a dict of
        {col: {"seasonal_strength": float, "trend_strength": float}}
    """
    logger.info("Running STL decompositions (period=%d) …", config.STL_PERIOD)
    strengths: dict[str, dict] = {}

    date_index = pd.date_range(
        start=wdf["month_date"].iloc[0], periods=len(wdf), freq="MS"
    )

    for col, label, _, _ in WEATHER_VARS:
        ts = pd.Series(wdf[col].values, index=date_index)
        stl = STL(ts, period=config.STL_PERIOD, robust=True)
        res = stl.fit()

        wdf[f"stl_{col}_trend"]    = res.trend
        wdf[f"stl_{col}_seasonal"] = res.seasonal
        wdf[f"stl_{col}_residual"] = res.resid

        var_r  = np.var(res.resid)
        var_sr = np.var(res.seasonal + res.resid)
        var_tr = np.var(res.trend + res.resid)
        s_str  = max(0.0, 1 - var_r / var_sr) if var_sr > 0 else 0.0
        t_str  = max(0.0, 1 - var_r / var_tr) if var_tr > 0 else 0.0

        strengths[col] = {"seasonal_strength": round(s_str, 3),
                           "trend_strength":   round(t_str, 3)}
        logger.info(
            "  %s → seasonal str: %.3f | trend str: %.3f",
            label, s_str, t_str,
        )
    return wdf, strengths


# ===========================================================================
# 4. OLS TREND DETECTION
# ===========================================================================

def detect_weather_trends(wdf: pd.DataFrame) -> dict[str, dict]:
    """
    Fit a linear OLS trend to each weather variable.

    Returns
    -------
    dict
        {col: {"slope": float, "r2": float, "p_value": float,
               "direction": str, "unit": str}}
    """
    logger.info("Detecting OLS trends for weather variables …")
    trends: dict[str, dict] = {}
    x = np.arange(len(wdf), dtype=float)

    for col, label, unit, _ in WEATHER_VARS:
        sl, intercept, r, p, _ = stats.linregress(x, wdf[col].values)
        wdf[f"ols_{col}"] = intercept + sl * x
        r2 = r ** 2
        direction = (
            "Significant upward"   if sl > 0 and p < config.ALPHA else
            "Significant downward" if sl < 0 and p < config.ALPHA else
            f"Slight {'up' if sl >= 0 else 'down'}ward (not significant)"
        )
        trends[col] = {
            "slope": round(sl, 4), "intercept": round(intercept, 4),
            "r2": round(r2, 4), "p_value": round(p, 6),
            "direction": direction, "unit": unit, "label": label,
        }
        logger.info(
            "  %s: slope=%.4f %s/month | R²=%.4f | p=%.4f | %s",
            label, sl, unit, r2, p, direction,
        )
    return trends


# ===========================================================================
# 5. ANOMALY DETECTION
# ===========================================================================

def detect_weather_anomalies(wdf: pd.DataFrame) -> pd.DataFrame:
    """
    Flag months where any weather variable exceeds ±ANOMALY_ZSCORE_THRESHOLD
    standard deviations from the mean.

    A separate z-score column and boolean flag is added per variable.
    """
    logger.info(
        "Detecting anomalies (|z| > %.1f) …", config.ANOMALY_ZSCORE_THRESHOLD
    )
    for col, label, _, _ in WEATHER_VARS:
        mean = wdf[col].mean()
        std  = wdf[col].std()
        wdf[f"z_{col}"] = ((wdf[col] - mean) / std).round(3)
        wdf[f"anomaly_{col}"] = wdf[f"z_{col}"].abs() > config.ANOMALY_ZSCORE_THRESHOLD

        anomaly_months = wdf[wdf[f"anomaly_{col}"]]["month_label"].tolist()
        logger.info("  %s: %d anomaly month(s) → %s",
                    label, len(anomaly_months), anomaly_months)
    return wdf


# ===========================================================================
# 6. MONSOON METRICS
# ===========================================================================

def compute_monsoon_metrics(wdf: pd.DataFrame) -> pd.DataFrame:
    """
    Extract monsoon season (Jun–Sep) statistics per year.

    Returns
    -------
    pd.DataFrame
        One row per year with: total_rain, avg_temp, avg_humidity,
        imputed_flag, n_months (number of monsoon months available).
    """
    logger.info("Computing monsoon metrics (Jun–Sep) …")
    mon = wdf[wdf["month_num"].isin(config.MONSOON_MONTHS)].copy()
    monsoon = (
        mon.groupby("year")
           .agg(
               total_rainfall_mm   =("total_rainfall_mm", "sum"),
               avg_temperature_c   =("avg_temperature_c", "mean"),
               avg_humidity        =("avg_humidity",      "mean"),
               n_months            =("month_num",         "count"),
               any_imputed         =("weather_imputed",   "any"),
           )
           .reset_index()
    )
    monsoon["total_rainfall_mm"] = monsoon["total_rainfall_mm"].round(2)
    monsoon["avg_temperature_c"] = monsoon["avg_temperature_c"].round(2)
    monsoon["avg_humidity"]      = monsoon["avg_humidity"].round(2)

    for _, row in monsoon.iterrows():
        logger.info(
            "  Monsoon %d: %.1fmm rain | %.1f°C | %.1f%% hum | "
            "%d months%s",
            int(row["year"]), row["total_rainfall_mm"],
            row["avg_temperature_c"], row["avg_humidity"],
            int(row["n_months"]),
            " [imputed]" if row["any_imputed"] else "",
        )
    return monsoon


# ===========================================================================
# 7. SHARED SAVE HELPER
# ===========================================================================

def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved → %s", path.name)


def _shade_imputed(ax: plt.Axes, wdf: pd.DataFrame) -> None:
    """Shade imputed months with a light yellow background band."""
    for i, imp in enumerate(wdf["weather_imputed"].values):
        if imp:
            ax.axvspan(i - 0.5, i + 0.5, color=CLR_IMPUTED, alpha=0.55, zorder=0)


def _xtick_setup(ax: plt.Axes, wdf: pd.DataFrame, step: int = 3) -> None:
    labels = wdf["month_label"].tolist()
    ticks  = list(range(0, len(labels), step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([labels[i] for i in ticks], rotation=40,
                       ha="right", fontsize=8.5)


def _clean_ax(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_facecolor(CLR_BG)


# ===========================================================================
# 8. GRAPH 1 — Temperature Trend
# ===========================================================================

def plot_temperature_trend(wdf: pd.DataFrame, out_dir: Path) -> None:
    """
    Line chart: avg temperature (°C) with 3M/6M rolling means,
    OLS trend, and imputed months shaded.
    """
    logger.info("Plotting Graph 1: Temperature Trend …")
    col = "avg_temperature_c"
    fig, ax = plt.subplots(figsize=(18, 7))
    fig.patch.set_facecolor(CLR_BG)
    _clean_ax(ax)

    x = range(len(wdf))
    y = wdf[col].values

    _shade_imputed(ax, wdf)
    ax.fill_between(x, y, alpha=0.08, color=CLR_TEMP)
    ax.plot(x, y, "-o", color=CLR_TEMP, lw=2.2, ms=4.5,
            label="Monthly Avg Temp (°C)", zorder=4)
    ax.plot(x, wdf[f"{col}_roll3m"].values, "--", color="#FF8A65",
            lw=1.8, label="3-Month Rolling Mean", zorder=3)
    ax.plot(x, wdf[f"{col}_roll6m"].values, "-.", color="#BF360C",
            lw=1.8, label="6-Month Rolling Mean", zorder=3)
    ax.plot(x, wdf[f"ols_{col}"].values, ":", color=CLR_TREND,
            lw=2.0, alpha=0.85, label="OLS Trend", zorder=2)

    # Peak & trough
    for idx_fn, lbl, col_ann in [(np.argmax, "Peak", "#B71C1C"),
                                  (np.argmin, "Trough", "#1565C0")]:
        idx = int(idx_fn(y))
        ax.annotate(
            f"{lbl}\n{wdf['month_label'].iloc[idx]}\n{y[idx]:.1f}°C",
            xy=(idx, y[idx]),
            xytext=(idx + 1.5, y[idx] + (1.8 if lbl == "Peak" else -1.8)),
            fontsize=8.5, color=col_ann, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=col_ann, lw=1.2),
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec=col_ann, alpha=0.9),
        )

    imp_patch = mpatches.Patch(color=CLR_IMPUTED, alpha=0.8,
                                label="Climatology Imputed")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [imp_patch], fontsize=9, framealpha=0.9)

    _xtick_setup(ax, wdf)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}°C"))
    ax.set_ylabel("Average Temperature (°C)", fontsize=11)
    ax.set_xlabel("Billing Month", fontsize=10)
    ax.set_title(
        "MAHACEF-200 | India National Average Temperature Trend  "
        "(Yellow shading = Climatology-Imputed)",
        fontsize=13, fontweight="bold", pad=14,
    )
    plt.tight_layout()
    _save(fig, out_dir / "01_temperature_trend.png")


# ===========================================================================
# 9. GRAPH 2 — Humidity Trend
# ===========================================================================

def plot_humidity_trend(wdf: pd.DataFrame, out_dir: Path) -> None:
    """
    Line chart: avg humidity (%) with rolling means and OLS trend.
    """
    logger.info("Plotting Graph 2: Humidity Trend …")
    col = "avg_humidity"
    fig, ax = plt.subplots(figsize=(18, 7))
    fig.patch.set_facecolor(CLR_BG)
    _clean_ax(ax)

    x = range(len(wdf))
    y = wdf[col].values

    _shade_imputed(ax, wdf)
    ax.fill_between(x, y, alpha=0.08, color=CLR_HUM)
    ax.plot(x, y, "-o", color=CLR_HUM, lw=2.2, ms=4.5,
            label="Monthly Avg Humidity (%)", zorder=4)
    ax.plot(x, wdf[f"{col}_roll3m"].values, "--", color="#64B5F6",
            lw=1.8, label="3-Month Rolling Mean", zorder=3)
    ax.plot(x, wdf[f"{col}_roll6m"].values, "-.", color="#0D47A1",
            lw=1.8, label="6-Month Rolling Mean", zorder=3)
    ax.plot(x, wdf[f"ols_{col}"].values, ":", color=CLR_TREND,
            lw=2.0, alpha=0.85, label="OLS Trend", zorder=2)

    # Monsoon shading (Jun–Sep)
    for i, mon in enumerate(wdf["month_num"].values):
        if mon in config.MONSOON_MONTHS and not wdf["weather_imputed"].iloc[i]:
            ax.axvspan(i - 0.5, i + 0.5, color="#B3E5FC", alpha=0.3, zorder=0)

    monsoon_patch = mpatches.Patch(color="#B3E5FC", alpha=0.6,
                                    label="Monsoon Months (Jun–Sep, observed)")
    imp_patch = mpatches.Patch(color=CLR_IMPUTED, alpha=0.8,
                                label="Climatology Imputed")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [monsoon_patch, imp_patch],
              fontsize=9, framealpha=0.9)

    _xtick_setup(ax, wdf)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_ylim(0, 110)
    ax.set_ylabel("Average Humidity (%)", fontsize=11)
    ax.set_xlabel("Billing Month", fontsize=10)
    ax.set_title(
        "MAHACEF-200 | India National Average Humidity Trend  "
        "(Blue shading = Monsoon, Yellow = Imputed)",
        fontsize=13, fontweight="bold", pad=14,
    )
    plt.tight_layout()
    _save(fig, out_dir / "02_humidity_trend.png")


# ===========================================================================
# 10. GRAPH 3 — Rainfall Trend
# ===========================================================================

def plot_rainfall_trend(wdf: pd.DataFrame, out_dir: Path) -> None:
    """
    Dual representation: bar chart (rainfall volume per month) overlaid
    with 3M rolling mean line. Monsoon months coloured distinctly.
    """
    logger.info("Plotting Graph 3: Rainfall Trend …")
    col = "total_rainfall_mm"
    fig, ax = plt.subplots(figsize=(18, 7))
    fig.patch.set_facecolor(CLR_BG)
    _clean_ax(ax)

    x    = np.arange(len(wdf))
    y    = wdf[col].values
    is_monsoon = wdf["month_num"].isin(config.MONSOON_MONTHS).values
    is_imp     = wdf["weather_imputed"].values

    bar_colors = []
    for monsoon, imputed in zip(is_monsoon, is_imp):
        if imputed:
            bar_colors.append("#FFE082")   # amber — imputed
        elif monsoon:
            bar_colors.append("#1B5E20")   # dark green — monsoon observed
        else:
            bar_colors.append("#A5D6A7")   # light green — non-monsoon observed

    bars = ax.bar(x, y, color=bar_colors, edgecolor="white",
                  width=0.75, zorder=2)

    # Rolling mean overlay
    ax.plot(x, wdf[f"{col}_roll3m"].values, "-o", color="#E65100",
            lw=2.0, ms=4, label="3-Month Rolling Mean", zorder=4)
    ax.plot(x, wdf[f"ols_{col}"].values, ":", color=CLR_TREND,
            lw=1.8, alpha=0.85, label="OLS Trend", zorder=3)

    # Annotate monsoon peaks
    for i in x:
        if y[i] > 8:
            ax.text(i, y[i] + 0.2, f"{y[i]:.1f}", ha="center",
                    fontsize=7.5, color="#1B5E20", fontweight="bold")

    mon_patch = mpatches.Patch(color="#1B5E20", label="Monsoon (observed)")
    imp_patch = mpatches.Patch(color="#FFE082", alpha=0.9,
                                label="Climatology Imputed")
    dry_patch = mpatches.Patch(color="#A5D6A7",
                                label="Non-Monsoon (observed)")
    handles, lbls = ax.get_legend_handles_labels()
    ax.legend(handles=[mon_patch, dry_patch, imp_patch] + handles,
              fontsize=9, framealpha=0.9)

    _xtick_setup(ax, wdf)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}mm"))
    ax.set_ylabel("Total Rainfall (mm)", fontsize=11)
    ax.set_xlabel("Billing Month", fontsize=10)
    ax.set_title(
        "MAHACEF-200 | India National Monthly Rainfall  "
        "(Dark green = Monsoon, Amber = Imputed)",
        fontsize=13, fontweight="bold", pad=14,
    )
    plt.tight_layout()
    _save(fig, out_dir / "03_rainfall_trend.png")


# ===========================================================================
# 11. GRAPH 4 — STL Decomposition (All 3 Variables)
# ===========================================================================

def plot_weather_stl(wdf: pd.DataFrame, out_dir: Path) -> None:
    """
    3 × 4 panel grid: each row = one weather variable,
    each column = Observed / Trend / Seasonal / Residual.
    """
    logger.info("Plotting Graph 4: STL Decomposition (all variables) …")

    fig = plt.figure(figsize=(24, 15))
    fig.patch.set_facecolor(CLR_BG)
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.35)

    x      = range(len(wdf))
    labels = wdf["month_label"].tolist()
    step   = max(1, len(wdf) // 12)

    col_titles = ["Observed", "STL Trend", "STL Seasonal", "STL Residual"]

    for row_idx, (col, var_label, unit, colour) in enumerate(WEATHER_VARS):
        components = [
            wdf[col].values,
            wdf[f"stl_{col}_trend"].values,
            wdf[f"stl_{col}_seasonal"].values,
            wdf[f"stl_{col}_residual"].values,
        ]
        styles = ["-o", "-", "-", "o"]
        alphas = [0.85, 0.9, 0.85, 0.75]

        for col_idx, (vals, style, alpha) in enumerate(
            zip(components, styles, alphas)
        ):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            ax.set_facecolor(CLR_BG)

            if row_idx == 0 and col_idx == 0:
                # Shade imputed in observed panel only
                _shade_imputed(ax, wdf)

            if "o" in style and "-" in style:
                ax.fill_between(x, vals, alpha=0.07, color=colour)
                ax.plot(x, vals, style, color=colour, lw=1.8,
                        ms=3, alpha=alpha)
            elif "-" in style:
                ax.fill_between(x, vals, alpha=0.1, color=colour)
                ax.plot(x, vals, style, color=colour, lw=2.0, alpha=alpha)
            else:
                ax.scatter(x, vals, color=colour, s=18, alpha=alpha, zorder=3)

            ax.axhline(0, color="#AAAAAA", lw=0.6, ls=":")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="y", ls="--", alpha=0.3)
            ax.tick_params(axis="both", labelsize=7.5)

            # Column title (top row only)
            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=10,
                             fontweight="bold", pad=8)
            # Row label (leftmost column only)
            if col_idx == 0:
                ax.set_ylabel(f"{var_label} ({unit})", fontsize=9,
                              color=colour, fontweight="semibold")

            # X-axis (bottom row only)
            if row_idx == 2:
                ticks = list(range(0, len(wdf), step))
                ax.set_xticks(ticks)
                ax.set_xticklabels([labels[i] for i in ticks],
                                   rotation=45, ha="right", fontsize=7)
            else:
                ax.set_xticks([])

    fig.suptitle(
        "MAHACEF-200 | STL Decomposition — Temperature, Humidity & Rainfall",
        fontsize=14, fontweight="bold", y=1.01,
    )
    _save(fig, out_dir / "04_weather_stl_decomposition.png")


# ===========================================================================
# 12. GRAPH 5 — Calendar Heatmap (Month × Year)
# ===========================================================================

def plot_weather_calendar_heatmap(wdf: pd.DataFrame, out_dir: Path) -> None:
    """
    Calendar heatmap: rows = calendar months (Jan–Dec),
    columns = years (2023–2026), colour = weather value.
    Three side-by-side subplots, one per variable.
    """
    logger.info("Plotting Graph 5: Weather Calendar Heatmap …")

    years = sorted(wdf["year"].unique())
    month_order = list(range(1, 13))
    ylabels = [MONTH_NAMES[m] for m in month_order]

    fig, axes = plt.subplots(1, 3, figsize=(18, 9))
    fig.patch.set_facecolor(CLR_BG)

    cmaps = ["YlOrRd", "Blues", "YlGnBu"]

    for ax, (col, var_label, unit, colour), cmap in zip(
        axes, WEATHER_VARS, cmaps
    ):
        # Build 12 × n_years pivot
        pivot = np.full((12, len(years)), np.nan)
        for yi, yr in enumerate(years):
            yr_data = wdf[wdf["year"] == yr].set_index("month_num")[col]
            for mi, mon in enumerate(month_order):
                if mon in yr_data.index:
                    pivot[mi, yi] = yr_data.loc[mon]

        # Draw heatmap
        ax.set_facecolor(CLR_BG)
        im = ax.imshow(pivot, aspect="auto", cmap=cmap,
                       interpolation="nearest",
                       vmin=np.nanmin(pivot), vmax=np.nanmax(pivot))

        # Annotate cells
        for mi in range(12):
            for yi in range(len(years)):
                val = pivot[mi, yi]
                if not np.isnan(val):
                    # Is this imputed?
                    bm = years[yi] * 100 + month_order[mi]
                    imp_row = wdf[wdf[config.COL_MONTH] == bm]
                    is_imp = bool(imp_row["weather_imputed"].values[0]) \
                             if len(imp_row) else False
                    txt = f"{val:.0f}" if unit != "°C" else f"{val:.1f}"
                    txt_colour = "#888888" if is_imp else "#111111"
                    ax.text(yi, mi, txt, ha="center", va="center",
                            fontsize=8, color=txt_colour,
                            fontstyle="italic" if is_imp else "normal")

        ax.set_yticks(range(12))
        ax.set_yticklabels(ylabels, fontsize=9)
        ax.set_xticks(range(len(years)))
        ax.set_xticklabels([str(y) for y in years], fontsize=10,
                           fontweight="bold")
        ax.set_title(f"{var_label} ({unit})", fontsize=12,
                     fontweight="bold", pad=10, color=colour)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label=f"{unit}")

    fig.suptitle(
        "MAHACEF-200 | Weather Calendar Heatmap  "
        "(Italic = Climatology-Imputed)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    _save(fig, out_dir / "05_weather_calendar_heatmap.png")


# ===========================================================================
# 13. GRAPH 6 — Monsoon Analysis
# ===========================================================================

def plot_monsoon_analysis(wdf: pd.DataFrame, monsoon: pd.DataFrame,
                          out_dir: Path) -> None:
    """
    Four-panel monsoon figure:
    P1 — Monsoon season total rainfall by year (bar)
    P2 — Monsoon season avg temperature by year (bar)
    P3 — Monsoon season avg humidity by year (bar)
    P4 — Month-by-month pattern for each monsoon year (multi-line)
    """
    logger.info("Plotting Graph 6: Monsoon Analysis …")

    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor(CLR_BG)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    years    = monsoon["year"].values.astype(int)
    n_years  = len(years)
    xpos     = np.arange(n_years)
    bar_cols = ["#FFE082" if monsoon["any_imputed"].iloc[i]
                else "#1B5E20" for i in range(n_years)]

    def _bar_panel(ax: plt.Axes, vals, ylabel: str, title: str,
                   bar_color_list: list, fmt: str = ".1f") -> None:
        ax.set_facecolor(CLR_BG)
        bars = ax.bar(xpos, vals, color=bar_color_list,
                      edgecolor="white", width=0.55)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + max(vals) * 0.01,
                    f"{v:{fmt}}", ha="center", fontsize=9.5,
                    fontweight="bold")
        ax.set_xticks(xpos)
        ax.set_xticklabels([str(y) for y in years], fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", ls="--", alpha=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    _bar_panel(ax1, monsoon["total_rainfall_mm"].values,
               "Total Rainfall (mm)", "Monsoon Season Rainfall (Jun–Sep)",
               bar_cols, ".1f")

    ax2 = fig.add_subplot(gs[0, 1])
    _bar_panel(ax2, monsoon["avg_temperature_c"].values,
               "Avg Temperature (°C)", "Monsoon Season Temperature (Jun–Sep)",
               bar_cols, ".1f")
    ax2.set_ylim(0, monsoon["avg_temperature_c"].max() * 1.25)

    ax3 = fig.add_subplot(gs[1, 0])
    _bar_panel(ax3, monsoon["avg_humidity"].values,
               "Avg Humidity (%)", "Monsoon Season Humidity (Jun–Sep)",
               bar_cols, ".1f")

    # Panel 4 — month-by-month rainfall per year
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(CLR_BG)
    monsoon_months_order = config.MONSOON_MONTHS  # [6,7,8,9]
    line_colors = ["#1B5E20", "#388E3C", "#F57F17", "#E65100"]
    mon_labels  = [MONTH_NAMES[m] for m in monsoon_months_order]

    for (_, yr_row), lc in zip(
        monsoon.iterrows(), line_colors[:n_years]
    ):
        yr  = int(yr_row["year"])
        yr_data = wdf[(wdf["year"] == yr) &
                      (wdf["month_num"].isin(monsoon_months_order))]
        yr_data = yr_data.set_index("month_num")["total_rainfall_mm"]
        vals = [yr_data.get(m, np.nan) for m in monsoon_months_order]
        style = "--" if bool(monsoon[monsoon["year"] == yr]["any_imputed"].values[0]) else "-"
        ax4.plot(range(4), vals, style + "o", color=lc, lw=2.0,
                 ms=7, label=f"{yr}{'*' if style == '--' else ''}")

    ax4.set_xticks(range(4))
    ax4.set_xticklabels(mon_labels, fontsize=10)
    ax4.set_ylabel("Rainfall (mm)", fontsize=10)
    ax4.set_title("Month-by-Month Monsoon Rainfall by Year\n"
                  "(*dashed = imputed data)", fontsize=11, fontweight="bold")
    ax4.legend(fontsize=9, framealpha=0.9)
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)
    ax4.grid(axis="y", ls="--", alpha=0.35)

    imp_patch = mpatches.Patch(color="#FFE082", alpha=0.9,
                                label="Climatology-Imputed Year")
    obs_patch = mpatches.Patch(color="#1B5E20", label="Observed Year")
    fig.legend(handles=[obs_patch, imp_patch], loc="lower center",
               ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        "MAHACEF-200 | Monsoon Season Analysis (Jun–Sep)",
        fontsize=14, fontweight="bold", y=1.01,
    )
    _save(fig, out_dir / "06_monsoon_analysis.png")


# ===========================================================================
# 14. GRAPH 7 — Weather Anomalies
# ===========================================================================

def plot_weather_anomalies(wdf: pd.DataFrame, out_dir: Path) -> None:
    """
    Three-panel z-score chart, one per weather variable.
    Anomalous months (|z| > threshold) annotated and highlighted.
    Reference lines at ±1σ and ±2σ.
    """
    logger.info("Plotting Graph 7: Weather Anomalies …")

    fig, axes = plt.subplots(3, 1, figsize=(18, 14), sharex=True)
    fig.patch.set_facecolor(CLR_BG)

    x      = range(len(wdf))
    step   = max(1, len(wdf) // 18)
    labels = wdf["month_label"].tolist()

    for ax, (col, var_label, unit, colour) in zip(axes, WEATHER_VARS):
        z_col = f"z_{col}"
        a_col = f"anomaly_{col}"
        z     = wdf[z_col].values
        flags = wdf[a_col].values

        ax.set_facecolor(CLR_BG)
        _shade_imputed(ax, wdf)

        # Reference bands
        ax.axhspan(-1, 1, color=colour, alpha=0.06, zorder=0)
        ax.axhspan(-config.ANOMALY_ZSCORE_THRESHOLD,
                    config.ANOMALY_ZSCORE_THRESHOLD,
                    color=colour, alpha=0.03, zorder=0)
        for level, ls in [(-1, "--"), (1, "--"),
                           (-config.ANOMALY_ZSCORE_THRESHOLD, ":"),
                           ( config.ANOMALY_ZSCORE_THRESHOLD, ":")]:
            ax.axhline(level, color=colour, lw=0.9, ls=ls, alpha=0.6)

        ax.axhline(0, color="#666666", lw=0.8)

        # Z-score line
        ax.fill_between(x, z, 0, where=(np.array(z) > 0),
                        color=colour, alpha=0.12)
        ax.fill_between(x, z, 0, where=(np.array(z) < 0),
                        color=colour, alpha=0.08)
        ax.plot(x, z, "-o", color=colour, lw=2.0, ms=4, zorder=3)

        # Anomaly markers
        anom_x = [i for i, f in enumerate(flags) if f]
        anom_z = [z[i] for i in anom_x]
        ax.scatter(anom_x, anom_z, color=CLR_ANOMALY, s=110, zorder=5,
                   edgecolors="white", lw=1.2,
                   label=f"Anomaly (|z|>{config.ANOMALY_ZSCORE_THRESHOLD})")
        for i, zi in zip(anom_x, anom_z):
            ax.annotate(
                f"{labels[i]}\nz={zi:+.2f}",
                xy=(i, zi),
                xytext=(i + 0.6, zi + (0.3 if zi > 0 else -0.3)),
                fontsize=8, color=CLR_ANOMALY, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=CLR_ANOMALY, lw=1.0),
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec=CLR_ANOMALY, alpha=0.9),
            )

        ax.set_ylabel(f"{var_label}\nZ-score", fontsize=9.5, color=colour)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", ls="--", alpha=0.3)
        if len(anom_x) > 0:
            ax.legend(fontsize=8.5, loc="upper right", framealpha=0.9)

    axes[-1].set_xticks(list(range(0, len(wdf), step)))
    axes[-1].set_xticklabels(
        [labels[i] for i in range(0, len(wdf), step)],
        rotation=40, ha="right", fontsize=8.5,
    )

    fig.suptitle(
        f"MAHACEF-200 | Weather Anomaly Detection  "
        f"(|z| > {config.ANOMALY_ZSCORE_THRESHOLD} = Anomaly  |  "
        f"Yellow = Imputed)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    _save(fig, out_dir / "07_weather_anomalies.png")


# ===========================================================================
# 15. EXPORT
# ===========================================================================

def export_phase3_data(wdf: pd.DataFrame, monsoon: pd.DataFrame) -> None:
    """Export monthly weather series and monsoon table to CSV, Excel, metadata."""

    # Build export frame with all computed columns
    export_cols = [
        config.COL_MONTH, "month_label", "year", "month_num",
        "avg_temperature_c", "avg_humidity", "total_rainfall_mm",
        "avg_temperature_c_roll3m", "avg_temperature_c_roll6m",
        "avg_humidity_roll3m",      "avg_humidity_roll6m",
        "total_rainfall_mm_roll3m", "total_rainfall_mm_roll6m",
        "ols_avg_temperature_c", "ols_avg_humidity", "ols_total_rainfall_mm",
        "stl_avg_temperature_c_trend",    "stl_avg_temperature_c_seasonal",
        "stl_avg_humidity_trend",          "stl_avg_humidity_seasonal",
        "stl_total_rainfall_mm_trend",     "stl_total_rainfall_mm_seasonal",
        "z_avg_temperature_c", "z_avg_humidity", "z_total_rainfall_mm",
        "anomaly_avg_temperature_c", "anomaly_avg_humidity",
        "anomaly_total_rainfall_mm",
        "weather_imputed", "imputation_method", "weather_obs_count",
    ]
    out = wdf[[c for c in export_cols if c in wdf.columns]].copy()

    export_csv(out, config.PHASE3_WEATHER_MONTHLY_CSV, logger=logger)

    # Multi-sheet Excel: monthly + monsoon
    with pd.ExcelWriter(str(config.PHASE3_WEATHER_XLSX), engine="openpyxl") as writer:
        out.to_excel(writer, sheet_name="Monthly_Weather_Trend", index=False)
        monsoon.to_excel(writer, sheet_name="Monsoon_Season", index=False)
    logger.info("Excel exported → %s", config.PHASE3_WEATHER_XLSX.name)

    meta_extra = {
        "rows": len(out),
        "months": int(out[config.COL_MONTH].nunique()),
        "observed_months": int((~out["weather_imputed"]).sum()),
        "imputed_months": int(out["weather_imputed"].sum()),
    }
    for path in [config.PHASE3_WEATHER_MONTHLY_CSV, config.PHASE3_WEATHER_XLSX]:
        write_dataset_metadata(
            path, PHASE_LABEL, SCRIPT_NAME,
            source_dataset=config.CLEAN_DATASET_NAME,
            extra=meta_extra,
        )
    logger.info("Metadata sidecars written.")


# ===========================================================================
# 16. REPORT BUILDER
# ===========================================================================

def build_report(
    wdf: pd.DataFrame,
    trends: dict,
    strengths: dict,
    monsoon: pd.DataFrame,
) -> str:
    """Build the standardised Phase 3 Weather Trend Markdown report."""

    n_obs     = int((~wdf["weather_imputed"]).sum())
    n_imputed = int(wdf["weather_imputed"].sum())

    # Trend summary table
    trend_rows = ""
    for col, _, unit, _ in WEATHER_VARS:
        t = trends[col]
        s = strengths[col]
        trend_rows += (
            f"| {t['label']} | {t['slope']:+.4f} {unit}/month "
            f"| {t['r2']:.4f} | {t['p_value']:.6f} "
            f"| {t['direction']} "
            f"| {s['seasonal_strength']:.3f} / {s['trend_strength']:.3f} |\n"
        )

    # Anomaly summary
    anom_rows = ""
    for col, lbl, unit, _ in WEATHER_VARS:
        anoms = wdf[wdf[f"anomaly_{col}"]]["month_label"].tolist()
        anom_rows += (
            f"| {lbl} | {len(anoms)} | "
            + (", ".join(anoms) if anoms else "None") + " |\n"
        )

    # Monsoon table
    mon_rows = ""
    for _, row in monsoon.iterrows():
        imp_flag = "⚠️ Imputed" if row["any_imputed"] else "✅ Observed"
        mon_rows += (
            f"| {int(row['year'])} | {row['total_rainfall_mm']:.1f}mm "
            f"| {row['avg_temperature_c']:.1f}°C "
            f"| {row['avg_humidity']:.1f}% "
            f"| {int(row['n_months'])}/4 months | {imp_flag} |\n"
        )

    objective = (
        "Characterise the **national-level weather patterns** that serve as the "
        "explanatory variable for MAHACEF-200 sales. This phase is *deliberately "
        "weather-only* — sales data is not referenced. The goal is to understand "
        "temperature, humidity, and rainfall on their own terms: seasonal structure, "
        "multi-year trend, anomalies, and the monsoon cycle."
    )

    dataset_used = (
        "| Attribute | Value |\n| --- | --- |\n"
        f"| File | `mahacef200_master_dataset_clean.csv` |\n"
        f"| Rows used | 39 unique billing months (weather deduplicated) |\n"
        f"| Weather variables | avg_temperature_c (°C), avg_humidity (%), "
        "total_rainfall_mm (mm) |\n"
        f"| Observed months | {n_obs} |\n"
        f"| Climatology-imputed months | {n_imputed} |\n"
        f"| Source | National-level weather API (India aggregate) |"
    )

    methodology = (
        "1. **Deduplication**: One row per billing_month extracted (all states "
        "share identical national weather values).\n"
        "2. **Rolling Means**: 3-month and 6-month trailing means per variable.\n"
        "3. **STL Decomposition**: `statsmodels.tsa.seasonal.STL` with "
        "`period=12` (annual), `robust=True` for each weather variable.\n"
        "   - Seasonal Strength = max(0, 1 − Var(R) / Var(S+R))\n"
        "   - Trend Strength    = max(0, 1 − Var(R) / Var(T+R))\n"
        "4. **OLS Trend**: `scipy.stats.linregress` on integer time index.\n"
        f"5. **Anomaly Detection**: Monthly z-score computed; "
        f"|z| > {config.ANOMALY_ZSCORE_THRESHOLD} flagged.\n"
        "6. **Monsoon Analysis**: Jun–Sep subset aggregated per year "
        "(total rainfall, mean temperature, mean humidity)."
    )

    key_findings = (
        "### Weather Trend Summary\n\n"
        "| Variable | OLS Slope | R² | p-value | Direction | Seasonal/Trend Strength |\n"
        "| --- | --- | --- | --- | --- | --- |\n"
        + trend_rows.strip()
        + "\n\n### Anomaly Summary\n\n"
        "| Variable | # Anomalies | Anomalous Months |\n"
        "| --- | --- | --- |\n"
        + anom_rows.strip()
        + "\n\n### Monsoon Season (Jun–Sep) by Year\n\n"
        "| Year | Total Rainfall | Avg Temp | Avg Humidity | Completeness | Data Quality |\n"
        "| --- | --- | --- | --- | --- | --- |\n"
        + mon_rows.strip()
    )

    business_insights = (
        "1. **Seasonal structure dominates weather**: High STL seasonal strength "
        "values confirm that all three weather variables follow a strong, "
        "repeatable annual cycle — exactly the structure that makes seasonal "
        "medicine demand patterns predictable.\n\n"
        "2. **Monsoon as key demand driver**: The Jun–Sep monsoon window produces "
        "the largest humidity and rainfall readings of the year. "
        "If MAHACEF-200 sales respond to upper-respiratory and water-borne "
        "infections, monsoon months should show elevated demand — to be confirmed in Phase 4.\n\n"
        "3. **Temperature peaks in May**: The pre-monsoon heat peak (May) coincides "
        "with the period when antibiotic consumption for heat-related infections "
        "is expected to rise — a testable hypothesis in Phase 4.\n\n"
        "4. **Anomalous months flagged**: Any correlation found in Phase 5 "
        "between weather anomalies and sales spikes (identified in Phase 2) "
        "would strengthen the causal argument for weather-driven demand."
    )

    limitations = (
        "- **National-level proxy**: Weather values are a single national "
        "average; state-level temperature and rainfall vary by 8–12°C and "
        "by orders of magnitude for rainfall. This attenuates state-level "
        "correlations.\n"
        "- **Imputed months (Apr 2023–Apr 2024 + Jun 2026)**: Climatology "
        "imputation preserves the seasonal cycle but cannot capture "
        "year-specific weather events (El Niño effects, heat waves, "
        "drought years). Trend and anomaly results for these months "
        "should be interpreted cautiously.\n"
        "- **39 monthly observations**: Short time-series limits the "
        "statistical power of trend tests. p < 0.05 threshold applied, "
        "but small-sample estimates carry wider confidence intervals.\n"
        "- **No sub-monthly resolution**: Extreme weather events within a "
        "month are averaged out; this may understate weather-sales relationships."
    )

    next_phase = (
        "**Phase 4 — Weather vs Sales Comparison**\n\n"
        "Having characterised sales (Phase 2) and weather (Phase 3) "
        "independently, Phase 4 overlays them visually:\n"
        "- Dual-axis time-series plots (sales + weather)\n"
        "- Rolling average seasonal overlays\n"
        "- Lag scatter plots: sales(t) vs weather(t-k) for k = 0, 1, 2, 3\n\n"
        "State-level analysis focuses on: **Top 5 states by net sales + Assam**."
    )

    return build_phase_report(
        phase_number="3",
        phase_title="Weather Trend Analysis",
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

def run_weather_trend_analysis() -> pd.DataFrame:
    """Execute the complete Phase 3 — Weather Trend Analysis pipeline."""

    logger.info("=" * 60)
    logger.info("PHASE 3 — WEATHER TREND ANALYSIS")
    logger.info("=" * 60)

    ensure_directories(
        config.PHASE3_GRAPHS_DIR,
        config.DATA_DIR,
        config.EXCEL_DIR,
        config.REPORTS_DIR,
        config.LOGS_DIR,
    )

    # ------------------------------------------------------------------ 1
    wdf = load_weather_monthly_series()

    # ------------------------------------------------------------------ 2
    wdf = compute_weather_rolling(wdf)

    # ------------------------------------------------------------------ 3
    wdf, strengths = run_weather_stl(wdf)

    # ------------------------------------------------------------------ 4
    trends = detect_weather_trends(wdf)

    # ------------------------------------------------------------------ 5
    wdf = detect_weather_anomalies(wdf)

    # ------------------------------------------------------------------ 6
    monsoon = compute_monsoon_metrics(wdf)

    # ------------------------------------------------------------------ 7 (graphs)
    out_dir = config.PHASE3_GRAPHS_DIR
    plot_temperature_trend(wdf, out_dir)
    plot_humidity_trend(wdf, out_dir)
    plot_rainfall_trend(wdf, out_dir)
    plot_weather_stl(wdf, out_dir)
    plot_weather_calendar_heatmap(wdf, out_dir)
    plot_monsoon_analysis(wdf, monsoon, out_dir)
    plot_weather_anomalies(wdf, out_dir)

    # ------------------------------------------------------------------ 8
    export_phase3_data(wdf, monsoon)

    # ------------------------------------------------------------------ 9
    report = build_report(wdf, trends, strengths, monsoon)
    write_markdown_report(config.REPORT_WEATHER_TREND, report, logger=logger)

    # Summary
    logger.info("-" * 60)
    logger.info("PHASE 3 COMPLETE")
    logger.info("  Months analysed  : %d", len(wdf))
    for col, label, unit, _ in WEATHER_VARS:
        t = trends[col]
        logger.info(
            "  %-20s : slope=%+.4f %s/month | R²=%.4f | %s",
            label, t["slope"], unit, t["r2"], t["direction"],
        )
    logger.info(
        "  Monsoon years    : %s",
        list(monsoon["year"].astype(int)),
    )
    logger.info("-" * 60)

    return wdf


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    try:
        wdf = run_weather_trend_analysis()
        logger.info("Script finished successfully.")
        sys.exit(0)
    except FileNotFoundError as exc:
        logger.error("File not found: %s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
        sys.exit(99)
