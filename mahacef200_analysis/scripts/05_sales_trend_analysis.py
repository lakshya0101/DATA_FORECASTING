"""
05_sales_trend_analysis.py
===========================
Phase 2 — Sales Trend Analysis

Objective:
    Understand the sales behaviour of MAHACEF-200 *before* introducing any
    weather variables.  This phase is intentionally weather-free so that
    underlying demand patterns can be identified independently.

Outputs
-------
data/phase2_monthly_sales.csv
data/phase2_monthly_sales.metadata.json
excel/Phase2_Sales_Trend.xlsx
graphs/phase2_sales/
    01_monthly_net_sales.png
    02_mom_yoy_growth.png
    03_stl_decomposition.png
    04_spike_detection.png
    05_state_monthly_heatmap.png
reports/Phase2_Sales_Trend.md

Usage
-----
    python mahacef200_analysis/scripts/05_sales_trend_analysis.py
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
import textwrap
import warnings

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
CLR_PRIMARY   = "#1565C0"   # deep blue — main series
CLR_ROLL3     = "#42A5F5"   # sky blue  — 3-month rolling
CLR_ROLL6     = "#00897B"   # teal      — 6-month rolling
CLR_POS       = "#2E7D32"   # green     — positive growth
CLR_NEG       = "#C62828"   # red       — negative growth
CLR_SPIKE     = "#E65100"   # orange    — spikes
CLR_TREND     = "#6A1B9A"   # purple    — OLS trend
CLR_BG        = "#F8F9FA"   # off-white — figure background
SCRIPT_NAME   = "05_sales_trend_analysis.py"
PHASE_LABEL   = "Phase 2 - Sales Trend Analysis"


# ===========================================================================
# 1. DATA LOADING
# ===========================================================================

def load_clean_dataset() -> pd.DataFrame:
    """
    Load mahacef200_master_dataset_clean.csv and return it.

    Raises
    ------
    FileNotFoundError
        If Phase 1.5 has not been run yet.
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
    df[config.COL_STATE] = normalize_state_name(df[config.COL_STATE])
    logger.info("  %d rows × %d cols loaded", *df.shape)
    return df


# ===========================================================================
# 2. NATIONAL MONTHLY AGGREGATION
# ===========================================================================

def build_national_monthly_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate all 24 states into a single India-level monthly time-series.

    Columns produced
    ----------------
    billing_month, month_label, month_date,
    net_sale_amt, gross_sale_amt, net_sale_qty, gross_sale_qty,
    fresh_ret_amt, return_rate_pct (weighted),
    n_states (number of states contributing that month)

    Parameters
    ----------
    df : pd.DataFrame
        Clean 935-row master dataset.

    Returns
    -------
    pd.DataFrame
        39-row national monthly time-series, sorted chronologically.
    """
    logger.info("Building national monthly aggregation …")

    agg = (
        df.groupby(config.COL_MONTH, as_index=False)
          .agg(
              net_sale_amt   =("net_sale_amt",    "sum"),
              gross_sale_amt =("gross_sale_amt",  "sum"),
              net_sale_qty   =("net_sale_qty",    "sum"),
              gross_sale_qty =("gross_sale_qty",  "sum"),
              fresh_ret_amt  =("fresh_ret_amt",   "sum"),
              expiry_amt     =("expiry_amt",       "sum"),
              brkg_amt       =("brkg_amt",         "sum"),
              n_states       =(config.COL_STATE,  "nunique"),
          )
          .sort_values(config.COL_MONTH)
          .reset_index(drop=True)
    )

    # Derived
    agg["return_rate_pct"] = np.where(
        agg["gross_sale_amt"] > 0,
        (agg["fresh_ret_amt"] / agg["gross_sale_amt"] * 100).round(3),
        0.0,
    )
    agg["total_deductions_amt"] = agg["fresh_ret_amt"] + agg["expiry_amt"] + agg["brkg_amt"]
    agg["net_to_gross_ratio"]   = (agg["net_sale_amt"] / agg["gross_sale_amt"] * 100).round(2)

    # Month labels
    agg["month_label"] = billing_month_label(agg[config.COL_MONTH])
    agg["month_date"]  = billing_month_to_date(agg[config.COL_MONTH])
    agg["month_num"]   = agg[config.COL_MONTH] % 100
    agg["year"]        = agg[config.COL_MONTH] // 100

    logger.info("  National monthly series: %d months  (₹%s total net sales)",
                len(agg), format_number(agg["net_sale_amt"].sum()))
    return agg


# ===========================================================================
# 3. ROLLING METRICS
# ===========================================================================

def compute_rolling_metrics(agg: pd.DataFrame) -> pd.DataFrame:
    """
    Add 3-month and 6-month rolling mean columns for net_sale_amt
    and net_sale_qty.

    Uses a minimum of 1 observation (min_periods=1) so the first rows
    are never NaN.

    Parameters
    ----------
    agg : pd.DataFrame
        National monthly aggregation.

    Returns
    -------
    pd.DataFrame
        Same frame with rolling columns added in-place.
    """
    logger.info("Computing rolling means (3M, 6M) …")
    for w in config.ROLLING_WINDOWS:
        agg[f"rolling_{w}m_net_sales"] = (
            agg["net_sale_amt"].rolling(window=w, min_periods=1).mean().round(2)
        )
        agg[f"rolling_{w}m_net_qty"] = (
            agg["net_sale_qty"].rolling(window=w, min_periods=1).mean().round(2)
        )
    return agg


# ===========================================================================
# 4. GROWTH RATES
# ===========================================================================

def compute_growth_rates(agg: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Month-over-Month (MoM) and Year-over-Year (YoY) net sales growth.

    MoM growth  = (current − previous) / |previous| × 100
    YoY growth  = (current − same_month_prior_year) / |same_month_prior_year| × 100

    YoY is only available from month 13 onwards (April 2024 when April 2023
    exists as a reference).

    Parameters
    ----------
    agg : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    logger.info("Computing MoM and YoY growth rates …")

    # MoM
    agg["mom_growth_pct"] = (
        agg["net_sale_amt"]
          .pct_change(periods=1)
          .mul(100)
          .round(2)
    )

    # YoY: merge on same calendar month, prior year
    same_month_prior = agg[["billing_month", "net_sale_amt"]].copy()
    same_month_prior["billing_month"] = same_month_prior["billing_month"] + 100   # +1 year
    same_month_prior.rename(columns={"net_sale_amt": "net_sale_prior_yr"}, inplace=True)

    agg = agg.merge(same_month_prior, on="billing_month", how="left")
    agg["yoy_growth_pct"] = np.where(
        agg["net_sale_prior_yr"].notna() & (agg["net_sale_prior_yr"] != 0),
        ((agg["net_sale_amt"] - agg["net_sale_prior_yr"]) / agg["net_sale_prior_yr"].abs() * 100).round(2),
        np.nan,
    )

    n_yoy = agg["yoy_growth_pct"].notna().sum()
    logger.info("  MoM: %d values  |  YoY: %d values", len(agg) - 1, n_yoy)
    return agg


# ===========================================================================
# 5. STL DECOMPOSITION
# ===========================================================================

def run_stl_decomposition(agg: pd.DataFrame) -> tuple[pd.DataFrame, object]:
    """
    Run STL (Seasonal-Trend decomposition using Loess) on the national
    net sales series with period = 12 (annual seasonality).

    Parameters
    ----------
    agg : pd.DataFrame
        National monthly series with net_sale_amt column.

    Returns
    -------
    tuple[pd.DataFrame, STLResults]
        agg with stl_trend/seasonal/residual columns added, and raw STL result.

    Notes
    -----
    STL requires at least 2 × period observations.  With 39 months and
    period=12 (→ threshold 24), this is satisfied.
    """
    logger.info("Running STL decomposition (period=%d) …", config.STL_PERIOD)

    ts = pd.Series(
        agg["net_sale_amt"].values,
        index=pd.date_range(
            start=agg["month_date"].iloc[0],
            periods=len(agg),
            freq="MS",
        ),
    )

    stl = STL(ts, period=config.STL_PERIOD, robust=True)
    result = stl.fit()

    agg["stl_trend"]    = result.trend.values
    agg["stl_seasonal"] = result.seasonal.values
    agg["stl_residual"] = result.resid.values

    # Seasonal strength: max(0, 1 − Var(R) / Var(S+R))
    var_resid   = np.var(result.resid)
    var_sr      = np.var(result.seasonal + result.resid)
    seasonal_strength = max(0.0, 1 - var_resid / var_sr) if var_sr > 0 else 0.0

    var_trend   = np.var(result.trend)
    var_tr      = np.var(result.trend + result.resid)
    trend_strength = max(0.0, 1 - var_resid / var_tr) if var_tr > 0 else 0.0

    logger.info("  STL seasonal strength: %.3f  |  trend strength: %.3f",
                seasonal_strength, trend_strength)
    return agg, result, seasonal_strength, trend_strength


# ===========================================================================
# 6. TREND DETECTION
# ===========================================================================

def detect_trend(agg: pd.DataFrame) -> dict:
    """
    Fit a linear OLS trend to the net sales series.

    Returns slope (₹/month), R², and p-value.

    Parameters
    ----------
    agg : pd.DataFrame

    Returns
    -------
    dict
        {"slope": float, "r2": float, "p_value": float,
         "intercept": float, "direction": str}
    """
    logger.info("Detecting linear trend …")
    x = np.arange(len(agg), dtype=float)
    y = agg["net_sale_amt"].values

    slope, intercept, r, p, se = stats.linregress(x, y)
    r2 = r ** 2
    direction = (
        "Strong upward"   if slope > 0 and p < 0.01 else
        "Moderate upward" if slope > 0 and p < 0.05 else
        "Weak upward"     if slope > 0 else
        "Strong downward" if slope < 0 and p < 0.01 else
        "Moderate downward" if slope < 0 and p < 0.05 else
        "Weak downward"
    )

    # Store fitted trend
    agg["ols_trend"] = intercept + slope * x

    logger.info("  Slope: ₹%.0f/month  |  R²: %.4f  |  p-value: %.4f  |  %s",
                slope, r2, p, direction)
    return {
        "slope":     round(slope, 2),
        "intercept": round(intercept, 2),
        "r2":        round(r2, 4),
        "p_value":   round(p, 6),
        "direction": direction,
    }


# ===========================================================================
# 7. SPIKE DETECTION
# ===========================================================================

def detect_spikes(agg: pd.DataFrame) -> pd.DataFrame:
    """
    Flag months whose net sales deviate more than SPIKE_STD_FACTOR × std
    from the rolling 6-month mean (adaptive threshold — better than a
    global mean for a series with strong trend).

    Parameters
    ----------
    agg : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        agg with 'is_spike' (bool) and 'spike_zscore' (float) columns.
    """
    logger.info("Detecting sales spikes (±%.1f σ) …", config.SPIKE_STD_FACTOR)

    roll_mean = agg["net_sale_amt"].rolling(12, min_periods=6, center=True).mean()
    roll_std  = agg["net_sale_amt"].rolling(12, min_periods=6, center=True).std()
    roll_mean.fillna(agg["net_sale_amt"].mean(), inplace=True)
    roll_std.fillna(agg["net_sale_amt"].std(), inplace=True)

    agg["spike_zscore"] = ((agg["net_sale_amt"] - roll_mean) / roll_std).round(3)
    agg["is_spike"]     = agg["spike_zscore"].abs() > config.SPIKE_STD_FACTOR

    n_spikes = int(agg["is_spike"].sum())
    logger.info("  %d spike month(s) detected: %s",
                n_spikes,
                list(agg[agg["is_spike"]]["month_label"]))
    return agg


# ===========================================================================
# 8. GRAPH 1 — Monthly Net Sales + Rolling Means
# ===========================================================================

def plot_monthly_net_sales(agg: pd.DataFrame, out_dir: Path) -> None:
    """
    Line chart: actual net sales, 3M rolling mean, 6M rolling mean,
    OLS trend line, with area fill and month annotations.
    """
    logger.info("Plotting Graph 1: Monthly Net Sales …")

    fig, ax = plt.subplots(figsize=(18, 7))
    fig.patch.set_facecolor(CLR_BG)
    ax.set_facecolor(CLR_BG)

    x  = range(len(agg))
    y  = agg["net_sale_amt"].values / 1e6   # convert to ₹ millions

    # Fill
    ax.fill_between(x, y, alpha=0.08, color=CLR_PRIMARY)

    # Actual line
    ax.plot(x, y, "-o", color=CLR_PRIMARY, linewidth=2.2, markersize=4.5,
            label="Monthly Net Sales", zorder=4)

    # Rolling 3M
    ax.plot(x, agg["rolling_3m_net_sales"].values / 1e6,
            "--", color=CLR_ROLL3, linewidth=1.8, label="3-Month Rolling Mean", zorder=3)

    # Rolling 6M
    ax.plot(x, agg["rolling_6m_net_sales"].values / 1e6,
            "-.", color=CLR_ROLL6, linewidth=1.8, label="6-Month Rolling Mean", zorder=3)

    # OLS trend
    ax.plot(x, agg["ols_trend"].values / 1e6,
            ":", color=CLR_TREND, linewidth=2.0, label="OLS Trend", zorder=2, alpha=0.85)

    # Annotate peak and trough
    peak_idx  = int(agg["net_sale_amt"].idxmax())
    trough_idx = int(agg["net_sale_amt"].idxmin())
    for idx, label_str, colour in [(peak_idx, "Peak", "#1B5E20"),
                                    (trough_idx, "Trough", "#B71C1C")]:
        ax.annotate(
            f"{label_str}\n{agg['month_label'].iloc[idx]}\n₹{y[idx]:.1f}M",
            xy=(idx, y[idx]),
            xytext=(idx + 1.2, y[idx] + (0.15 if idx == peak_idx else -0.15) * max(y)),
            fontsize=8.5, color=colour, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=colour, lw=1.2),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=colour, alpha=0.85),
        )

    # X-axis formatting
    step = max(1, len(agg) // 18)
    tick_pos    = list(range(0, len(agg), step))
    tick_labels = [agg["month_label"].iloc[i] for i in tick_pos]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, rotation=40, ha="right", fontsize=8.5)

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"₹{v:.0f}M"))
    ax.set_ylabel("Net Sales (₹ Millions)", fontsize=11)
    ax.set_xlabel("Billing Month", fontsize=10)
    ax.set_title("MAHACEF-200 | National Monthly Net Sales with Trend & Rolling Means",
                 fontsize=14, fontweight="bold", pad=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    _save(fig, out_dir / "01_monthly_net_sales.png")


# ===========================================================================
# 9. GRAPH 2 — MoM & YoY Growth
# ===========================================================================

def plot_growth_rates(agg: pd.DataFrame, out_dir: Path) -> None:
    """
    Two-panel bar chart: top = MoM growth (%), bottom = YoY growth (%).
    Bars are coloured green (positive) or red (negative).
    """
    logger.info("Plotting Graph 2: Growth Rates …")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 11), sharex=False)
    fig.patch.set_facecolor(CLR_BG)

    def _growth_bars(ax: plt.Axes, series: pd.Series, labels: list,
                     title: str, ylabel: str) -> None:
        valid_mask = series.notna()
        vals   = series[valid_mask].values
        xlabs  = [labels[i] for i, m in enumerate(valid_mask) if m]
        xpos   = range(len(vals))
        colors = [CLR_POS if v >= 0 else CLR_NEG for v in vals]

        bars = ax.bar(xpos, vals, color=colors, edgecolor="white", width=0.7)
        ax.axhline(0, color="#333333", linewidth=0.8)

        for bar, val in zip(bars, vals):
            va = "bottom" if val >= 0 else "top"
            offset = 0.15 if val >= 0 else -0.15
            ax.text(bar.get_x() + bar.get_width() / 2,
                    val + offset * (max(abs(vals)) or 1),
                    f"{val:+.1f}%", ha="center", va=va, fontsize=7.5, fontweight="bold")

        ax.set_xticks(list(xpos))
        ax.set_xticklabels(xlabs, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_facecolor(CLR_BG)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.35)

        pos_patch = mpatches.Patch(color=CLR_POS, label="Positive growth")
        neg_patch = mpatches.Patch(color=CLR_NEG, label="Negative growth")
        ax.legend(handles=[pos_patch, neg_patch], fontsize=8, loc="upper right")

    _growth_bars(ax1, agg["mom_growth_pct"],  list(agg["month_label"]),
                 "Month-over-Month (MoM) Net Sales Growth (%)", "MoM Growth (%)")
    _growth_bars(ax2, agg["yoy_growth_pct"],  list(agg["month_label"]),
                 "Year-over-Year (YoY) Net Sales Growth (%)", "YoY Growth (%)")

    fig.suptitle("MAHACEF-200 | National Net Sales Growth Rates",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig, out_dir / "02_mom_yoy_growth.png")


# ===========================================================================
# 10. GRAPH 3 — STL Decomposition
# ===========================================================================

def plot_stl_decomposition(agg: pd.DataFrame, stl_result: object,
                            s_strength: float, t_strength: float,
                            out_dir: Path) -> None:
    """
    Four-panel STL decomposition plot: Observed, Trend, Seasonal, Residual.
    """
    logger.info("Plotting Graph 3: STL Decomposition …")

    fig, axes = plt.subplots(4, 1, figsize=(17, 13), sharex=True)
    fig.patch.set_facecolor(CLR_BG)

    x      = range(len(agg))
    labels = list(agg["month_label"])
    step   = max(1, len(agg) // 18)

    panels = [
        (agg["net_sale_amt"].values / 1e6,    "Observed (₹M)",  CLR_PRIMARY, "o-"),
        (agg["stl_trend"].values / 1e6,       "Trend (₹M)",     CLR_TREND,   "-"),
        (agg["stl_seasonal"].values / 1e6,    "Seasonal (₹M)",  "#00897B",   "-"),
        (agg["stl_residual"].values / 1e6,    "Residual (₹M)",  "#EF6C00",   "o"),
    ]

    for ax, (vals, ylabel, colour, style) in zip(axes, panels):
        ax.set_facecolor(CLR_BG)
        if "o" in style:
            ax.plot(x, vals, style, color=colour, linewidth=1.8, markersize=3.5, alpha=0.9)
        else:
            ax.fill_between(x, vals, alpha=0.12, color=colour)
            ax.plot(x, vals, style, color=colour, linewidth=2.0)
        ax.axhline(0, color="#999999", linewidth=0.6, linestyle=":")
        ax.set_ylabel(ylabel, fontsize=9.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}"))

    axes[-1].set_xticks(list(range(0, len(agg), step)))
    axes[-1].set_xticklabels(
        [labels[i] for i in range(0, len(agg), step)],
        rotation=40, ha="right", fontsize=8.5
    )

    fig.suptitle(
        f"MAHACEF-200 | STL Decomposition  "
        f"(Seasonal Strength: {s_strength:.3f}  |  Trend Strength: {t_strength:.3f})",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    _save(fig, out_dir / "03_stl_decomposition.png")


# ===========================================================================
# 11. GRAPH 4 — Spike Detection
# ===========================================================================

def plot_spike_detection(agg: pd.DataFrame, out_dir: Path) -> None:
    """
    Line chart with spike months highlighted in orange and annotated.
    Background band shows the ±1.5σ adaptive threshold.
    """
    logger.info("Plotting Graph 4: Spike Detection …")

    fig, ax = plt.subplots(figsize=(18, 7))
    fig.patch.set_facecolor(CLR_BG)
    ax.set_facecolor(CLR_BG)

    x = range(len(agg))
    y = agg["net_sale_amt"].values / 1e6

    # Main line
    ax.fill_between(x, y, alpha=0.06, color=CLR_PRIMARY)
    ax.plot(x, y, "-o", color=CLR_PRIMARY, linewidth=2.0,
            markersize=4, label="Net Sales", zorder=3)

    # Highlight spikes
    spike_mask = agg["is_spike"].values
    ax.scatter(
        [i for i, m in enumerate(spike_mask) if m],
        [y[i] for i, m in enumerate(spike_mask) if m],
        color=CLR_SPIKE, s=120, zorder=5, label=f"Spike (|z| > {config.SPIKE_STD_FACTOR})",
        edgecolors="white", linewidths=1.2,
    )

    # Annotate each spike
    for i, (is_spk, zscore) in enumerate(zip(agg["is_spike"], agg["spike_zscore"])):
        if is_spk:
            ax.annotate(
                f"{agg['month_label'].iloc[i]}\nz={zscore:+.2f}",
                xy=(i, y[i]),
                xytext=(i + 0.5, y[i] + 0.06 * max(y) * (1 if zscore > 0 else -1)),
                fontsize=8, color=CLR_SPIKE, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=CLR_SPIKE, lw=1.1),
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor=CLR_SPIKE, alpha=0.85),
            )

    step = max(1, len(agg) // 18)
    ax.set_xticks(list(range(0, len(agg), step)))
    ax.set_xticklabels(
        [agg["month_label"].iloc[i] for i in range(0, len(agg), step)],
        rotation=40, ha="right", fontsize=8.5,
    )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"₹{v:.0f}M"))
    ax.set_ylabel("Net Sales (₹ Millions)", fontsize=11)
    ax.set_xlabel("Billing Month", fontsize=10)
    ax.set_title(
        f"MAHACEF-200 | Spike Detection  (Threshold: rolling mean ± {config.SPIKE_STD_FACTOR}σ)",
        fontsize=14, fontweight="bold", pad=14,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(fontsize=9, framealpha=0.9)

    plt.tight_layout()
    _save(fig, out_dir / "04_spike_detection.png")


# ===========================================================================
# 12. GRAPH 5 — State × Month Heatmap
# ===========================================================================

def plot_state_monthly_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Heatmap of net_sale_amt with states on y-axis and months on x-axis.
    Colour intensity reflects sales volume — reveals geographic-temporal
    demand patterns at a glance.
    """
    logger.info("Plotting Graph 5: State × Month Heatmap …")

    # Pivot: state (rows) × billing_month (cols)
    pivot = df.pivot_table(
        index=config.COL_STATE,
        columns=config.COL_MONTH,
        values="net_sale_amt",
        aggfunc="sum",
    )
    pivot.sort_index(axis=1, inplace=True)

    # Row-normalise so we can see seasonal pattern within each state
    pivot_norm = pivot.div(pivot.max(axis=1), axis=0)

    n_states = len(pivot)
    n_months = len(pivot.columns)
    col_labels = billing_month_label(pd.Series(pivot.columns.tolist())).tolist()

    fig, axes = plt.subplots(1, 2, figsize=(24, max(10, n_states * 0.48)),
                              gridspec_kw={"width_ratios": [1.2, 1]})
    fig.patch.set_facecolor(CLR_BG)

    def _draw_heatmap(ax: plt.Axes, data: pd.DataFrame,
                      title: str, fmt_fn) -> None:
        ax.set_facecolor(CLR_BG)
        im = ax.imshow(
            data.values, aspect="auto",
            cmap="YlOrRd", interpolation="nearest",
        )
        # State labels (y-axis)
        ax.set_yticks(range(n_states))
        ax.set_yticklabels(data.index, fontsize=8.5)
        # Month labels (x-axis) — every 3rd to avoid crowding
        step = 3
        ax.set_xticks(list(range(0, n_months, step)))
        ax.set_xticklabels(col_labels[::step], rotation=55, ha="right", fontsize=8)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    _draw_heatmap(axes[0], pivot / 1e6,
                  "Absolute Net Sales (₹M)", lambda v: f"{v:.0f}")
    _draw_heatmap(axes[1], pivot_norm,
                  "Row-Normalised (0–1 within state)", lambda v: f"{v:.2f}")

    fig.suptitle(
        "MAHACEF-200 | State × Month Net Sales Heatmap",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    _save(fig, out_dir / "05_state_monthly_heatmap.png")


# ===========================================================================
# 13. EXPORT DATA
# ===========================================================================

def export_phase2_data(agg: pd.DataFrame) -> None:
    """Export the national monthly series to CSV and Excel with metadata sidecars."""

    export_cols = [
        "billing_month", "month_label", "year", "month_num",
        "net_sale_amt", "gross_sale_amt", "net_sale_qty", "gross_sale_qty",
        "fresh_ret_amt", "total_deductions_amt", "return_rate_pct",
        "net_to_gross_ratio", "n_states",
        "rolling_3m_net_sales", "rolling_6m_net_sales",
        "rolling_3m_net_qty", "rolling_6m_net_qty",
        "mom_growth_pct", "yoy_growth_pct",
        "stl_trend", "stl_seasonal", "stl_residual",
        "ols_trend", "is_spike", "spike_zscore",
    ]
    out = agg[[c for c in export_cols if c in agg.columns]].copy()

    export_csv(out, config.PHASE2_MONTHLY_SALES_CSV, logger=logger)
    export_excel(out, config.PHASE2_SALES_XLSX, sheet_name="Monthly_Sales_Trend", logger=logger)

    meta_extra = {
        "rows": len(out),
        "months": int(out["billing_month"].nunique()),
        "total_net_sales_inr": float(out["net_sale_amt"].sum()),
    }
    for path in [config.PHASE2_MONTHLY_SALES_CSV, config.PHASE2_SALES_XLSX]:
        write_dataset_metadata(
            path, PHASE_LABEL, SCRIPT_NAME,
            source_dataset=config.CLEAN_DATASET_NAME,
            extra=meta_extra,
        )
    logger.info("Metadata sidecars written.")


# ===========================================================================
# 14. REPORT BUILDER
# ===========================================================================

def build_report(agg: pd.DataFrame, trend: dict,
                 s_strength: float, t_strength: float) -> str:
    """Build the standardized Phase 2 Sales Trend Markdown report."""

    # Key statistics
    total_net   = agg["net_sale_amt"].sum()
    mean_month  = agg["net_sale_amt"].mean()
    peak_row    = agg.loc[agg["net_sale_amt"].idxmax()]
    trough_row  = agg.loc[agg["net_sale_amt"].idxmin()]
    n_spikes    = int(agg["is_spike"].sum())
    spike_months = ", ".join(agg[agg["is_spike"]]["month_label"].tolist()) or "None"

    avg_mom_pos = agg[agg["mom_growth_pct"] > 0]["mom_growth_pct"].mean()
    avg_mom_neg = agg[agg["mom_growth_pct"] < 0]["mom_growth_pct"].mean()
    yoy_valid   = agg["yoy_growth_pct"].dropna()
    avg_yoy     = yoy_valid.mean() if len(yoy_valid) else float("nan")

    # Seasonal index: which calendar months are above/below average
    agg["season_idx"] = agg["net_sale_amt"] / mean_month
    season_summary = (
        agg.groupby("month_num")["season_idx"]
           .mean()
           .reset_index()
    )
    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    season_table_rows = ""
    for _, row in season_summary.iterrows():
        tag = "🟢 Above avg" if row["season_idx"] > 1.0 else "🔴 Below avg"
        season_table_rows += (
            f"| {month_names[int(row['month_num'])]} | "
            f"{row['season_idx']:.3f} | {tag} |\n"
        )

    # Findings table
    findings_table = f"""
| Metric | Value |
| --- | --- |
| Study Period | {agg["month_label"].iloc[0]} → {agg["month_label"].iloc[-1]} |
| Total Months | {len(agg)} |
| Total Net Sales | ₹{format_number(total_net)} |
| Avg Monthly Net Sales | ₹{format_number(mean_month)} |
| Peak Month | {peak_row["month_label"]} (₹{format_number(peak_row["net_sale_amt"])}) |
| Trough Month | {trough_row["month_label"]} (₹{format_number(trough_row["net_sale_amt"])}) |
| OLS Trend Slope | ₹{format_number(trend["slope"])}/month ({trend["direction"]}) |
| Trend R² | {trend["r2"]:.4f} |
| Trend p-value | {trend["p_value"]:.6f} |
| STL Seasonal Strength | {s_strength:.3f} |
| STL Trend Strength | {t_strength:.3f} |
| Avg MoM Growth (positive months) | +{avg_mom_pos:.2f}% |
| Avg MoM Growth (negative months) | {avg_mom_neg:.2f}% |
| Avg YoY Growth | {avg_yoy:+.2f}% |
| Spike Months Detected | {n_spikes} ({spike_months}) |
"""

    objective = (
        "Understand the sales behaviour of **MAHACEF-200** independently, before "
        "introducing any weather variables. This phase establishes the demand baseline — "
        "trends, seasonality, growth rates, and anomalies — that will be used as the "
        "reference point for all weather-driven analyses in Phases 3–9."
    )

    dataset_used = (
        "| Attribute | Value |\n| --- | --- |\n"
        f"| File | `mahacef200_master_dataset_clean.csv` |\n"
        f"| Rows used | 935 (all 24 states, 39 months) |\n"
        f"| Aggregation | India-level monthly sum (24 states collapsed) |\n"
        f"| Weather columns | **Excluded** from this phase |\n"
        f"| Imputed rows | 336 (weather_imputed=True, sales values are original) |"
    )

    methodology = (
        "1. **Aggregation**: All 24 states summed to India-level monthly series (39 data points).\n"
        "2. **Rolling Means**: 3-month and 6-month trailing means using `min_periods=1`.\n"
        "3. **MoM Growth**: `pct_change(1) × 100`.\n"
        "4. **YoY Growth**: Current month vs same calendar month 12 months prior (available from month 13 onwards, "
        f"yielding {int(agg['yoy_growth_pct'].notna().sum())} data points).\n"
        f"5. **STL Decomposition**: `statsmodels.tsa.seasonal.STL` with `period=12` (annual), `robust=True` "
        "(Huber weights reduce outlier influence on trend/seasonal estimates).\n"
        "6. **OLS Trend**: `scipy.stats.linregress` on raw sales vs integer time index.\n"
        f"7. **Spike Detection**: Adaptive threshold — rolling 12-month window mean ± {config.SPIKE_STD_FACTOR}σ "
        "(z-score based, more robust than a fixed global threshold for a trending series)."
    )

    key_findings = findings_table.strip() + "\n\n### Seasonal Index by Calendar Month\n\n" + \
        "| Month | Seasonal Index | Signal |\n| --- | --- | --- |\n" + season_table_rows.strip()

    business_insights = (
        "1. **Growth trajectory**: An OLS slope of "
        f"₹{format_number(trend['slope'])}/month with "
        f"R²={trend['r2']:.3f} ({trend['direction']} trend) indicates that MAHACEF-200 "
        "demand has a measurable directional pattern — a critical input for long-term sales planning.\n\n"
        "2. **Seasonal demand structure**: STL seasonal strength of "
        f"{s_strength:.3f} (>0.6 = strong, >0.4 = moderate) confirms that "
        "calendar-month effects drive meaningful variance in sales. Territory managers "
        "should align inventory build-up with the high-seasonal-index months identified above.\n\n"
        "3. **Spike months**: The "
        f"{n_spikes} spike month(s) ({spike_months}) represent "
        "demand events worth investigating — potential disease outbreaks, stockist "
        "bulk orders, or promotional campaigns. These months will be flagged in "
        "subsequent correlation analyses to avoid spurious weather-sales associations.\n\n"
        "4. **YoY growth**: An average YoY of "
        f"{avg_yoy:+.2f}% across {int(agg['yoy_growth_pct'].notna().sum())} months "
        "sets the organic growth expectation baseline. Months significantly above or "
        "below this baseline are candidates for weather-driven demand explanation in Phase 4."
    )

    limitations = (
        "- **National aggregation**: State-level heterogeneity is collapsed in this phase. "
        "High-volume states (U.P., Maharashtra) dominate the national signal.\n"
        "- **Short history**: 39 months provides two full seasonal cycles. STL estimates, "
        "particularly the trend, carry wider uncertainty than they would with 5+ years of data.\n"
        "- **No causal inference**: Growth rates and trend direction are descriptive. "
        "Causal attribution to weather, market events, or policy changes requires Phase 4–7 analysis.\n"
        "- **Imputed weather rows**: Sales values in the 336 imputed rows are original "
        "(imputation only affected weather columns), so sales trend analysis is unaffected."
    )

    next_phase = (
        "**Phase 3 — Weather Trend Analysis**\n\n"
        "Having established the sales baseline, Phase 3 analyses weather patterns "
        "independently — temperature, humidity, and rainfall trends, seasonal "
        "decomposition, anomaly detection, and monsoon analysis. "
        "This separation ensures that weather and sales patterns are each understood "
        "on their own terms before a causal relationship is hypothesised in Phase 4."
    )

    return build_phase_report(
        phase_number="2",
        phase_title="Sales Trend Analysis",
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
# SHARED UTILITY
# ===========================================================================

def _save(fig: plt.Figure, path: Path) -> None:
    """Save figure, close it, and log."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved → %s", path.name)


# ===========================================================================
# MAIN PIPELINE
# ===========================================================================

def run_sales_trend_analysis() -> pd.DataFrame:
    """
    Execute the complete Phase 2 — Sales Trend Analysis pipeline.

    Returns
    -------
    pd.DataFrame
        National monthly aggregation with all computed metrics.
    """
    logger.info("=" * 60)
    logger.info("PHASE 2 — SALES TREND ANALYSIS")
    logger.info("=" * 60)

    ensure_directories(
        config.PHASE2_GRAPHS_DIR,
        config.DATA_DIR,
        config.EXCEL_DIR,
        config.REPORTS_DIR,
        config.LOGS_DIR,
    )

    # ------------------------------------------------------------------ 1
    df = load_clean_dataset()

    # ------------------------------------------------------------------ 2
    agg = build_national_monthly_series(df)

    # ------------------------------------------------------------------ 3
    agg = compute_rolling_metrics(agg)

    # ------------------------------------------------------------------ 4
    agg = compute_growth_rates(agg)

    # ------------------------------------------------------------------ 5
    agg, stl_result, s_strength, t_strength = run_stl_decomposition(agg)

    # ------------------------------------------------------------------ 6
    trend = detect_trend(agg)

    # ------------------------------------------------------------------ 7
    agg = detect_spikes(agg)

    # ------------------------------------------------------------------ 8 – 12 (graphs)
    out_dir = config.PHASE2_GRAPHS_DIR
    plot_monthly_net_sales(agg, out_dir)
    plot_growth_rates(agg, out_dir)
    plot_stl_decomposition(agg, stl_result, s_strength, t_strength, out_dir)
    plot_spike_detection(agg, out_dir)
    plot_state_monthly_heatmap(df, out_dir)

    # ------------------------------------------------------------------ 13
    export_phase2_data(agg)

    # ------------------------------------------------------------------ 14
    report = build_report(agg, trend, s_strength, t_strength)
    write_markdown_report(config.REPORT_SALES_TREND, report, logger=logger)

    # Console summary
    logger.info("-" * 60)
    logger.info("PHASE 2 COMPLETE")
    logger.info("  Months analysed    : %d", len(agg))
    logger.info("  Total net sales    : ₹%s", format_number(agg["net_sale_amt"].sum()))
    logger.info("  Trend direction    : %s (R²=%.4f)", trend["direction"], trend["r2"])
    logger.info("  STL seasonal str.  : %.3f", s_strength)
    logger.info("  Spikes detected    : %d", int(agg["is_spike"].sum()))
    logger.info("-" * 60)

    return agg


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    try:
        df = run_sales_trend_analysis()
        logger.info("Script finished successfully.")
        sys.exit(0)
    except FileNotFoundError as exc:
        logger.error("File not found: %s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
        sys.exit(99)
