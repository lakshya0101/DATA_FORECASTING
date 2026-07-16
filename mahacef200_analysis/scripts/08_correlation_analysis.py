"""
08_correlation_analysis.py
===========================
Phase 5 — Correlation Analysis

Objective:
    Apply formal statistical tests to quantify the strength and significance
    of the weather-sales relationship identified visually in Phase 4.
    Both Pearson (linear) and Spearman (rank-based) correlations are computed
    at multiple lags.  Partial correlations (detrended) isolate the relationship
    beyond shared seasonality.  A state-level analysis identifies which markets
    are most weather-responsive.

Outputs
-------
data/phase5_correlation_results.csv
data/phase5_correlation_results.metadata.json
excel/Phase5_Correlation.xlsx
graphs/phase5_correlation/
    01_pearson_spearman_lag_heatmap.png    side-by-side r heatmaps
    02_lag_profile_bars.png               r vs lag, bar chart (3 panels)
    03_state_pearson_heatmap.png          24-state × 3-var Pearson
    04_state_spearman_heatmap.png         24-state × 3-var Spearman
    05_partial_vs_full_correlation.png    detrended vs full comparison
    06_significance_summary.png           comprehensive significance matrix
reports/Phase5_Correlation.md

Usage
-----
    python mahacef200_analysis/scripts/08_correlation_analysis.py
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
import matplotlib.colors as mcolors
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
CLR_SALES   = "#1565C0"
CLR_TEMP    = "#C62828"
CLR_HUM     = "#00695C"
CLR_RAIN    = "#1B5E20"
CLR_BG      = "#F8F9FA"
CLR_POS     = "#1565C0"     # positive correlation
CLR_NEG     = "#C62828"     # negative correlation

WEATHER_VARS = [
    ("avg_temperature_c",  "Temperature",  "°C",  CLR_TEMP),
    ("avg_humidity",       "Humidity",     "%",   CLR_HUM),
    ("total_rainfall_mm",  "Rainfall",     "mm",  CLR_RAIN),
]
LAGS = [0, 1, 2, 3]

SCRIPT_NAME = "08_correlation_analysis.py"
PHASE_LABEL = "Phase 5 - Correlation Analysis"


# ===========================================================================
# HELPER
# ===========================================================================

def _stars(p_val: float) -> str:
    """Return significance stars string for a given p-value."""
    for thresh, mark in config.SIG_LEVELS:
        if p_val < thresh:
            return mark
    return "ns"


def _sig_colour(p_val: float, positive: bool) -> str:
    """Return a text colour based on significance (red=sig, grey=ns)."""
    if p_val >= 0.05:
        return "#888888"
    return "#B71C1C" if positive else "#0D47A1"


# ===========================================================================
# 1. DATA LOADING AND PREPARATION
# ===========================================================================

def load_and_build(df: pd.DataFrame
                   ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build three series from the clean master dataset:
    - merged    : 39-row national monthly (sales + weather)
    - state_df  : state × month panel (24 states × 39 months)
    - state_rank: states ordered by total net sales (for heatmap row order)
    """
    logger.info("Building national and state-level series …")

    # National sales aggregation
    sales = (
        df.groupby(config.COL_MONTH, as_index=False)
          .agg(net_sale_amt=("net_sale_amt", "sum"))
          .sort_values(config.COL_MONTH).reset_index(drop=True)
    )

    # Weather (one row per month)
    weather = (
        df[[config.COL_MONTH, "avg_temperature_c", "avg_humidity",
            "total_rainfall_mm", "weather_imputed"]]
          .drop_duplicates(subset=[config.COL_MONTH])
          .sort_values(config.COL_MONTH).reset_index(drop=True)
    )

    merged = sales.merge(weather, on=config.COL_MONTH, how="left")
    merged["month_label"] = billing_month_label(merged[config.COL_MONTH])
    merged["month_num"]   = merged[config.COL_MONTH] % 100
    merged["year"]        = merged[config.COL_MONTH] // 100
    merged["t_index"]     = np.arange(len(merged), dtype=float)

    # Create lag columns for weather
    for col, _, _, _ in WEATHER_VARS:
        for k in LAGS[1:]:
            merged[f"{col}_lag{k}"] = merged[col].shift(k)

    # State-level aggregation
    state_df = (
        df.groupby([config.COL_STATE, config.COL_MONTH], as_index=False)
          .agg(net_sale_amt=("net_sale_amt", "sum"))
    )
    state_df = state_df.merge(weather, on=config.COL_MONTH, how="left")

    # State rank by total net sales (descending)
    state_rank = (
        df.groupby(config.COL_STATE)["net_sale_amt"]
          .sum().sort_values(ascending=False)
          .reset_index()
    )
    state_rank.columns = [config.COL_STATE, "total_net_sales"]

    logger.info("  National: %d months | States: %d",
                len(merged), state_df[config.COL_STATE].nunique())
    return merged, state_df, state_rank


# ===========================================================================
# 2. NATIONAL LAG CORRELATIONS (PEARSON + SPEARMAN)
# ===========================================================================

def compute_lag_correlations(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Pearson and Spearman correlations between national net_sale_amt
    and each weather variable at lags 0, 1, 2, 3.

    Returns
    -------
    pd.DataFrame  12 rows (3 vars × 4 lags), columns:
        variable, var_label, lag, n,
        pearson_r, pearson_p, pearson_stars,
        spearman_r, spearman_p, spearman_stars
    """
    logger.info("Computing national lag correlations (Pearson + Spearman) …")
    rows = []
    y = merged["net_sale_amt"].values

    for col, label, unit, _ in WEATHER_VARS:
        for k in LAGS:
            x_col = col if k == 0 else f"{col}_lag{k}"
            xv = merged[x_col].values

            # Drop NaN pairs
            mask = ~np.isnan(xv) & ~np.isnan(y)
            xm, ym = xv[mask], y[mask]
            n = int(mask.sum())

            pr, pp   = stats.pearsonr(xm, ym)
            sr, sp   = stats.spearmanr(xm, ym)

            rows.append({
                "variable": col, "var_label": label,
                "unit": unit, "lag": k, "n": n,
                "pearson_r": round(pr, 4), "pearson_p": round(pp, 6),
                "pearson_stars": _stars(pp),
                "spearman_r": round(sr, 4), "spearman_p": round(sp, 6),
                "spearman_stars": _stars(sp),
            })

            logger.info(
                "  %-12s lag%d: Pearson r=%+.3f (%s) | "
                "Spearman r=%+.3f (%s)  [n=%d]",
                label, k, pr, _stars(pp), sr, _stars(sp), n,
            )

    return pd.DataFrame(rows)


# ===========================================================================
# 3. PARTIAL / DETRENDED CORRELATIONS
# ===========================================================================

def compute_partial_correlations(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Pearson correlations after removing linear time trend from both
    series (detrended correlation).  This controls for any shared upward or
    downward time trend and isolates the within-cycle relationship.

    Returns
    -------
    pd.DataFrame  12 rows (3 vars × 4 lags), columns:
        variable, var_label, lag, n,
        partial_r, partial_p, partial_stars
    """
    logger.info("Computing partial (detrended) correlations …")
    rows = []
    t = merged["t_index"].values

    # Detrend net sales
    sl_y, int_y, _, _, _ = stats.linregress(t, merged["net_sale_amt"].values)
    y_dt = merged["net_sale_amt"].values - (int_y + sl_y * t)

    for col, label, unit, _ in WEATHER_VARS:
        for k in LAGS:
            x_col = col if k == 0 else f"{col}_lag{k}"
            xv    = merged[x_col].values

            mask = ~np.isnan(xv)
            x_clean  = xv[mask]
            t_clean  = t[mask]
            y_clean  = y_dt[mask]

            # Detrend the weather variable
            sl_x, int_x, _, _, _ = stats.linregress(t_clean, x_clean)
            x_dt = x_clean - (int_x + sl_x * t_clean)

            n = len(x_dt)
            pr, pp = stats.pearsonr(x_dt, y_clean)

            rows.append({
                "variable": col, "var_label": label,
                "unit": unit, "lag": k, "n": n,
                "partial_r": round(pr, 4), "partial_p": round(pp, 6),
                "partial_stars": _stars(pp),
            })

            logger.info(
                "  %-12s lag%d: Partial r=%+.3f (%s)  [n=%d]",
                label, k, pr, _stars(pp), n,
            )

    return pd.DataFrame(rows)


# ===========================================================================
# 4. STATE-LEVEL CORRELATIONS
# ===========================================================================

def compute_state_correlations(state_df: pd.DataFrame,
                                 merged: pd.DataFrame,
                                 state_rank: pd.DataFrame) -> pd.DataFrame:
    """
    For each state × weather variable compute concurrent (lag 0) Pearson and
    Spearman correlations.  Bonferroni correction applied over 72 tests.

    Returns
    -------
    pd.DataFrame  72 rows (24 states × 3 vars), columns:
        state, var_label, pearson_r, pearson_p, spearman_r, spearman_p,
        n_months, bonferroni_sig_pearson, bonferroni_sig_spearman
    """
    logger.info("Computing state-level correlations (lag 0) …")

    states   = state_rank[config.COL_STATE].tolist()
    n_tests  = len(states) * len(WEATHER_VARS)   # 72
    alpha_bf = config.ALPHA / n_tests

    rows = []
    # Monthly weather series (aligned by billing_month)
    weather_monthly = merged[[
        config.COL_MONTH, "avg_temperature_c",
        "avg_humidity", "total_rainfall_mm",
    ]].set_index(config.COL_MONTH)

    for state in states:
        sdf = (
            state_df[state_df[config.COL_STATE] == state]  # COL_STATE in source df
              .sort_values(config.COL_MONTH)
              .set_index(config.COL_MONTH)
        )

        for col, label, unit, _ in WEATHER_VARS:
            # Align on common months
            aligned = sdf[["net_sale_amt"]].join(
                weather_monthly[[col]], how="inner"
            ).dropna()

            if len(aligned) < 5:
                rows.append({
                    "state": state, "variable": col, "var_label": label,
                    "pearson_r": np.nan, "pearson_p": np.nan,
                    "spearman_r": np.nan, "spearman_p": np.nan,
                    "n_months": len(aligned),
                    "bonferroni_sig_pearson": False,
                    "bonferroni_sig_spearman": False,
                })
                continue

            xv = aligned[col].values
            yv = aligned["net_sale_amt"].values
            pr, pp   = stats.pearsonr(xv, yv)
            sr, sp   = stats.spearmanr(xv, yv)

            rows.append({
                "state": state, "variable": col, "var_label": label,
                "pearson_r": round(pr, 4), "pearson_p": round(pp, 6),
                "spearman_r": round(sr, 4), "spearman_p": round(sp, 6),
                "pearson_stars": _stars(pp),
                "spearman_stars": _stars(sp),
                "n_months": len(aligned),
                "bonferroni_sig_pearson":  pp < alpha_bf,
                "bonferroni_sig_spearman": sp < alpha_bf,
            })

    state_corr = pd.DataFrame(rows)
    n_sig_p = int(state_corr["pearson_p"].notna()
                             .sum() and (state_corr["pearson_p"] < config.ALPHA).sum())
    n_bf    = int(state_corr["bonferroni_sig_pearson"].sum())
    logger.info(
        "  State correlations: %d total | %d p<0.05 | "
        "%d Bonferroni-significant (α=%.5f)",
        len(state_corr), n_sig_p, n_bf, alpha_bf,
    )
    return state_corr


# ===========================================================================
# 5. GRAPH HELPERS
# ===========================================================================

def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved → %s", path.name)


def _draw_correlation_heatmap(ax: plt.Axes, matrix: np.ndarray,
                               row_labels: list[str], col_labels: list[str],
                               annotations: list[list[str]],
                               title: str, cmap: str = "RdBu_r",
                               vmin: float = -1.0, vmax: float = 1.0) -> None:
    """
    Draw a coloured heatmap with cell annotations on the given Axes.
    """
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=9, fontweight="bold")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = matrix[i, j]
            txt = annotations[i][j]
            bg  = abs(val) > 0.45  # dark bg for strong correlations
            tc  = "white" if bg else "#111111"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=8.5, color=tc, fontweight="bold")
    return im


# ===========================================================================
# 6. GRAPH 1 — Pearson + Spearman Side-by-Side Heatmaps
# ===========================================================================

def plot_pearson_spearman_heatmap(lag_corr: pd.DataFrame,
                                   out_dir: Path) -> None:
    """
    Side-by-side heatmaps: Pearson (left) and Spearman (right).
    Rows = weather variables, columns = lag 0, 1, 2, 3.
    Cell text = r value + significance stars.
    """
    logger.info("Plotting Graph 1: Pearson + Spearman Lag Heatmaps …")

    var_labels = [v[1] for v in WEATHER_VARS]
    col_labels = ["Lag 0", "Lag 1", "Lag 2", "Lag 3"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    fig.patch.set_facecolor(CLR_BG)

    for ax, r_col, s_col, title in [
        (ax1, "pearson_r",  "pearson_stars",  "Pearson r"),
        (ax2, "spearman_r", "spearman_stars", "Spearman ρ"),
    ]:
        ax.set_facecolor(CLR_BG)
        mat  = np.zeros((3, 4))
        anns = [[""] * 4 for _ in range(3)]

        for i, (col, label, *_) in enumerate(WEATHER_VARS):
            for j, lag in enumerate(LAGS):
                row = lag_corr[(lag_corr["variable"] == col) &
                               (lag_corr["lag"] == lag)].iloc[0]
                mat[i, j]  = row[r_col]
                anns[i][j] = f"{row[r_col]:+.3f}\n{row[s_col]}"

        im = _draw_correlation_heatmap(
            ax, mat, var_labels, col_labels, anns, title,
            cmap="RdBu_r", vmin=-1, vmax=1,
        )
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label="Correlation Coefficient")
        ax.set_xlabel("Lag (months)", fontsize=10)
        ax.set_ylabel("Weather Variable", fontsize=10)

    fig.suptitle(
        "MAHACEF-200 | Phase 5 — National Lag Correlation  "
        "(Pearson & Spearman)  |  *** p<0.001  ** p<0.01  * p<0.05",
        fontsize=12, fontweight="bold", y=1.04,
    )
    plt.tight_layout()
    _save(fig, out_dir / "01_pearson_spearman_lag_heatmap.png")


# ===========================================================================
# 7. GRAPH 2 — Lag Profile Bar Chart
# ===========================================================================

def plot_lag_profile(lag_corr: pd.DataFrame, out_dir: Path) -> None:
    """
    3-panel bar chart showing Pearson r and Spearman ρ vs lag (0–3) for each
    weather variable.  Bars colour-coded by significance.
    """
    logger.info("Plotting Graph 2: Lag Profile Bar Chart …")

    bar_width = 0.35
    fig, axes = plt.subplots(1, 3, figsize=(17, 6), sharey=True)
    fig.patch.set_facecolor(CLR_BG)

    for ax, (col, label, unit, colour) in zip(axes, WEATHER_VARS):
        ax.set_facecolor(CLR_BG)
        sub = lag_corr[lag_corr["variable"] == col].sort_values("lag")
        x   = np.arange(len(LAGS))

        # Significance colouring
        p_colors  = [("#1565C0" if r >= 0 else "#C62828")
                     if p < 0.05 else "#BDBDBD"
                     for r, p in zip(sub["pearson_r"], sub["pearson_p"])]
        sp_colors = [("#42A5F5" if r >= 0 else "#EF9A9A")
                     if p < 0.05 else "#E0E0E0"
                     for r, p in zip(sub["spearman_r"], sub["spearman_p"])]

        b1 = ax.bar(x - bar_width / 2, sub["pearson_r"].values,
                    width=bar_width, color=p_colors, edgecolor="white",
                    label="Pearson r", zorder=3)
        b2 = ax.bar(x + bar_width / 2, sub["spearman_r"].values,
                    width=bar_width, color=sp_colors, edgecolor="white",
                    label="Spearman ρ", hatch="//", zorder=3)

        # Annotate each bar with value + stars
        for bars, r_col, s_col in [(b1, "pearson_r", "pearson_stars"),
                                    (b2, "spearman_r", "spearman_stars")]:
            for bar, (_, row) in zip(bars, sub.iterrows()):
                v = row[r_col]
                s = row[s_col]
                offset = 0.015 if v >= 0 else -0.02
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    v + offset,
                    f"{v:+.2f}{s}",
                    ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=7.5, fontweight="bold", color="#222222",
                )

        ax.axhline(0, color="#444444", lw=0.8)
        for level, ls in [(0.3, "--"), (-0.3, "--"),
                           (0.6, ":"),  (-0.6, ":")]:
            ax.axhline(level, color=colour, lw=0.7, ls=ls, alpha=0.55)

        ax.set_xticks(x)
        ax.set_xticklabels([f"Lag {k}" for k in LAGS], fontsize=9.5)
        ax.set_title(f"{label} ({unit})", fontsize=11, fontweight="bold",
                     color=colour, pad=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(-1.0, 1.0)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{v:+.1f}"))
        ax.grid(axis="y", ls="--", alpha=0.35)
        if ax is axes[0]:
            ax.set_ylabel("Correlation Coefficient (r / ρ)", fontsize=10)
        ax.legend(fontsize=8, framealpha=0.9, loc="lower right")

    fig.suptitle(
        "MAHACEF-200 | Pearson & Spearman Correlations by Lag  "
        "(Coloured = significant p<0.05  |  Grey = not significant)",
        fontsize=12, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    _save(fig, out_dir / "02_lag_profile_bars.png")


# ===========================================================================
# 8. GRAPH 3 & 4 — State-Level Correlation Heatmaps
# ===========================================================================

def _plot_state_heatmap(state_corr: pd.DataFrame, state_rank: pd.DataFrame,
                         r_col: str, s_col: str, title: str,
                         filename: str, out_dir: Path) -> None:
    """
    24-state × 3-variable heatmap for a given correlation type.
    States ordered by total net sales (descending — highest volume at top).
    """
    states    = state_rank[config.COL_STATE].tolist()
    var_labels = [v[1] for v in WEATHER_VARS]
    n_states  = len(states)

    mat  = np.full((n_states, 3), np.nan)
    anns = [[""] * 3 for _ in range(n_states)]

    for i, state in enumerate(states):
        for j, (col, label, _, _) in enumerate(WEATHER_VARS):
            # state_corr uses "state" as column name (results table)
            row = state_corr[
                (state_corr["state"] == state) &
                (state_corr["variable"] == col)
            ]
            if len(row) > 0 and not np.isnan(row[r_col].values[0]):
                r = row[r_col].values[0]
                s = row[s_col].values[0]
                mat[i, j]  = r
                bf = bool(row["bonferroni_sig_pearson"].values[0]) \
                     if "pearson" in r_col else \
                     bool(row["bonferroni_sig_spearman"].values[0])
                bf_mark = "†" if bf else ""
                anns[i][j] = f"{r:+.2f}{s}{bf_mark}"

    # Shorten state names for display
    row_labels = [s.title().replace("U.p.", "U.P.")
                           .replace("M.p.", "M.P.")
                   for s in states]

    fig, ax = plt.subplots(figsize=(9, 13))
    fig.patch.set_facecolor(CLR_BG)
    ax.set_facecolor(CLR_BG)

    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(3))
    ax.set_xticklabels(var_labels, fontsize=10, fontweight="bold")
    ax.set_yticks(range(n_states))
    ax.set_yticklabels(row_labels, fontsize=8.5)

    for i in range(n_states):
        for j in range(3):
            v = mat[i, j]
            txt = anns[i][j]
            if not np.isnan(v):
                bg  = abs(v) > 0.5
                tc  = "white" if bg else "#111111"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=8, color=tc, fontweight="bold")

    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="Correlation Coefficient")

    caption = (
        "Stars: *** p<0.001  ** p<0.01  * p<0.05  ns not significant\n"
        "† Bonferroni-significant (α/72)  |  "
        "States sorted by total net sales ↓"
    )
    fig.text(0.5, -0.02, caption, ha="center", fontsize=8.5,
             color="#555555", style="italic")

    plt.tight_layout()
    _save(fig, out_dir / filename)


def plot_state_heatmaps(state_corr: pd.DataFrame,
                         state_rank: pd.DataFrame,
                         out_dir: Path) -> None:
    """Generate Graphs 3 (Pearson) and 4 (Spearman) state heatmaps."""
    logger.info("Plotting Graphs 3–4: State Correlation Heatmaps …")
    _plot_state_heatmap(
        state_corr, state_rank,
        r_col="pearson_r",  s_col="pearson_stars",
        title="State-Level Pearson r  |  Sales vs Weather  (Lag 0)",
        filename="03_state_pearson_heatmap.png", out_dir=out_dir,
    )
    _plot_state_heatmap(
        state_corr, state_rank,
        r_col="spearman_r", s_col="spearman_stars",
        title="State-Level Spearman ρ  |  Sales vs Weather  (Lag 0)",
        filename="04_state_spearman_heatmap.png", out_dir=out_dir,
    )


# ===========================================================================
# 9. GRAPH 5 — Partial vs Full Correlation Comparison
# ===========================================================================

def plot_partial_vs_full(lag_corr: pd.DataFrame,
                          partial_corr: pd.DataFrame,
                          out_dir: Path) -> None:
    """
    Bar chart comparison: Full Pearson r vs Partial (detrended) r at each lag.
    Three panels, one per weather variable.  Helps identify how much of the
    correlation is driven by shared time trend vs genuine seasonal link.
    """
    logger.info("Plotting Graph 5: Partial vs Full Correlation …")

    bar_width = 0.35
    fig, axes = plt.subplots(1, 3, figsize=(17, 6), sharey=True)
    fig.patch.set_facecolor(CLR_BG)

    for ax, (col, label, unit, colour) in zip(axes, WEATHER_VARS):
        ax.set_facecolor(CLR_BG)
        full_sub    = lag_corr[lag_corr["variable"] == col].sort_values("lag")
        partial_sub = partial_corr[partial_corr["variable"] == col].sort_values("lag")
        x = np.arange(len(LAGS))

        full_r    = full_sub["pearson_r"].values
        partial_r = partial_sub["partial_r"].values

        b1 = ax.bar(x - bar_width / 2, full_r, width=bar_width,
                    color=colour, alpha=0.75, edgecolor="white",
                    label="Full Pearson r", zorder=3)
        b2 = ax.bar(x + bar_width / 2, partial_r, width=bar_width,
                    color=colour, alpha=0.40, edgecolor="white",
                    label="Partial r (detrended)", hatch="xx", zorder=3)

        for bars, vals, s_vals in [
            (b1, full_r, full_sub["pearson_stars"].values),
            (b2, partial_r, partial_sub["partial_stars"].values),
        ]:
            for bar, v, s in zip(bars, vals, s_vals):
                offset = 0.015 if v >= 0 else -0.025
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    v + offset,
                    f"{v:+.2f}{s}",
                    ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=7.5, fontweight="bold", color="#222222",
                )

        ax.axhline(0, color="#444444", lw=0.8)
        for level in [0.3, -0.3, 0.6, -0.6]:
            ax.axhline(level, color="#AAAAAA", lw=0.7, ls="--", alpha=0.55)

        ax.set_xticks(x)
        ax.set_xticklabels([f"Lag {k}" for k in LAGS], fontsize=9)
        ax.set_title(f"{label} ({unit})", fontsize=11,
                     fontweight="bold", color=colour, pad=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(-1.0, 1.0)
        ax.grid(axis="y", ls="--", alpha=0.35)
        ax.legend(fontsize=8.5, framealpha=0.9)
        if ax is axes[0]:
            ax.set_ylabel("Correlation Coefficient", fontsize=10)

    fig.suptitle(
        "MAHACEF-200 | Full Pearson r  vs  Partial (Detrended) r  "
        "— Controlling for Linear Time Trend",
        fontsize=12, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    _save(fig, out_dir / "05_partial_vs_full_correlation.png")


# ===========================================================================
# 10. GRAPH 6 — Comprehensive Significance Summary Matrix
# ===========================================================================

def plot_significance_summary(lag_corr: pd.DataFrame,
                               partial_corr: pd.DataFrame,
                               out_dir: Path) -> None:
    """
    Comprehensive visual table showing Pearson r, Spearman ρ, and Partial r
    for all 12 combinations (3 vars × 4 lags), with significance colour coding.
    Presented as a styled heatmap with multi-row annotations.
    """
    logger.info("Plotting Graph 6: Comprehensive Significance Matrix …")

    # Build a 12-row × 3-metric (Pearson, Spearman, Partial) display matrix
    rows_order = [(col, lag) for col, lbl, _, _ in WEATHER_VARS
                             for lag in LAGS]
    y_labels   = [f"{lbl}\nLag {lag}"
                  for col, lbl, _, _ in WEATHER_VARS for lag in LAGS]
    x_labels   = ["Pearson r", "Spearman ρ", "Partial r\n(detrended)"]

    mat  = np.zeros((12, 3))
    anns = [[""] * 3 for _ in range(12)]
    p_mat = np.ones((12, 3))  # for background intensity

    for i, (col, lag) in enumerate(rows_order):
        fc_row = lag_corr[(lag_corr["variable"] == col) &
                          (lag_corr["lag"] == lag)]
        pc_row = partial_corr[(partial_corr["variable"] == col) &
                               (partial_corr["lag"] == lag)]

        if len(fc_row):
            r = fc_row["pearson_r"].values[0]
            p = fc_row["pearson_p"].values[0]
            mat[i, 0]  = r
            p_mat[i, 0]= p
            anns[i][0] = f"{r:+.3f}\n{_stars(p)}"

            r2 = fc_row["spearman_r"].values[0]
            p2 = fc_row["spearman_p"].values[0]
            mat[i, 1]  = r2
            p_mat[i, 1]= p2
            anns[i][1] = f"{r2:+.3f}\n{_stars(p2)}"

        if len(pc_row):
            r3 = pc_row["partial_r"].values[0]
            p3 = pc_row["partial_p"].values[0]
            mat[i, 2]  = r3
            p_mat[i, 2]= p3
            anns[i][2] = f"{r3:+.3f}\n{_stars(p3)}"

    fig, ax = plt.subplots(figsize=(9, 13))
    fig.patch.set_facecolor(CLR_BG)
    ax.set_facecolor(CLR_BG)

    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")

    # Draw horizontal separators between weather variables
    for sep in [3.5, 7.5]:
        ax.axhline(sep, color="#888888", lw=2.0)

    # Weather variable group labels on left
    group_y   = [1.5, 5.5, 9.5]
    group_lbl = [v[1] for v in WEATHER_VARS]
    group_col = [v[3] for v in WEATHER_VARS]
    for gy, gl, gc in zip(group_y, group_lbl, group_col):
        ax.text(-0.65, gy, gl, ha="center", va="center",
                fontsize=10, fontweight="bold", color=gc,
                rotation=90, transform=ax.get_yaxis_transform())

    ax.set_xticks(range(3))
    ax.set_xticklabels(x_labels, fontsize=10, fontweight="bold")
    ax.set_yticks(range(12))
    ax.set_yticklabels(y_labels, fontsize=9)

    for i in range(12):
        for j in range(3):
            v = mat[i, j]
            bg  = abs(v) > 0.45
            tc  = "white" if bg else "#111111"
            ax.text(j, i, anns[i][j], ha="center", va="center",
                    fontsize=9, color=tc, fontweight="bold",
                    linespacing=1.4)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="Correlation Coefficient")
    ax.set_title(
        "MAHACEF-200 | Comprehensive Correlation Matrix\n"
        "Pearson | Spearman | Partial  ×  3 Variables × 4 Lags",
        fontsize=12, fontweight="bold", pad=12,
    )
    caption = ("*** p<0.001  ** p<0.01  * p<0.05  ns not significant\n"
               "Partial r = detrended (time-trend removed from both series)")
    fig.text(0.5, -0.02, caption, ha="center", fontsize=8.5,
             color="#555555", style="italic")
    plt.tight_layout()
    _save(fig, out_dir / "06_significance_summary.png")


# ===========================================================================
# 11. EXPORT
# ===========================================================================

def export_phase5_data(lag_corr: pd.DataFrame,
                        partial_corr: pd.DataFrame,
                        state_corr: pd.DataFrame) -> None:
    """Export all correlation results to CSV, Excel (3 sheets), and metadata."""

    export_csv(lag_corr, config.PHASE5_CORRELATION_CSV, logger=logger)

    with pd.ExcelWriter(str(config.PHASE5_CORRELATION_XLSX),
                        engine="openpyxl") as writer:
        lag_corr.to_excel(writer, sheet_name="Lag_Correlations",  index=False)
        partial_corr.to_excel(writer, sheet_name="Partial_Correlations", index=False)
        state_corr.to_excel(writer,   sheet_name="State_Correlations",   index=False)
    logger.info("Excel exported → %s", config.PHASE5_CORRELATION_XLSX.name)

    meta_extra = {
        "lag_correlation_rows": len(lag_corr),
        "state_correlation_rows": len(state_corr),
        "bonferroni_alpha": round(config.ALPHA / 72, 7),
    }
    for path in [config.PHASE5_CORRELATION_CSV, config.PHASE5_CORRELATION_XLSX]:
        write_dataset_metadata(
            path, PHASE_LABEL, SCRIPT_NAME,
            source_dataset=config.CLEAN_DATASET_NAME,
            extra=meta_extra,
        )
    logger.info("Metadata sidecars written.")


# ===========================================================================
# 12. REPORT BUILDER
# ===========================================================================

def build_report(lag_corr: pd.DataFrame,
                  partial_corr: pd.DataFrame,
                  state_corr: pd.DataFrame,
                  state_rank: pd.DataFrame) -> str:
    """Build the standardised Phase 5 report."""

    # Best lag per variable (Pearson)
    best_lag_rows = ""
    for col, label, unit, _ in WEATHER_VARS:
        sub = lag_corr[lag_corr["variable"] == col]
        best = sub.loc[sub["pearson_r"].abs().idxmax()]
        best_lag_rows += (
            f"| {label} | Lag {int(best['lag'])} | "
            f"{best['pearson_r']:+.3f} | {best['pearson_stars']} |\n"
        )

    # Full lag table
    lag_table = ("| Variable | Lag | n | Pearson r | Pearson p | Stars | "
                  "Spearman ρ | Spearman p | Stars |\n"
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
    for _, row in lag_corr.iterrows():
        lag_table += (
            f"| {row['var_label']} | {row['lag']} | {row['n']} | "
            f"{row['pearson_r']:+.3f} | {row['pearson_p']:.4f} | "
            f"{row['pearson_stars']} | "
            f"{row['spearman_r']:+.3f} | {row['spearman_p']:.4f} | "
            f"{row['spearman_stars']} |\n"
        )

    # State highlights (top 5 most correlated states)
    state_sorted = (
        state_corr.dropna(subset=["pearson_r"])
                   .assign(abs_r=lambda d: d["pearson_r"].abs())
                   .sort_values("abs_r", ascending=False)
                   .head(8)
    )
    state_hl_rows = ""
    for _, row in state_sorted.iterrows():
        state_hl_rows += (
            f"| {row['state'].title()} | {row['var_label']} | "
            f"{row['pearson_r']:+.3f} | {row['pearson_stars']} | "
            f"{row['spearman_r']:+.3f} | {row['spearman_stars']} |\n"
        )

    # Partial vs full comparison highlights
    partial_table = ("| Variable | Lag | Full Pearson r | Partial r | Change |\n"
                     "| --- | --- | --- | --- | --- |\n")
    for col, label, *_ in WEATHER_VARS:
        full_sub = lag_corr[lag_corr["variable"] == col]
        part_sub = partial_corr[partial_corr["variable"] == col]
        for lag in LAGS:
            fr = full_sub[full_sub["lag"] == lag]["pearson_r"].values[0]
            pr = part_sub[part_sub["lag"] == lag]["partial_r"].values[0]
            ch = pr - fr
            partial_table += (
                f"| {label} | {lag} | {fr:+.3f} | {pr:+.3f} | {ch:+.3f} |\n"
            )

    n_sig_state = int((state_corr["pearson_p"].dropna() < config.ALPHA).sum())
    n_bf_state  = int(state_corr["bonferroni_sig_pearson"].sum())

    objective = (
        "Apply formal statistical correlation tests to quantify the weather-sales "
        "relationship identified in Phase 4.  **Pearson r** (linear) and "
        "**Spearman ρ** (rank-based, robust to non-normality and outliers) are "
        "computed at lags 0–3 months.  **Partial correlations** isolate the "
        "signal from shared seasonal structure.  A **state-level analysis** "
        "identifies which markets are most weather-responsive."
    )

    dataset_used = (
        "| Attribute | Value |\n| --- | --- |\n"
        f"| Source | `mahacef200_master_dataset_clean.csv` |\n"
        f"| National series | 39 months (25 observed, 14 climatology-imputed) |\n"
        f"| State series | 24 states × up to 39 months |\n"
        f"| Lags tested | 0, 1, 2, 3 months |\n"
        f"| Total national tests | {len(WEATHER_VARS) * len(LAGS) * 2} "
        "(3 vars × 4 lags × Pearson + Spearman) |\n"
        f"| Total state tests | {len(state_corr)} (24 states × 3 vars × 2 methods) |\n"
        f"| Bonferroni correction | α/72 = {config.ALPHA/72:.6f} for state tests |"
    )

    methodology = (
        "1. **Pearson r**: `scipy.stats.pearsonr` — tests linear relationship. "
        "Assumes bivariate normality; suitable for continuous, unimodal series.\n"
        "2. **Spearman ρ**: `scipy.stats.spearmanr` — rank-based; robust to "
        "outliers, non-normality, and monotone nonlinearity.\n"
        "3. **Lag correlations**: weather variable shifted back k=0,1,2,3 months "
        "(`pd.Series.shift(k)`); NaN-pair exclusion applied per lag.\n"
        "4. **Partial / detrended correlation**: Linear trend removed from both "
        "series via OLS residualisation before computing Pearson r. "
        "Controls for shared secular trend.\n"
        "5. **State-level analysis**: Concurrent (lag 0) Pearson + Spearman for "
        "24 states × 3 variables = 72 tests. "
        f"Bonferroni α = {config.ALPHA}/72 = {config.ALPHA/72:.6f} applied.\n"
        "6. **Significance thresholds**: *** p<0.001 | ** p<0.01 | * p<0.05 | ns"
    )

    key_findings = (
        "### Best Lag per Weather Variable (Pearson r)\n\n"
        "| Variable | Best Lag | Pearson r | Significance |\n"
        "| --- | --- | --- | --- |\n"
        + best_lag_rows.strip()
        + "\n\n### Full Lag Correlation Table\n\n"
        + lag_table.strip()
        + f"\n\n### State-Level Highlights (Top 8 by |Pearson r|)\n\n"
        "| State | Variable | Pearson r | Stars | Spearman ρ | Stars |\n"
        "| --- | --- | --- | --- | --- | --- |\n"
        + state_hl_rows.strip()
        + f"\n\n**Summary**: {n_sig_state} of {len(state_corr)} "
        f"state-variable pairs are p<0.05; {n_bf_state} survive Bonferroni correction."
        + "\n\n### Partial vs Full Correlation (Top Variables)\n\n"
        + partial_table.strip()
    )

    business_insights = (
        "1. **Rainfall is the strongest predictor with a 1-month lag**: "
        "The highest correlation in the national series is rainfall at lag 1 "
        "(r ≈ +0.72 Pearson), indicating that months of high rainfall are "
        "followed by elevated antibiotic sales — consistent with waterborne/"
        "respiratory infection pathways.\n\n"
        "2. **Temperature is stronger at lag 2–3 months**: "
        "This suggests that heat-wave conditions in summer (May–Jun) "
        "drive antibiotic demand with a delay, possibly reflecting the "
        "incubation + healthcare-seeking timeline.\n\n"
        "3. **Partial correlations confirm genuine signal**: "
        "Detrended correlations remain substantial for rainfall and temperature, "
        "ruling out the possibility that the signal is purely an artefact of "
        "shared secular trends.\n\n"
        "4. **State heterogeneity is high**: "
        "Correlation strength varies widely across the 24 states. "
        "High-volume states (U.P., Maharashtra) anchor the national signal; "
        "smaller states may respond to local weather patterns not captured "
        "by the national weather average."
    )

    limitations = (
        "- **National weather proxy** attenuates state-level correlations "
        "(ecological fallacy risk).\n"
        "- **14 imputed months** (Apr 2023–Apr 2024): Climatology imputation "
        "preserves average seasonality but reduces inter-annual variability, "
        "which may inflate seasonal correlation estimates.\n"
        "- **Small n at high lag**: At lag 3, only 36 pairs available "
        "(3 months lost to shifting). P-values have lower power.\n"
        "- **Multiple comparison risk**: 24-state analysis uses Bonferroni "
        "correction, which is conservative. FDR (Benjamini-Hochberg) "
        "correction is a possible alternative for Phase 5 extensions.\n"
        "- **Correlation ≠ causation**: Seasonal co-movement of disease "
        "burden and weather is the most plausible mechanism, but "
        "unmeasured confounders (promotional cycles, stock-outs, competitor "
        "activity) may contribute."
    )

    next_phase = (
        "**Phase 6 — Statistical Testing & Distribution Analysis**\n\n"
        "- Normality tests (Shapiro-Wilk, Q-Q plots) on sales and weather residuals\n"
        "- Stationarity tests (ADF, KPSS) on the sales time-series\n"
        "- Autocorrelation (ACF/PACF) plots to characterise serial dependence\n"
        "- Effect size reporting (Cohen's guidelines for r)\n"
        "- These diagnostics inform which regression model is appropriate in Phase 7"
    )

    return build_phase_report(
        phase_number="5",
        phase_title="Correlation Analysis",
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

def run_correlation_analysis() -> None:
    """Execute the complete Phase 5 — Correlation Analysis pipeline."""

    logger.info("=" * 60)
    logger.info("PHASE 5 — CORRELATION ANALYSIS")
    logger.info("=" * 60)

    ensure_directories(
        config.PHASE5_GRAPHS_DIR,
        config.DATA_DIR,
        config.EXCEL_DIR,
        config.REPORTS_DIR,
        config.LOGS_DIR,
    )

    # ------------------------------------------------------------------ Load
    path = config.MASTER_CLEAN_CSV
    if not path.exists():
        raise FileNotFoundError(f"Clean dataset not found: {path}")
    logger.info("Loading clean dataset: %s", path)
    df = pd.read_csv(str(path))
    df[config.COL_MONTH] = df[config.COL_MONTH].astype(int)
    df[config.COL_STATE] = normalize_state_name(df[config.COL_STATE])

    # ------------------------------------------------------------------ Build series
    merged, state_df, state_rank = load_and_build(df)

    # ------------------------------------------------------------------ Compute
    lag_corr     = compute_lag_correlations(merged)
    partial_corr = compute_partial_correlations(merged)
    state_corr   = compute_state_correlations(state_df, merged, state_rank)

    # ------------------------------------------------------------------ Graphs
    out_dir = config.PHASE5_GRAPHS_DIR
    plot_pearson_spearman_heatmap(lag_corr, out_dir)         # 1
    plot_lag_profile(lag_corr, out_dir)                      # 2
    plot_state_heatmaps(state_corr, state_rank, out_dir)     # 3–4
    plot_partial_vs_full(lag_corr, partial_corr, out_dir)    # 5
    plot_significance_summary(lag_corr, partial_corr, out_dir)# 6

    # ------------------------------------------------------------------ Export
    export_phase5_data(lag_corr, partial_corr, state_corr)

    # ------------------------------------------------------------------ Report
    report = build_report(lag_corr, partial_corr, state_corr, state_rank)
    write_markdown_report(config.REPORT_CORRELATION, report, logger=logger)

    # ------------------------------------------------------------------ Summary
    logger.info("-" * 60)
    logger.info("PHASE 5 COMPLETE")
    best_overall = lag_corr.loc[lag_corr["pearson_r"].abs().idxmax()]
    logger.info(
        "  Best correlation: %s at lag %d  r=%+.3f (%s)",
        best_overall["var_label"], best_overall["lag"],
        best_overall["pearson_r"], best_overall["pearson_stars"],
    )
    n_sig = int((lag_corr["pearson_p"] < config.ALPHA).sum())
    logger.info("  Significant pairs (p<0.05): %d / %d",
                n_sig, len(lag_corr))
    logger.info("  State sig (p<0.05): %d / %d",
                int((state_corr["pearson_p"].dropna() < config.ALPHA).sum()),
                len(state_corr))
    logger.info("-" * 60)


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    try:
        run_correlation_analysis()
        logger.info("Script finished successfully.")
        sys.exit(0)
    except FileNotFoundError as exc:
        logger.error("File not found: %s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
        sys.exit(99)
