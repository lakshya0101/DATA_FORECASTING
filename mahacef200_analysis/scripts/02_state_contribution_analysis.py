"""
02_state_contribution_analysis.py
==================================
Phase 1 – Step 3 | State Contribution Analysis

Reads the aggregated MAHACEF-200 sales data produced by Step 2,
computes state-level sales metrics, classifies each state into a region,
calculates regional contributions, and generates publication-quality graphs.

Outputs
-------
data/mahacef200_statewise_sales.csv
excel/Mahacef200_Statewise_Sales.xlsx
graphs/01_statewise_net_sales.png
graphs/02_top10_states.png
graphs/03_regional_sales_distribution.png
reports/Phase1_State_Contribution.md

Usage
-----
    python mahacef200_analysis/scripts/02_state_contribution_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path when script is run directly
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_MODULE_DIR = _SCRIPT_DIR.parent
_PROJECT_ROOT = _MODULE_DIR.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Standard library + third-party imports
# ---------------------------------------------------------------------------
import textwrap
import warnings

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Internal imports
# ---------------------------------------------------------------------------
from mahacef200_analysis import config
from mahacef200_analysis.utils import (
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
    validate_required_columns,
    write_markdown_report,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Colour palette (consistent across all charts)
# ---------------------------------------------------------------------------
REGION_PALETTE: dict[str, str] = {
    "North": "#2196F3",
    "South": "#4CAF50",
    "East": "#FF9800",
    "West": "#9C27B0",
    "Central": "#F44336",
    "North-East": "#00BCD4",
    "Unknown": "#9E9E9E",
}

BAR_COLOUR = "#1565C0"
ACCENT_COLOUR = "#E53935"


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
        If the file does not exist (Step 2 must be run first).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Aggregated sales file not found: {path}\n"
            "Please run Step 2 (01_extract_mahacef_sales.py) first."
        )
    logger.info("Loading aggregated sales data from: %s", path)
    df = pd.read_csv(str(path))
    logger.info("Loaded %d rows × %d cols.", *df.shape)
    return df


def assign_region(state_series: pd.Series) -> pd.Series:
    """
    Map each state name to its geographic region using config.STATE_REGION_MAP.

    States not found in the mapping are classified as 'Unknown'.

    Parameters
    ----------
    state_series : pd.Series
        Normalised (UPPER) state name strings.

    Returns
    -------
    pd.Series
        Region strings.
    """
    return state_series.map(config.STATE_REGION_MAP).fillna("Unknown")


def compute_statewise_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate all sales periods into a single row per state.

    Computed columns
    ----------------
    gross_sale_amt, net_sale_amt, gross_sale_qty, net_sale_qty,
    fresh_ret_amt (returns), region, contribution_pct, rank.

    Parameters
    ----------
    df : pd.DataFrame
        Monthly MAHACEF-200 sales (output of Step 2).

    Returns
    -------
    pd.DataFrame
        One row per state, sorted by net_sale_amt descending.
    """
    logger.info("Aggregating state-level totals …")

    agg_cols = {
        "gross_sale_amt": "sum",
        "net_sale_amt": "sum",
        "gross_sale_qty": "sum",
        "net_sale_qty": "sum",
        "fresh_ret_amt": "sum",
        "fresh_ret_qty": "sum",
        "expiry_amt": "sum",
        "expiry_qty": "sum",
        "brkg_amt": "sum",
        "brkg_qty": "sum",
    }

    state_df = (
        df.groupby(config.COL_STATE, as_index=False)
          .agg(agg_cols)
          .sort_values("net_sale_amt", ascending=False)
          .reset_index(drop=True)
    )

    # Derived metrics
    total_net_sales = state_df["net_sale_amt"].sum()
    state_df["contribution_pct"] = (
        state_df["net_sale_amt"] / total_net_sales * 100
    ).round(2)
    state_df["rank"] = state_df["net_sale_amt"].rank(
        method="dense", ascending=False
    ).astype(int)

    # Returns value
    state_df["total_returns_amt"] = (
        state_df["fresh_ret_amt"] + state_df["expiry_amt"] + state_df["brkg_amt"]
    )

    # Region classification
    state_df["region"] = assign_region(
        normalize_state_name(state_df[config.COL_STATE])
    )

    logger.info(
        "State aggregation complete: %d states, total net sales = ₹%s",
        len(state_df),
        format_number(total_net_sales),
    )
    return state_df


def compute_regional_aggregates(state_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate state-level data to the regional level.

    Parameters
    ----------
    state_df : pd.DataFrame
        Output of compute_statewise_aggregates().

    Returns
    -------
    pd.DataFrame
        One row per region, sorted by net_sale_amt descending.
    """
    logger.info("Computing regional aggregates …")
    region_df = (
        state_df.groupby("region", as_index=False)
                .agg(
                    gross_sale_amt=("gross_sale_amt", "sum"),
                    net_sale_amt=("net_sale_amt", "sum"),
                    gross_sale_qty=("gross_sale_qty", "sum"),
                    net_sale_qty=("net_sale_qty", "sum"),
                    num_states=("root_state_name", "count"),
                )
                .sort_values("net_sale_amt", ascending=False)
                .reset_index(drop=True)
    )

    total = region_df["net_sale_amt"].sum()
    region_df["contribution_pct"] = (region_df["net_sale_amt"] / total * 100).round(2)
    return region_df


def find_top_state_per_region(state_df: pd.DataFrame, regions: list[str]) -> dict[str, str]:
    """
    Return the highest contributing state for each requested region.

    Parameters
    ----------
    state_df : pd.DataFrame
        State-level aggregated data.
    regions : list[str]
        List of region names to find top state for.

    Returns
    -------
    dict[str, str]
        {region: "STATE_NAME (contribution_pct%)"}
    """
    result: dict[str, str] = {}
    for region in regions:
        subset = state_df[state_df["region"] == region]
        if subset.empty:
            result[region] = "No data"
            continue
        top_row = subset.sort_values("net_sale_amt", ascending=False).iloc[0]
        result[region] = (
            f"{top_row[config.COL_STATE]} ({top_row['contribution_pct']:.2f}% of total)"
        )
    return result


# ===========================================================================
# GRAPHING FUNCTIONS
# ===========================================================================

def _style_axis(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    """Apply a consistent, publication-quality style to a matplotlib Axes."""
    ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=9)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"₹{x/1e6:.1f}M")
    )


def plot_statewise_net_sales(state_df: pd.DataFrame, out_path: Path) -> None:
    """
    Horizontal bar chart — net sales for every state, coloured by region.

    Parameters
    ----------
    state_df : pd.DataFrame
        State-level aggregated data.
    out_path : Path
        Destination PNG file path.
    """
    logger.info("Generating graph: Statewise Net Sales …")

    plot_df = state_df.sort_values("net_sale_amt", ascending=True)

    fig, ax = plt.subplots(figsize=(14, max(8, len(plot_df) * 0.45)))
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    colours = [REGION_PALETTE.get(r, "#9E9E9E") for r in plot_df["region"]]
    bars = ax.barh(
        plot_df[config.COL_STATE],
        plot_df["net_sale_amt"],
        color=colours,
        edgecolor="white",
        linewidth=0.5,
        height=0.7,
    )

    # Value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width * 1.01,
            bar.get_y() + bar.get_height() / 2,
            f"₹{width/1e6:.1f}M",
            va="center",
            fontsize=8,
            color="#333333",
        )

    # Region legend
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(color=colour, label=region)
        for region, colour in REGION_PALETTE.items()
        if region in plot_df["region"].values
    ]
    ax.legend(
        handles=legend_handles,
        title="Region",
        loc="lower right",
        fontsize=8,
        title_fontsize=9,
        framealpha=0.8,
    )

    ax.set_title(
        f"MAHACEF-200 | Net Sales by State (All Periods)",
        fontsize=14, fontweight="bold", pad=14,
    )
    ax.set_xlabel("Net Sales (₹)", fontsize=11)
    ax.set_ylabel("State", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"₹{x/1e6:.1f}M")
    )
    ax.tick_params(axis="both", labelsize=9)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)


def plot_top10_states(state_df: pd.DataFrame, out_path: Path) -> None:
    """
    Vertical bar chart — top 10 states by net sales with contribution % labels.

    Parameters
    ----------
    state_df : pd.DataFrame
        State-level aggregated data.
    out_path : Path
        Destination PNG file path.
    """
    logger.info("Generating graph: Top 10 States …")

    top10 = state_df.nlargest(config.TOP_N_STATES, "net_sale_amt").copy()
    colours = [REGION_PALETTE.get(r, "#9E9E9E") for r in top10["region"]]

    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE_WIDE)
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    bars = ax.bar(
        top10[config.COL_STATE],
        top10["net_sale_amt"],
        color=colours,
        edgecolor="white",
        linewidth=0.8,
        width=0.65,
    )

    # Value + contribution % labels above bars
    for bar, (_, row) in zip(bars, top10.iterrows()):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h * 1.01,
            f"₹{h/1e6:.1f}M\n({row['contribution_pct']:.1f}%)",
            ha="center", va="bottom",
            fontsize=8.5, color="#222222",
            fontweight="semibold",
        )

    ax.set_title(
        f"MAHACEF-200 | Top {config.TOP_N_STATES} States by Net Sales",
        fontsize=14, fontweight="bold", pad=16,
    )
    ax.set_xlabel("State", fontsize=11)
    ax.set_ylabel("Net Sales (₹)", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"₹{x/1e6:.1f}M")
    )
    ax.tick_params(axis="x", rotation=30, labelsize=9)
    ax.tick_params(axis="y", labelsize=9)

    # Rank annotations inside bars
    for bar, rank in zip(bars, top10["rank"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 0.05,
            f"#{rank}",
            ha="center", va="bottom",
            fontsize=9, color="white", fontweight="bold",
        )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)


def plot_regional_distribution(region_df: pd.DataFrame, out_path: Path) -> None:
    """
    Side-by-side pie + bar chart for regional net sales distribution.

    Parameters
    ----------
    region_df : pd.DataFrame
        Regional aggregate data.
    out_path : Path
        Destination PNG file path.
    """
    logger.info("Generating graph: Regional Sales Distribution …")

    fig, (ax_pie, ax_bar) = plt.subplots(
        1, 2, figsize=(16, 8), gridspec_kw={"width_ratios": [1, 1.3]}
    )
    fig.patch.set_facecolor("#F8F9FA")
    for ax in (ax_pie, ax_bar):
        ax.set_facecolor("#F8F9FA")

    pie_colours = [REGION_PALETTE.get(r, "#9E9E9E") for r in region_df["region"]]

    # --- Pie chart ---
    wedges, texts, autotexts = ax_pie.pie(
        region_df["net_sale_amt"],
        labels=region_df["region"],
        colors=pie_colours,
        autopct="%1.1f%%",
        startangle=140,
        pctdistance=0.75,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_fontweight("bold")
    for t in texts:
        t.set_fontsize(10)

    ax_pie.set_title(
        "Regional Net Sales Share (%)",
        fontsize=13, fontweight="bold", pad=12,
    )

    # --- Horizontal bar chart ---
    plot_region = region_df.sort_values("net_sale_amt", ascending=True)
    bar_colours = [REGION_PALETTE.get(r, "#9E9E9E") for r in plot_region["region"]]

    hbars = ax_bar.barh(
        plot_region["region"],
        plot_region["net_sale_amt"],
        color=bar_colours,
        edgecolor="white",
        linewidth=0.8,
        height=0.6,
    )

    for bar, pct in zip(hbars, plot_region["contribution_pct"]):
        w = bar.get_width()
        ax_bar.text(
            w * 1.01, bar.get_y() + bar.get_height() / 2,
            f"₹{w/1e6:.1f}M ({pct:.1f}%)",
            va="center", fontsize=9, color="#333333",
        )

    ax_bar.set_title(
        "Regional Net Sales (₹)",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax_bar.set_xlabel("Net Sales (₹)", fontsize=11)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"₹{x/1e6:.1f}M")
    )
    ax_bar.tick_params(axis="both", labelsize=9)

    fig.suptitle(
        "MAHACEF-200 | Regional Sales Distribution",
        fontsize=15, fontweight="bold", y=1.01,
    )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)


# ===========================================================================
# REPORT BUILDER
# ===========================================================================

def build_state_contribution_report(
    state_df: pd.DataFrame,
    region_df: pd.DataFrame,
    top_by_region: dict[str, str],
) -> str:
    """
    Build the Phase 1 State Contribution markdown report.

    Parameters
    ----------
    state_df : pd.DataFrame
        State-level aggregates.
    region_df : pd.DataFrame
        Regional aggregates.
    top_by_region : dict[str, str]
        Top state per region.

    Returns
    -------
    str
        Full markdown report.
    """
    total_net = state_df["net_sale_amt"].sum()
    top_state_row = state_df.iloc[0]
    bottom_state_row = state_df.iloc[-1]

    # State table rows
    state_table_rows = ""
    for _, row in state_df.iterrows():
        state_table_rows += (
            f"| {int(row['rank'])} | {row[config.COL_STATE]} | {row['region']} "
            f"| ₹{format_number(row['gross_sale_amt'])} "
            f"| ₹{format_number(row['net_sale_amt'])} "
            f"| {format_number(row['gross_sale_qty'])} "
            f"| {format_number(row['net_sale_qty'])} "
            f"| ₹{format_number(row['total_returns_amt'])} "
            f"| {row['contribution_pct']:.2f}% |\n"
        )

    # Regional table rows
    region_table_rows = ""
    for _, row in region_df.sort_values("net_sale_amt", ascending=False).iterrows():
        region_table_rows += (
            f"| {row['region']} | {int(row['num_states'])} "
            f"| ₹{format_number(row['net_sale_amt'])} "
            f"| {row['contribution_pct']:.2f}% |\n"
        )

    # Top-state-per-region table
    top_region_rows = ""
    for region, info in top_by_region.items():
        top_region_rows += f"| {region} | {info} |\n"

    report = textwrap.dedent(f"""
    # Phase 1 – State Contribution Analysis Report
    ## MAHACEF-200 | Mankind Cures

    **Generated:** {current_timestamp()} UTC

    ---

    ## 1. Objective

    This report analyses state-level sales contributions for **{config.TARGET_PRODUCT}**,
    classifies each state into a geographic region, and identifies key business
    insights on geographic sales distribution.

    ---

    ## 2. Executive Summary

    | Metric | Value |
    | --- | --- |
    | Total States with Sales | {len(state_df)} |
    | Total Regions Covered | {region_df['region'].nunique()} |
    | Total Net Sales (All States) | ₹{format_number(total_net)} |
    | Top State | {top_state_row[config.COL_STATE]} ({top_state_row['contribution_pct']:.2f}%) |
    | Bottom State | {bottom_state_row[config.COL_STATE]} ({bottom_state_row['contribution_pct']:.2f}%) |

    ---

    ## 3. State-wise Sales Breakdown

    | Rank | State | Region | Gross Sales (₹) | Net Sales (₹) | Gross Qty | Net Qty | Returns (₹) | Contribution % |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    {state_table_rows.strip()}

    ---

    ## 4. Regional Sales Summary

    | Region | States | Net Sales (₹) | Contribution % |
    | --- | --- | --- | --- |
    {region_table_rows.strip()}

    ---

    ## 5. Top Contributing State per Region

    | Region | Top State (% of Total) |
    | --- | --- |
    {top_region_rows.strip()}

    ---

    ## 6. Business Insights

    1. **Dominant Geography**: **{top_state_row[config.COL_STATE]}** leads with
       {top_state_row['contribution_pct']:.2f}% of total net sales,
       indicating a strong concentration of MAHACEF-200 demand in this market.

    2. **Regional Skew**: The top-performing region by net sales is
       **{region_df.sort_values('net_sale_amt', ascending=False).iloc[0]['region']}**,
       contributing
       {region_df.sort_values('net_sale_amt', ascending=False).iloc[0]['contribution_pct']:.1f}%
       of national sales — suggesting favourable market penetration or higher
       antibiotic prescription rates in that zone.

    3. **Returns Analysis**: States with high gross-to-net spread may indicate
       stockist-level return pressure or demand seasonality — a key risk factor
       for accurate forecasting.

    4. **Untapped Markets**: States at the bottom of the ranking represent
       potential growth opportunities for targeted sales interventions,
       especially in regions under-penetrated by MAHACEF-200.

    5. **Weather–Sales Hypothesis**: States with strong seasonal weather
       variation (e.g., high rainfall, temperature swings) are expected to
       exhibit stronger correlation between weather and antibiotic sales
       — this will be explored in Phase 2.

    ---

    ## 7. Graphs Generated

    | Graph File | Description |
    | --- | --- |
    | `graphs/01_statewise_net_sales.png` | Net sales bar chart — all states, coloured by region |
    | `graphs/02_top10_states.png` | Top 10 states with contribution % labels |
    | `graphs/03_regional_sales_distribution.png` | Pie + bar chart of regional sales share |

    ---

    ## 8. Output Files

    | File | Description |
    | --- | --- |
    | `data/mahacef200_statewise_sales.csv` | State-level aggregates with region & contribution % |
    | `excel/Mahacef200_Statewise_Sales.xlsx` | Same data in Excel format |

    ---
    *Report auto-generated by `02_state_contribution_analysis.py` | Phase 1 Pipeline*
    """).lstrip()

    return report


# ===========================================================================
# MAIN PIPELINE
# ===========================================================================

def run_state_analysis() -> pd.DataFrame:
    """
    Execute the complete state contribution analysis pipeline.

    Returns
    -------
    pd.DataFrame
        State-level aggregated DataFrame.
    """
    logger.info("=" * 60)
    logger.info("PHASE 1 – STEP 3: STATE CONTRIBUTION ANALYSIS")
    logger.info("=" * 60)

    ensure_directories(
        config.DATA_DIR,
        config.EXCEL_DIR,
        config.GRAPHS_DIR,
        config.REPORTS_DIR,
        config.LOGS_DIR,
    )

    # ------------------------------------------------------------------
    # 1. Load aggregated sales
    # ------------------------------------------------------------------
    df = load_mahacef_sales(config.MAHACEF_SALES_CSV)

    # Validate
    required = [config.COL_STATE, config.COL_MONTH, "net_sale_amt", "gross_sale_amt"]
    validate_required_columns(df, required, "mahacef200_sales.csv")
    validate_not_empty(df, "mahacef200_sales.csv")

    # Normalise state names
    df[config.COL_STATE] = normalize_state_name(df[config.COL_STATE])

    # ------------------------------------------------------------------
    # 2. State-level aggregation
    # ------------------------------------------------------------------
    state_df = compute_statewise_aggregates(df)

    # ------------------------------------------------------------------
    # 3. Regional aggregation
    # ------------------------------------------------------------------
    region_df = compute_regional_aggregates(state_df)

    # ------------------------------------------------------------------
    # 4. Top state per key region
    # ------------------------------------------------------------------
    top_by_region = find_top_state_per_region(
        state_df,
        regions=["North", "South", "East", "West", "Central", "North-East"],
    )
    logger.info("Top states per region: %s", top_by_region)

    # ------------------------------------------------------------------
    # 5. Generate graphs
    # ------------------------------------------------------------------
    plot_statewise_net_sales(state_df, config.GRAPH_STATEWISE_NET_SALES)
    plot_top10_states(state_df, config.GRAPH_TOP10_STATES)
    plot_regional_distribution(region_df, config.GRAPH_REGIONAL_DIST)

    # ------------------------------------------------------------------
    # 6. Export state data
    # ------------------------------------------------------------------
    export_csv(state_df, config.STATEWISE_SALES_CSV, logger=logger)
    export_excel(
        state_df,
        config.STATEWISE_SALES_XLSX,
        sheet_name="Statewise_Sales",
        logger=logger,
    )

    # ------------------------------------------------------------------
    # 7. Write report
    # ------------------------------------------------------------------
    report_text = build_state_contribution_report(state_df, region_df, top_by_region)
    write_markdown_report(config.REPORT_STATE_CONTRIBUTION, report_text, logger=logger)

    # Console summary
    logger.info("-" * 60)
    logger.info("STATE CONTRIBUTION ANALYSIS COMPLETE")
    logger.info("  States analysed  : %d", len(state_df))
    logger.info("  Regions covered  : %d", region_df["region"].nunique())
    logger.info(
        "  Total net sales  : ₹%s",
        format_number(state_df["net_sale_amt"].sum()),
    )
    logger.info("-" * 60)

    return state_df


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    try:
        df = run_state_analysis()
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
