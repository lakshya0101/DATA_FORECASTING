"""
04_monthly_sales_trend.py
=============================================================================
Project: Weather Driven Pharmaceutical Sales Analytics & Forecasting
Module:  Monthly Sales Trend & Executive Business Metrics
Author:  Lakshya Dogra
Location: Noida, Uttar Pradesh, India
Date:    June 2026
=============================================================================
Description:
    End-to-end pipeline for converting YYYYMM sales data into business
    insights. Calculates KPIs, detects spikes using rolling statistics,
    and generates Deloitte/McKinsey-tier dashboard visualizations and reports.
=============================================================================
"""

import os
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter

# =============================================================================
# CONFIGURATION & SETUP
# =============================================================================

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Input and Output Paths
INPUT_FILE = "data/Sale_Details.xlsx"
BASE_DIRS = [
    "sales_eda/excel/business/",
    "sales_eda/graphs/business/",
    "reports/phase_2/"
]

# Required Columns strictly enforced
REQUIRED_COLUMNS = [
    "billing_month", "gross_sale_amt", "gross_sale_qty",
    "fresh_ret_amt", "fresh_ret_qty", "expiry_amt", "expiry_qty",
    "brkg_amt", "brkg_qty", "net_sale_amt", "net_sale_qty"
]

# Matplotlib Global Parameters for Presentation Quality
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'axes.titlesize': 20,
    'axes.titleweight': 'bold',
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'figure.figsize': (16, 9),
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight'
})

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def setup_directories():
    """Create necessary output directories if they do not exist."""
    for directory in BASE_DIRS:
        os.makedirs(directory, exist_ok=True)
    logging.info("Output directories verified and ready.")

def format_indian_currency(num, pos=None):
    """
    Format large numbers into Indian currency system (Lakhs, Crores).
    Compatible with Matplotlib FuncFormatter.
    """
    if pd.isna(num):
        return "0.00"
    
    is_negative = num < 0
    abs_num = abs(num)
    
    if abs_num >= 1e7:
        formatted = f"{abs_num/1e7:.2f} Cr"
    elif abs_num >= 1e5:
        formatted = f"{abs_num/1e5:.2f} Lakh"
    elif abs_num >= 1e3:
        formatted = f"{abs_num/1e3:.2f} K"
    else:
        formatted = f"{abs_num:.2f}"
        
    return f"-{formatted}" if is_negative else formatted

def validate_data(df):
    """Validate presence of all required columns."""
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        logging.error(f"Missing required columns: {missing_cols}")
        raise ValueError(f"Input dataset is missing columns: {missing_cols}")
    logging.info("Dataset validation passed.")

# =============================================================================
# DATA PROCESSING & METRICS GENERATION
# =============================================================================

def load_and_preprocess(filepath):
    """Load dataset, handle missing values, and format dates."""
    logging.info(f"Loading data from {filepath}...")
    try:
        df = pd.read_excel(filepath)
    except FileNotFoundError:
        logging.error(f"File not found: {filepath}. Please ensure data exists.")
        # Create a mock dataframe strictly for execution fallback if data is missing
        # In production, this exception would just halt execution.
        raise

    validate_data(df)

    # Handle Missing Values
    df[REQUIRED_COLUMNS[1:]] = df[REQUIRED_COLUMNS[1:]].fillna(0)

    # Convert billing_month from YYYYMM to Datetime and Formatted String
    df['billing_month'] = df['billing_month'].astype(str)
    df['date_obj'] = pd.to_datetime(df['billing_month'], format='%Y%m')
    df['Month-Year'] = df['date_obj'].dt.strftime('%b-%Y')

    return df

def aggregate_monthly_metrics(df):
    """Generate all business metrics, rolling averages, and detect spikes."""
    logging.info("Calculating Monthly Business Metrics...")
    
    # Monthly Aggregation
    monthly_df = df.groupby(['date_obj', 'Month-Year']).agg({
        'gross_sale_amt': 'sum',
        'gross_sale_qty': 'sum',
        'net_sale_amt': 'sum',
        'net_sale_qty': 'sum',
        'fresh_ret_amt': 'sum',
        'expiry_amt': 'sum',
        'brkg_amt': 'sum'
    }).reset_index()

    monthly_df = monthly_df.sort_values('date_obj')
    monthly_df = monthly_df.reset_index(drop=True)

    # Total Returns
    monthly_df['total_returns_amt'] = (
        monthly_df['fresh_ret_amt'] + 
        monthly_df['expiry_amt'] + 
        monthly_df['brkg_amt']
    )

    # Cumulative Sales & Contribution
    monthly_df['cumulative_net_sales'] = monthly_df['net_sale_amt'].cumsum()
    total_sales = monthly_df['net_sale_amt'].sum()
    monthly_df['contribution_pct'] = (monthly_df['net_sale_amt'] / total_sales) * 100

    # Growth Calculations
    monthly_df['prev_month_net'] = monthly_df['net_sale_amt'].shift(1)
    monthly_df['mom_growth_pct'] = np.where(
        monthly_df['prev_month_net'] > 0,
        ((monthly_df['net_sale_amt'] - monthly_df['prev_month_net']) / monthly_df['prev_month_net']) * 100,
        0
    )

    # Spike Detection: Rolling Mean, Std, and Z-Score
    monthly_df['rolling_3m_avg'] = monthly_df['net_sale_amt'].rolling(window=3, min_periods=1).mean()
    monthly_df['rolling_3m_std'] = monthly_df['net_sale_amt'].rolling(window=3, min_periods=1).std().fillna(0)
    
    # Z-Score based on rolling metrics to catch local spikes
    monthly_df['z_score'] = np.where(
        monthly_df['rolling_3m_std'] > 0,
        (monthly_df['net_sale_amt'] - monthly_df['rolling_3m_avg']) / monthly_df['rolling_3m_std'],
        0
    )
    
    # Define Spikes (Z-score > 1.5 indicates a significant positive deviation)
    monthly_df['is_spike'] = monthly_df['z_score'] > 1.5

    return monthly_df

# =============================================================================
# PRESENTATION-QUALITY VISUALIZATIONS
# =============================================================================

def _add_value_labels(ax, x_coords, y_coords, format_func, is_pct=False):
    """Helper to add text labels on plots"""
    for x, y in zip(x_coords, y_coords):
        if pd.isna(y): continue
        label = f"{y:.1f}%" if is_pct else format_func(y)
        ax.text(x, y, label, ha='center', va='bottom', fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))

def generate_individual_graphs(df):
    """Generate and export high-resolution individual graphs."""
    logging.info("Generating Presentation Quality Graphs...")
    
    # x_labels = df['Month-Year']
    # x_pos = np.arange(len(df))

    x_values = df['date_obj']
    x_labels = df['Month-Year']
    formatter = FuncFormatter(format_indian_currency)

    max_idx = df['net_sale_amt'].idxmax()
    min_idx = df['net_sale_amt'].idxmin()

    graphs_config = [
        ("Monthly Net Sales Trend", 'net_sale_amt', False, "1_Monthly_Net_Sales"),
        ("Monthly Gross Sales Trend", 'gross_sale_amt', False, "2_Monthly_Gross_Sales"),
        ("Monthly Net Quantity Trend", 'net_sale_qty', False, "3_Monthly_Net_Qty"),
        ("Monthly Growth Percentage", 'mom_growth_pct', True, "4_Monthly_Growth_Pct"),
        ("3 Month Rolling Average", 'rolling_3m_avg', False, "5_Rolling_Average"),
        ("Cumulative Sales", 'cumulative_net_sales', False, "6_Cumulative_Sales"),
        ("Monthly Returns Trend", 'total_returns_amt', False, "7_Monthly_Returns")
    ]

    for title, col, is_pct, filename in graphs_config:
        fig, ax = plt.subplots()
        
        # Plot base line
        color = '#2c3e50'
        ax.plot(
            x_values,
            df[col],
            marker='o',
            linewidth=2.5,
            markersize=8,
            color=color,
            label=title
        )        
        # Format axes
        # ax.set_xticks(x_pos)
        # ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.xaxis.set_major_formatter(
            mdates.DateFormatter("%b-%Y")
        )

        ax.xaxis.set_major_locator(
            mdates.MonthLocator(interval=2)
        )

        plt.xticks(rotation=45)
        ax.set_title(title)
        
        if not is_pct and 'qty' not in col:
            ax.yaxis.set_major_formatter(formatter)
            ax.set_ylabel("Amount (INR)")
        elif is_pct:
            ax.set_ylabel("Growth (%)")
        else:
            ax.set_ylabel("Quantity")

        # Highlights
        if col == 'net_sale_amt':
            # Highest / Lowest markers
            ax.plot(max_idx, df[col].iloc[max_idx], marker='o', markersize=12, color='#27ae60', label='Highest Month')
            ax.plot(min_idx, df[col].iloc[min_idx], marker='o', markersize=12, color='#e74c3c', label='Lowest Month')
            
            # Spike Annotations
            spikes = df[df['is_spike']]
            if not spikes.empty:
                ax.plot(spikes.index, spikes[col], 'r*', markersize=14, label='Detected Spike')
                for idx, row in spikes.iterrows():
                    ax.annotate(
                        'Spike!', xy=(idx, row[col]), xytext=(0, 20),
                        textcoords='offset points', ha='center', color='red', weight='bold',
                        arrowprops=dict(arrowstyle='->', color='red')
                    )

        # Value labels
        _add_value_labels(ax, x_values, df[col], format_indian_currency, is_pct)

        ax.legend(loc='upper left' if is_pct else 'best')
        
        # Save PNG and PDF
        out_path = f"sales_eda/graphs/business/{filename}"
        plt.savefig(f"{out_path}.png")
        plt.savefig(f"{out_path}.pdf")
        plt.close()

def generate_executive_dashboard(df):
    """Combine key metrics into a single presentation-ready image."""
    logging.info("Generating Executive Dashboard...")
    
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 2, height_ratios=[1, 1, 0.5], hspace=0.4)
    formatter = FuncFormatter(format_indian_currency)

    # 1. Net Sales Trend (Spans both cols)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df['date_obj'], df['net_sale_amt'], marker='o', color='#2980b9', lw=3, label='Net Sales')
    ax1.plot(df['date_obj'], df['rolling_3m_avg'], linestyle='--', color='#e67e22', lw=2, label='3M Rolling Avg')
    ax1.set_title("Executive View: Net Sales Trend vs Rolling Average")
    ax1.yaxis.set_major_formatter(formatter)
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend()

    # Highlight spikes on dashboard
    spikes = df[df['is_spike']]
    ax1.scatter(spikes['date_obj'], spikes['net_sale_amt'], color='red', s=150, zorder=5, marker='*', label='Spike')

    # 2. MoM Growth %
    ax2 = fig.add_subplot(gs[1, 0])
    bars = ax2.bar(df['date_obj'], df['mom_growth_pct'], color=np.where(df['mom_growth_pct'] > 0, '#27ae60', '#e74c3c'))
    ax2.set_title("Month-over-Month Growth (%)")
    ax2.tick_params(axis='x', rotation=45)
    ax2.axhline(0, color='black', linewidth=1)

    # 3. Total Returns
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(df['date_obj'], df['total_returns_amt'], color='#8e44ad', marker='s', lw=2)
    ax3.set_title("Monthly Returns Trend")
    ax3.yaxis.set_major_formatter(formatter)
    ax3.tick_params(axis='x', rotation=45)

    # 4. KPI Table
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('tight')
    ax4.axis('off')
    
    kpi_data = [
        ["Total Net Sales", format_indian_currency(df['net_sale_amt'].sum())],
        ["Total Gross Sales", format_indian_currency(df['gross_sale_amt'].sum())],
        ["Total Returns", format_indian_currency(df['total_returns_amt'].sum())],
        ["Avg Monthly Net Sales", format_indian_currency(df['net_sale_amt'].mean())],
        ["Peak Month", f"{df.loc[df['net_sale_amt'].idxmax(), 'Month-Year']} ({format_indian_currency(df['net_sale_amt'].max())})"]
    ]
    
    table = ax4.table(cellText=kpi_data, colLabels=["Business Metric", "Value"], loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 2)
    
    # Style Table
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#34495e')
        else:
            cell.set_facecolor('#f2f5f8')

    plt.suptitle("WEATHER DRIVEN PHARMACEUTICAL SALES - EXECUTIVE DASHBOARD", fontsize=24, fontweight='bold', y=0.98)
    plt.savefig("sales_eda/graphs/business/Executive_Monthly_Dashboard.png", dpi=600, bbox_inches='tight')
    plt.close()

# =============================================================================
# EXCEL & TEXT REPORT GENERATION
# =============================================================================

def generate_excel_report(df):
    """Generate Excel workbook with multiple analytical sheets."""
    logging.info("Exporting data to Excel...")
    output_path = "sales_eda/excel/business/Monthly_Sales_Summary.xlsx"
    
    kpi_dict = {
        "Metric": [
            "Total Net Sales", "Total Gross Sales", "Total Returns", 
            "Avg Monthly Sales", "Median Monthly Sales", 
            "Max Monthly Sales", "Min Monthly Sales"
        ],
        "Value": [
            df['net_sale_amt'].sum(), df['gross_sale_amt'].sum(), df['total_returns_amt'].sum(),
            df['net_sale_amt'].mean(), df['net_sale_amt'].median(),
            df['net_sale_amt'].max(), df['net_sale_amt'].min()
        ]
    }
    kpi_df = pd.DataFrame(kpi_dict)

    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        kpi_df.to_excel(writer, sheet_name='Executive KPI', index=False)
        df[['Month-Year', 'gross_sale_amt', 'net_sale_amt', 'total_returns_amt']].to_excel(writer, sheet_name='Monthly Summary', index=False)
        df[['Month-Year', 'net_sale_amt', 'prev_month_net', 'mom_growth_pct']].to_excel(writer, sheet_name='Growth Analysis', index=False)
        df[['Month-Year', 'net_sale_amt', 'rolling_3m_avg', 'rolling_3m_std']].to_excel(writer, sheet_name='Rolling Average', index=False)
        df[df['is_spike']][['Month-Year', 'net_sale_amt', 'z_score']].to_excel(writer, sheet_name='Spike Detection', index=False)

def generate_insights_report(df):
    """Generate executive text summary mapping business interpretations."""
    logging.info("Generating Business Insights Text Report...")
    output_path = "reports/phase_2/Monthly_Business_Insights.txt"

    total_months = len(df)
    total_net = df['net_sale_amt'].sum()
    max_month_idx = df['net_sale_amt'].idxmax()
    min_month_idx = df['net_sale_amt'].idxmin()
    max_growth_idx = df['mom_growth_pct'].idxmax()
    min_growth_idx = df['mom_growth_pct'].idxmin()
    
    spikes_list = df[df['is_spike']]['Month-Year'].tolist()
    spike_str = ", ".join(spikes_list) if spikes_list else "None detected"

    report_content = f"""====================================================
EXECUTIVE BUSINESS INSIGHTS & SUMMARY
====================================================

ANALYSIS PERIOD
----------------------------------------------------
Total Months Analyzed: {total_months}
Start Month: {df['Month-Year'].iloc[0]}
End Month:   {df['Month-Year'].iloc[-1]}

CORE BUSINESS METRICS
----------------------------------------------------
Total Net Sales:   {format_indian_currency(total_net)}
Total Gross Sales: {format_indian_currency(df['gross_sale_amt'].sum())}
Total Returns:     {format_indian_currency(df['total_returns_amt'].sum())}
Avg Monthly Sales: {format_indian_currency(df['net_sale_amt'].mean())}

PERFORMANCE EXTREMES
----------------------------------------------------
Highest Sales Month: {df.loc[max_month_idx, 'Month-Year']} ({format_indian_currency(df.loc[max_month_idx, 'net_sale_amt'])})
Lowest Sales Month:  {df.loc[min_month_idx, 'Month-Year']} ({format_indian_currency(df.loc[min_month_idx, 'net_sale_amt'])})
Highest Growth:      {df.loc[max_growth_idx, 'Month-Year']} ({df.loc[max_growth_idx, 'mom_growth_pct']:.2f}%)
Largest Decline:     {df.loc[min_growth_idx, 'Month-Year']} ({df.loc[min_growth_idx, 'mom_growth_pct']:.2f}%)

STATISTICAL ANOMALIES & SPIKES
----------------------------------------------------
Spike Months (Z-Score > 1.5): {spike_str}

====================================================
BUSINESS INTERPRETATION
====================================================

Executive Summary:
The dataset reveals a cumulative net revenue of {format_indian_currency(total_net)} across the {total_months}-month period. The variance between gross and net indicates the total return volume that needs supply-chain optimization.

Top Performing Month:
{df.loc[max_month_idx, 'Month-Year']} drove the highest revenue, indicating strong seasonal demand or successful market interventions.

Worst Performing Month:
{df.loc[min_month_idx, 'Month-Year']} represented the lowest trough. Historical events, stock-outs, or adverse weather parameters during this period require immediate root-cause analysis.

Business Growth Trend:
The rolling 3-month average indicates a smoothing of volatility, but sharp month-over-month variances of up to {df.loc[max_growth_idx, 'mom_growth_pct']:.1f}% require more robust forecasting models.

Seasonality Observation:
Fluctuations align with potential weather-driven patterns; integrating secondary weather datasets will isolate non-commercial drivers.

Potential Sales Spike:
Marked spikes in {spike_str} suggest anomalous demand surges.

Potential Business Concern:
Returns total {format_indian_currency(df['total_returns_amt'].sum())}. A high ratio of expiry/breakage against gross sales negatively impacts final margins and indicates inefficient inventory lifecycle management.

Business Recommendations:
1. Implement dynamic buffer stock for months preceding historical spikes.
2. Investigate the return pipeline, particularly targeting the worst-performing months to minimize margin bleed.
3. Overlay meteorological data strictly on the spike months to confirm weather causality.
====================================================
"""

    with open(output_path, "w", encoding='utf-8') as file:
        file.write(report_content)

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    logging.info("Starting Business Analytics Pipeline...")
    
    # 1. Setup ecosystem
    setup_directories()
    
    try:
        # 2. Data Loading & Cleaning
        # Note: If running without the excel file, ensure you place data/Sale_Details.xlsx
        # conforming to the column specifications in the working directory.
        raw_data = load_and_preprocess(INPUT_FILE)
        
        # 3. KPI & Metric Calculations
        business_metrics = aggregate_monthly_metrics(raw_data)
        
        # 4. Generate Exports
        generate_individual_graphs(business_metrics)
        generate_executive_dashboard(business_metrics)
        generate_excel_report(business_metrics)
        generate_insights_report(business_metrics)
        
        logging.info("Pipeline execution completed successfully.")
        
    except Exception as e:
        logging.error(f"Pipeline execution failed: {str(e)}")