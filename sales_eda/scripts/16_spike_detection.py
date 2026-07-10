"""
16_spike_detection.py
=============================================================================
Project: Weather Driven Pharmaceutical Sales Analytics & Forecasting
Module:  Executive Spike & Volatility Detection
Author:  Principal Data Scientist & Time Series Analyst
=============================================================================
Description:
    End-to-end time series pipeline for detecting, classifying, and 
    visualizing spikes in monthly sales and return metrics. Utilizes 
    3-month rolling averages, moving standard deviations, and Z-scores.
    Generates presentation-grade dashboards and automated business insights.
=============================================================================
"""

import os
import logging
import numpy as np
import pandas as pd
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
    "sales_eda/excel/spikes/",
    "sales_eda/graphs/spikes/",
    "reports/phase_2/"
]

# Matplotlib Global Parameters for Presentation Quality
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'sans-serif'],
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

# Corporate Color Palette
COLORS = {
    'primary': '#1f4e79',      # Corporate Navy
    'secondary': '#a5a5a5',    # Gray
    'rolling': '#f39c12',      # Orange
    'spike_high': '#c00000',   # Dark Red
    'spike_low': '#8e44ad',    # Purple
    'background': '#f8f9fa'    # Light Gray
}

# Metrics to Analyze
TARGET_METRICS = [
    'net_sale_amt', 'gross_sale_amt', 
    'net_sale_qty', 'gross_sale_qty', 
    'total_returns_amt'
]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def setup_directories():
    """Create all required output directories safely."""
    for directory in BASE_DIRS:
        os.makedirs(directory, exist_ok=True)
    logging.info("Output directories verified and ready.")

def format_number(num, pos=None):
    """Format large numbers for executive readability (Indian Numbering)."""
    if pd.isna(num): return "0.00"
    is_neg = num < 0
    abs_num = abs(num)
    if abs_num >= 1e7: return f"{'-' if is_neg else ''}{abs_num/1e7:.2f} Cr"
    elif abs_num >= 1e5: return f"{'-' if is_neg else ''}{abs_num/1e5:.2f} L"
    elif abs_num >= 1e3: return f"{'-' if is_neg else ''}{abs_num/1e3:.2f} K"
    return f"{num:.2f}"

def classify_spike(z_score):
    """Automatically classify the severity of a spike based on Rolling Z-Score."""
    if z_score >= 2.5:
        return 'Very High'
    elif z_score >= 1.5:
        return 'High'
    elif z_score <= -2.5:
        return 'Very Low'
    elif z_score <= -1.5:
        return 'Low'
    else:
        return 'Normal'

# =============================================================================
# DATA PROCESSING & TIME SERIES ANALYSIS
# =============================================================================

def load_and_aggregate(filepath):
    """Load dataset, calculate returns, and aggregate monthly."""
    logging.info(f"Loading data from {filepath}...")
    try:
        df = pd.read_excel(filepath)
        
        # Calculate Total Returns
        df['total_returns_amt'] = df.get('fresh_ret_amt', 0) + df.get('expiry_amt', 0) + df.get('brkg_amt', 0)
        
        # Ensure billing_month exists
        if 'billing_month' not in df.columns:
            raise ValueError("Required column 'billing_month' is missing.")
            
        # Convert YYYYMM to Datetime for accurate sorting and plotting
        df['billing_month'] = df['billing_month'].astype(str)
        df['Date'] = pd.to_datetime(df['billing_month'], format='%Y%m', errors='coerce')
        df = df.dropna(subset=['Date'])
        df['Month_Year'] = df['Date'].dt.strftime('%b-%Y')
        
        # Aggregate
        monthly_df = df.groupby(['Date', 'Month_Year'])[TARGET_METRICS].sum().reset_index()
        monthly_df = monthly_df.sort_values('Date').reset_index(drop=True)
        
        logging.info(f"Successfully aggregated {len(monthly_df)} months of data.")
        return monthly_df
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise

def detect_spikes(df, window=3):
    """Calculate rolling statistics, Z-scores, and classify anomalies."""
    logging.info("Executing Spike Detection Algorithms...")
    
    analyzed_dfs = []
    
    for metric in TARGET_METRICS:
        if metric not in df.columns:
            continue
            
        temp_df = df[['Date', 'Month_Year', metric]].copy()
        temp_df.rename(columns={metric: 'Actual_Value'}, inplace=True)
        temp_df['Metric_Name'] = metric
        
        # Rolling Statistics (3 Month Window)
        temp_df['Rolling_Mean'] = temp_df['Actual_Value'].rolling(window=window, min_periods=1).mean()
        temp_df['Rolling_Std'] = temp_df['Actual_Value'].rolling(window=window, min_periods=1).std().fillna(1e-6)
        
        # Rolling Z-Score (Local anomaly detection)
        # We replace 0 std with a small number to avoid division by zero
        temp_df['Rolling_Std'] = temp_df['Rolling_Std'].replace(0, 1e-6)
        temp_df['Rolling_Z_Score'] = (temp_df['Actual_Value'] - temp_df['Rolling_Mean']) / temp_df['Rolling_Std']
        
        # Global Z-Score (Overall anomaly detection)
        global_mean = temp_df['Actual_Value'].mean()
        global_std = temp_df['Actual_Value'].std()
        temp_df['Global_Z_Score'] = (temp_df['Actual_Value'] - global_mean) / (global_std if global_std > 0 else 1e-6)
        
        # Classification
        temp_df['Spike_Class'] = temp_df['Rolling_Z_Score'].apply(classify_spike)
        
        # Magnitude (Absolute percentage variance from rolling mean)
        temp_df['Variance_From_Trend_%'] = np.where(
            temp_df['Rolling_Mean'] > 0,
            ((temp_df['Actual_Value'] - temp_df['Rolling_Mean']) / temp_df['Rolling_Mean']) * 100,
            0
        )
        
        analyzed_dfs.append(temp_df)
        
    final_df = pd.concat(analyzed_dfs, ignore_index=True)
    return final_df

def extract_kpis(df):
    """Extract executive summary KPIs from the analyzed data."""
    net_sales_df = df[df['Metric_Name'] == 'net_sale_amt']
    
    kpis = {
        "Total Months Analyzed": len(net_sales_df),
        "Highest Sales Month": net_sales_df.loc[net_sales_df['Actual_Value'].idxmax(), 'Month_Year'],
        "Lowest Sales Month": net_sales_df.loc[net_sales_df['Actual_Value'].idxmin(), 'Month_Year'],
        "Total Net Sales": net_sales_df['Actual_Value'].sum(),
        "Avg Monthly Sales": net_sales_df['Actual_Value'].mean(),
        "Total Volatile Months (Spikes)": len(net_sales_df[net_sales_df['Spike_Class'].isin(['High', 'Very High'])]),
        "Total Depressed Months (Dips)": len(net_sales_df[net_sales_df['Spike_Class'].isin(['Low', 'Very Low'])])
    }
    return kpis

# =============================================================================
# PRESENTATION-QUALITY VISUALIZATIONS
# =============================================================================

def plot_spike_visualizations(df):
    """Generate high-resolution trend and spike analysis charts."""
    logging.info("Generating Spike Visualization Graphs...")
    
    out_dir = "sales_eda/graphs/spikes/"
    formatter = FuncFormatter(format_number)
    
    # Isolate Net Sales for primary charting
    net_df = df[df['Metric_Name'] == 'net_sale_amt'].copy()
    
    # 1. Primary Monthly Trend with Spike Highlights
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # Base Lines
    ax.plot(net_df['Date'], net_df['Actual_Value'], marker='o', color=COLORS['primary'], linewidth=2.5, label='Actual Net Sales')
    ax.plot(net_df['Date'], net_df['Rolling_Mean'], linestyle='--', color=COLORS['rolling'], linewidth=2, label='3-Month Rolling Trend')
    
    # Highlight Spikes
    high_spikes = net_df[net_df['Spike_Class'].isin(['High', 'Very High'])]
    low_spikes = net_df[net_df['Spike_Class'].isin(['Low', 'Very Low'])]
    
    if not high_spikes.empty:
        ax.scatter(high_spikes['Date'], high_spikes['Actual_Value'], color=COLORS['spike_high'], s=150, zorder=5, label='High Spike')
        for _, row in high_spikes.iterrows():
            ax.annotate(f"Spike\n+{row['Variance_From_Trend_%']:.0f}%", 
                        xy=(row['Date'], row['Actual_Value']), xytext=(0, 15), 
                        textcoords='offset points', ha='center', color=COLORS['spike_high'], weight='bold')

    if not low_spikes.empty:
        ax.scatter(low_spikes['Date'], low_spikes['Actual_Value'], color=COLORS['spike_low'], s=150, zorder=5, label='Deep Dip')
        for _, row in low_spikes.iterrows():
            ax.annotate(f"Dip\n{row['Variance_From_Trend_%']:.0f}%", 
                        xy=(row['Date'], row['Actual_Value']), xytext=(0, -25), 
                        textcoords='offset points', ha='center', color=COLORS['spike_low'], weight='bold')

    ax.set_title("Executive Trend Analysis: Monthly Net Sales & Spike Detection", pad=20)
    ax.set_ylabel("Net Sales (INR)")
    ax.yaxis.set_major_formatter(formatter)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
    plt.xticks(rotation=45)
    ax.legend(loc='upper left', frameon=True, shadow=True)
    plt.savefig(os.path.join(out_dir, "01_Monthly_Trend_Spikes.png"))
    plt.close()

def plot_executive_dashboard(df, kpis):
    """Generate the comprehensive Executive Spike Dashboard."""
    logging.info("Generating Executive Spike Dashboard...")
    
    out_dir = "sales_eda/graphs/spikes/"
    formatter = FuncFormatter(format_number)
    net_df = df[df['Metric_Name'] == 'net_sale_amt'].copy()
    
    fig = plt.figure(figsize=(24, 16))
    fig.patch.set_facecolor(COLORS['background'])
    gs = GridSpec(3, 2, figure=fig, height_ratios=[0.5, 1.5, 1], wspace=0.2, hspace=0.4)
    
    # 1. Top KPI Cards
    ax_kpi = fig.add_subplot(gs[0, :])
    ax_kpi.axis('off')
    
    kpi_data = [
        ("TOTAL NET SALES", format_number(kpis['Total Net Sales']), COLORS['primary']),
        ("HIGHEST SALES MONTH", kpis['Highest Sales Month'], COLORS['spike_high']),
        ("LOWEST SALES MONTH", kpis['Lowest Sales Month'], COLORS['spike_low']),
        ("IDENTIFIED SPIKES (>1.5σ)", str(kpis['Total Volatile Months (Spikes)']), COLORS['rolling'])
    ]
    
    for i, (title, val, color) in enumerate(kpi_data):
        x_offset = i * 0.25
        rect = plt.Rectangle((x_offset + 0.02, 0.1), 0.21, 0.8, fill=True, color='white', 
                             edgecolor=color, linewidth=2, transform=ax_kpi.transAxes)
        ax_kpi.add_patch(rect)
        ax_kpi.text(x_offset + 0.125, 0.65, title, fontsize=14, fontweight='bold', color=COLORS['secondary'], ha='center', transform=ax_kpi.transAxes)
        ax_kpi.text(x_offset + 0.125, 0.35, val, fontsize=28, fontweight='bold', color=color, ha='center', transform=ax_kpi.transAxes)

    # 2. Main Trend View (Mid Row, spans 2 cols)
    ax_trend = fig.add_subplot(gs[1, :])
    ax_trend.fill_between(net_df['Date'], net_df['Actual_Value'], color=COLORS['primary'], alpha=0.1)
    ax_trend.plot(net_df['Date'], net_df['Actual_Value'], marker='o', color=COLORS['primary'], linewidth=2)
    ax_trend.plot(net_df['Date'], net_df['Rolling_Mean'], linestyle='--', color=COLORS['rolling'], linewidth=2)
    
    spikes = net_df[net_df['Spike_Class'].isin(['High', 'Very High'])]
    if not spikes.empty:
        ax_trend.scatter(spikes['Date'], spikes['Actual_Value'], color=COLORS['spike_high'], s=150, zorder=5, marker='*')
        for _, row in spikes.iterrows():
            ax_trend.annotate(f"{row['Month_Year']}\n{format_number(row['Actual_Value'])}", 
                        xy=(row['Date'], row['Actual_Value']), xytext=(0, 20), 
                        textcoords='offset points', ha='center', color='black', weight='bold',
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor=COLORS['spike_high'], pad=2))

    ax_trend.set_title("Revenue Volatility & Smoothing (Net Sales vs 3M Rolling Average)", weight='bold')
    ax_trend.yaxis.set_major_formatter(formatter)
    ax_trend.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
    
    # 3. Z-Score Oscillator (Bottom Left)
    ax_osc = fig.add_subplot(gs[2, 0])
    bars = ax_osc.bar(net_df['Month_Year'], net_df['Rolling_Z_Score'], 
                      color=np.where(net_df['Rolling_Z_Score'] > 0, COLORS['primary'], COLORS['secondary']))
    
    ax_osc.axhline(1.5, color=COLORS['spike_high'], linestyle='--', linewidth=1.5, label='High Spike Threshold (+1.5)')
    ax_osc.axhline(-1.5, color=COLORS['spike_low'], linestyle='--', linewidth=1.5, label='Deep Dip Threshold (-1.5)')
    ax_osc.set_title("Volatility Oscillator (Rolling Z-Score)", weight='bold')
    ax_osc.tick_params(axis='x', rotation=45)
    ax_osc.legend(loc='lower right')

    # 4. Spike Summary Table (Bottom Right)
    ax_tbl = fig.add_subplot(gs[2, 1])
    ax_tbl.axis('tight')
    ax_tbl.axis('off')
    
    # Prep Table Data: Top 6 anomalies
    anomalies = net_df[net_df['Spike_Class'] != 'Normal'].sort_values('Rolling_Z_Score', key=abs, ascending=False).head(6)
    if not anomalies.empty:
        tbl_df = anomalies[['Month_Year', 'Actual_Value', 'Variance_From_Trend_%', 'Spike_Class']].copy()
        tbl_df['Actual_Value'] = tbl_df['Actual_Value'].apply(lambda x: format_number(x))
        tbl_df['Variance_From_Trend_%'] = tbl_df['Variance_From_Trend_%'].apply(lambda x: f"{x:+.1f}%")
        
        table = ax_tbl.table(cellText=tbl_df.values, colLabels=["Month", "Net Sales", "Trend Variance", "Classification"], loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(13)
        table.scale(1, 2)
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor(COLORS['primary'])
            else:
                if "High" in tbl_df.iloc[row-1]['Spike_Class']:
                    cell.set_facecolor('#fadbd8') # Light red
                elif "Low" in tbl_df.iloc[row-1]['Spike_Class']:
                    cell.set_facecolor('#e8daef') # Light purple
    else:
        ax_tbl.text(0.5, 0.5, "No Volatile Anomalies Detected", ha='center', va='center', fontsize=18, color=COLORS['secondary'])

    plt.suptitle("EXECUTIVE TIME SERIES & SPIKE DETECTION DASHBOARD", fontsize=28, fontweight='bold', color=COLORS['primary'], y=0.98)
    plt.savefig(os.path.join(out_dir, "02_Executive_Spike_Dashboard.png"), facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()

# =============================================================================
# EXCEL & TEXT REPORT GENERATION
# =============================================================================

def generate_excel_report(df, kpis):
    """Export volatile time series analysis to Excel."""
    logging.info("Exporting Time Series Data to Excel...")
    output_path = "sales_eda/excel/spikes/Spike_Detection.xlsx"
    
    # Re-pivot data for a cleaner monthly summary view
    monthly_pivot = df.pivot(index=['Date', 'Month_Year'], columns='Metric_Name', values='Actual_Value').reset_index()
    monthly_pivot = monthly_pivot.sort_values('Date').drop(columns=['Date'])
    
    # Filter only anomalies across all metrics
    spikes_df = df[df['Spike_Class'] != 'Normal'].sort_values(['Metric_Name', 'Date'])
    
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        pd.DataFrame(list(kpis.items()), columns=['KPI', 'Value']).to_excel(writer, sheet_name='Executive KPI', index=False)
        spikes_df[['Metric_Name', 'Month_Year', 'Actual_Value', 'Rolling_Mean', 'Variance_From_Trend_%', 'Rolling_Z_Score', 'Spike_Class']].to_excel(writer, sheet_name='Spike Summary', index=False)
        monthly_pivot.to_excel(writer, sheet_name='Monthly Summary', index=False)
        
        # Raw rolling stats dump
        df[['Metric_Name', 'Month_Year', 'Actual_Value', 'Rolling_Mean', 'Rolling_Std', 'Rolling_Z_Score']].to_excel(writer, sheet_name='Rolling Statistics', index=False)

        # Formatting
        workbook = writer.book
        header_format = workbook.add_format({'bold': True, 'bg_color': COLORS['primary'], 'font_color': 'white'})
        for sheet in writer.sheets:
            writer.sheets[sheet].set_row(0, None, header_format)
            writer.sheets[sheet].set_column('A:H', 18)

def generate_insights_report(df, kpis):
    """Generate strategic business interpretations of temporal volatility."""
    logging.info("Generating Business Insights Text Report...")
    output_path = "reports/phase_2/Spike_Business_Insights.txt"

    net_df = df[df['Metric_Name'] == 'net_sale_amt']
    spikes = net_df[net_df['Spike_Class'].isin(['High', 'Very High'])]
    dips = net_df[net_df['Spike_Class'].isin(['Low', 'Very Low'])]
    
    max_spike_str = f"{spikes.iloc[0]['Month_Year']} (+{spikes.iloc[0]['Variance_From_Trend_%']:.1f}% vs trend)" if not spikes.empty else "None detected"
    max_dip_str = f"{dips.iloc[0]['Month_Year']} ({dips.iloc[0]['Variance_From_Trend_%']:.1f}% vs trend)" if not dips.empty else "None detected"

    report_content = f"""====================================================
EXECUTIVE TIME SERIES & SPIKE DETECTION INSIGHTS
====================================================

OVERALL TREND ANALYSIS
----------------------------------------------------
Analysis Period:          {kpis['Total Months Analyzed']} Months
Highest Sales Volume:     {kpis['Highest Sales Month']}
Lowest Sales Volume:      {kpis['Lowest Sales Month']}
Average Monthly Run Rate: {format_number(kpis['Avg Monthly Sales'])}

VOLATILITY & ANOMALIES
----------------------------------------------------
Months with Positive Spikes: {len(spikes)}
Highest Magnitude Spike:     {max_spike_str}

Months with Severe Dips:     {len(dips)}
Lowest Magnitude Dip:        {max_dip_str}

====================================================
BUSINESS INTERPRETATION & CAUSALITY HYPOTHESES
====================================================

1. Possible Seasonality vs Actual Spikes:
By utilizing a 3-Month Rolling Average, this model effectively filters out gradual seasonal climbs. Events flagged as 'High' or 'Very High' spikes represent abrupt, anomalous surges in demand that break standard short-term trendlines.

2. Potential Promotion Impact:
If a 'Very High' spike aligns directly with a known marketing intervention, institutional tender win, or pricing discount, it proves the elasticity and success of that campaign. 

3. Potential Inventory Effect:
Deep dips ('Very Low') immediately following 'Very High' spikes typically indicate aggressive channel stuffing (forward buying by distributors) resulting in inventory overhang the subsequent month. Alternatively, a dip may reflect severe stock-outs resulting from poor buffer capacity during the preceding spike.

====================================================
STRATEGIC RECOMMENDATIONS
====================================================
* Supply Chain Action: Identify the specific SKUs that drove the {max_spike_str} anomaly. Dynamic buffer inventory must be allocated ahead of this specific month in the upcoming financial year to prevent out-of-stock scenarios.
* Root Cause Audit: Investigate the depressed volume during {max_dip_str}. If this was driven by adverse weather conditions disrupting logistics, overlay meteorological data to build a predictive shock-warning system.
====================================================
"""
    with open(output_path, "w", encoding='utf-8') as file:
        file.write(report_content)

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    logging.info("Starting Executive Spike Detection Pipeline...")
    
    setup_directories()
    
    try:
        # Load and aggregate
        monthly_data = load_and_aggregate(INPUT_FILE)
        
        # Analyze Series
        analyzed_data = detect_spikes(monthly_data, window=3)
        executive_kpis = extract_kpis(analyzed_data)
        
        # Output Generation
        plot_spike_visualizations(analyzed_data)
        plot_executive_dashboard(analyzed_data, executive_kpis)
        generate_excel_report(analyzed_data, executive_kpis)
        generate_insights_report(analyzed_data, executive_kpis)
        
        logging.info("Spike Detection Analysis completed successfully.")
        
    except Exception as e:
        logging.error(f"Pipeline execution failed: {str(e)}")