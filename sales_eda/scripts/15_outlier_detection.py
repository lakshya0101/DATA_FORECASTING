"""
15_outlier_detection.py
=============================================================================
Project: Weather Driven Pharmaceutical Sales Analytics & Forecasting
Module:  Executive Outlier & Anomaly Detection Analysis
Author:  Principal Data Scientist & Senior BI Consultant
=============================================================================
Description:
    End-to-end pipeline for detecting anomalies in financial and volumetric 
    sales data. Utilizes IQR, standard Z-Score, Modified Z-Score (MAD), 
    and Machine Learning (Isolation Forest). Generates Fortune 500 tier 
    visualizations, multidimensional Excel reports, and business insights.
=============================================================================
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import IsolationForest
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
    "sales_eda/excel/outliers/",
    "sales_eda/graphs/outliers/",
    "reports/phase_2/"
]

# Target Columns for Outlier Detection
TARGET_COLUMNS = [
    "gross_sale_amt", "net_sale_amt", 
    "gross_sale_qty", "net_sale_qty", 
    "fresh_ret_amt", "expiry_amt", "brkg_amt"
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
    'primary': '#2c3e50',      # Corporate Navy
    'secondary': '#bdc3c7',    # Silver/Grey
    'normal': '#3498db',       # Light Blue
    'outlier': '#e74c3c',      # Alizarin Red (Anomaly)
    'iqr': '#f39c12',          # Orange
    'zscore': '#9b59b6',       # Purple
    'iforest': '#27ae60',      # Green
    'background': '#f8f9fa'    # Light Grey
}

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
    if pd.isna(num): return "0"
    is_neg = num < 0
    abs_num = abs(num)
    if abs_num >= 1e7: return f"{'-' if is_neg else ''}{abs_num/1e7:.2f} Cr"
    elif abs_num >= 1e5: return f"{'-' if is_neg else ''}{abs_num/1e5:.2f} L"
    elif abs_num >= 1e3: return f"{'-' if is_neg else ''}{abs_num/1e3:.2f} K"
    return f"{num:.2f}"

# =============================================================================
# DATA PROCESSING & ANOMALY DETECTION
# =============================================================================

def load_data(filepath):
    """Load and validate dataset."""
    logging.info(f"Loading data from {filepath}...")
    try:
        df = pd.read_excel(filepath)
        valid_cols = [col for col in TARGET_COLUMNS if col in df.columns]
        if not valid_cols:
            raise ValueError("No target columns found in the dataset.")
        
        # Fill missing values with 0 to allow algorithm execution
        df[valid_cols] = df[valid_cols].fillna(0)
        logging.info(f"Successfully loaded {len(df)} records.")
        return df, valid_cols
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise

def apply_outlier_methods(df, columns):
    """Execute all 4 outlier detection methodologies per column."""
    logging.info("Executing Outlier Detection Algorithms (IQR, Z, Mod-Z, I-Forest)...")
    
    outlier_results = {
        'IQR': {},
        'ZScore': {},
        'ModZScore': {},
        'IsolationForest': {}
    }
    
    summary_list = []
    
    for col in columns:
        data = df[col].values
        total_records = len(data)
        
        # 1. IQR Method
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        iqr_outliers = (data < lower_bound) | (data > upper_bound)
        outlier_results['IQR'][col] = iqr_outliers
        
        # 2. Standard Z-Score (|Z| > 3)
        z_scores = np.abs(stats.zscore(data))
        z_outliers = z_scores > 3
        outlier_results['ZScore'][col] = z_outliers
        
        # 3. Modified Z-Score (Robust to extreme outliers using MAD)
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        if mad == 0: mad = 1e-6  # Prevent division by zero
        mod_z_scores = 0.6745 * (data - median) / mad
        mod_z_outliers = np.abs(mod_z_scores) > 3.5
        outlier_results['ModZScore'][col] = mod_z_outliers
        
        # 4. Isolation Forest (Machine Learning)
        # Using a conservative contamination rate of 1%
        iso = IsolationForest(contamination=0.01, random_state=42, n_jobs=-1)
        iso_preds = iso.fit_predict(data.reshape(-1, 1))
        iso_outliers = iso_preds == -1
        outlier_results['IsolationForest'][col] = iso_outliers
        
        # Summarize Findings
        summary_list.append({
            'Metric': col,
            'Total Records': total_records,
            'IQR Outliers': iqr_outliers.sum(),
            'Z-Score Outliers': z_outliers.sum(),
            'Mod Z-Score Outliers': mod_z_outliers.sum(),
            'I-Forest Outliers': iso_outliers.sum(),
            'Max Outlier Value': data[iqr_outliers].max() if iqr_outliers.sum() > 0 else np.nan,
            'Min Outlier Value': data[iqr_outliers].min() if iqr_outliers.sum() > 0 else np.nan
        })
        
    summary_df = pd.DataFrame(summary_list)
    return outlier_results, summary_df

# =============================================================================
# PRESENTATION-QUALITY VISUALIZATIONS
# =============================================================================

def plot_anomaly_visualizations(df, columns, outlier_results):
    """Generate high-resolution Boxplots, Scatter Plots, and Distributions."""
    logging.info("Generating Anomaly Visualizations...")
    
    out_dir = "sales_eda/graphs/outliers/"
    formatter = FuncFormatter(format_number)
    
    # We will chart the primary business metric (Net Sales Amt) extensively, 
    # and provide standard charts for the rest.
    main_col = 'net_sale_amt' if 'net_sale_amt' in columns else columns[0]
    
    for col in columns:
        data = df[col]
        iqr_mask = outlier_results['IQR'][col]
        
        if iqr_mask.sum() == 0: continue
        
        # 1. Boxplot (Standard Outlier View)
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(x=data, color=COLORS['normal'], flierprops=dict(markerfacecolor=COLORS['outlier'], marker='o', markersize=8), ax=ax)
        ax.set_title(f"Boxplot Outlier Detection: {col}", pad=15)
        ax.set_xlabel("Value")
        ax.xaxis.set_major_formatter(formatter)
        plt.savefig(os.path.join(out_dir, f"{col}_boxplot.png"))
        plt.close()

        # 2. Outlier Scatter Plot (Index vs Value)
        fig, ax = plt.subplots(figsize=(16, 8))
        x_idx = np.arange(len(data))
        ax.scatter(x_idx[~iqr_mask], data[~iqr_mask], color=COLORS['secondary'], alpha=0.5, label='Normal Baseline', s=20)
        ax.scatter(x_idx[iqr_mask], data[iqr_mask], color=COLORS['outlier'], alpha=0.9, label='Detected Anomaly (IQR)', s=60, edgecolor='black')
        
        ax.set_title(f"Time/Index Series Anomaly Scatter: {col}", pad=15)
        ax.set_ylabel("Metric Value")
        ax.set_xlabel("Record Index")
        ax.yaxis.set_major_formatter(formatter)
        ax.legend(loc='upper right', frameon=True)
        plt.savefig(os.path.join(out_dir, f"{col}_scatter_anomalies.png"))
        plt.close()

def plot_executive_dashboard(df, summary_df, outlier_results, main_col='net_sale_amt'):
    """Generate the comprehensive Executive Outlier Dashboard."""
    logging.info("Generating Executive Outlier Dashboard...")
    
    if main_col not in df.columns: main_col = df.columns[0]
    
    out_dir = "sales_eda/graphs/outliers/"
    formatter = FuncFormatter(format_number)
    
    fig = plt.figure(figsize=(24, 16))
    fig.patch.set_facecolor(COLORS['background'])
    gs = GridSpec(3, 2, figure=fig, height_ratios=[0.5, 1.2, 1.2], wspace=0.2, hspace=0.4)
    
    # 1. KPI Cards (Top Row)
    kpi_ax = fig.add_subplot(gs[0, :])
    kpi_ax.axis('off')
    
    total_records = summary_df.iloc[0]['Total Records']
    main_iqr_outliers = summary_df[summary_df['Metric'] == main_col]['IQR Outliers'].values[0]
    outlier_pct = (main_iqr_outliers / total_records) * 100
    highest_anomaly = summary_df[summary_df['Metric'] == main_col]['Max Outlier Value'].values[0]

    kpis = [
        ("TOTAL RECORDS EVALUATED", f"{total_records:,}", COLORS['primary']),
        (f"ANOMALIES IN {main_col.upper()}", f"{main_iqr_outliers:,} ({outlier_pct:.1f}%)", COLORS['outlier']),
        ("HIGHEST DETECTED SPIKE", format_number(highest_anomaly), COLORS['iqr'])
    ]
    
    for i, (title, val, color) in enumerate(kpis):
        x_offset = i * 0.33
        rect = plt.Rectangle((x_offset + 0.02, 0.1), 0.29, 0.8, fill=True, color='white', 
                             edgecolor=color, linewidth=2, transform=kpi_ax.transAxes)
        kpi_ax.add_patch(rect)
        kpi_ax.text(x_offset + 0.165, 0.65, title, fontsize=14, fontweight='bold', color=COLORS['secondary'], ha='center', transform=kpi_ax.transAxes)
        kpi_ax.text(x_offset + 0.165, 0.35, val, fontsize=30, fontweight='bold', color=color, ha='center', transform=kpi_ax.transAxes)

    # 2. Method Comparison Bar Chart (Mid Left)
    ax_bar = fig.add_subplot(gs[1, 0])
    main_summary = summary_df[summary_df['Metric'] == main_col].iloc[0]
    methods = ['IQR', 'Z-Score', 'Mod Z-Score', 'Isolation Forest']
    counts = [main_summary['IQR Outliers'], main_summary['Z-Score Outliers'], 
              main_summary['Mod Z-Score Outliers'], main_summary['I-Forest Outliers']]
    
    bars = ax_bar.bar(methods, counts, color=[COLORS['iqr'], COLORS['zscore'], COLORS['primary'], COLORS['iforest']])
    ax_bar.set_title(f"Anomaly Detection Methodology Variance ({main_col})", weight='bold')
    ax_bar.set_ylabel("Outliers Detected")
    
    # Add labels
    for bar in bars:
        yval = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width()/2, yval + (max(counts)*0.02), f'{int(yval)}', ha='center', va='bottom', weight='bold')

    # 3. Distribution with Outlier Zones (Mid Right)
    ax_dist = fig.add_subplot(gs[1, 1])
    data = df[main_col]
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    upper_bound = q3 + (1.5 * iqr)
    
    # Exclude extreme top 0.1% just so the KDE isn't a flat line
    plot_data = data[data < np.percentile(data, 99.9)] 
    sns.kdeplot(plot_data, color=COLORS['primary'], fill=True, alpha=0.3, ax=ax_dist)
    ax_dist.axvline(upper_bound, color=COLORS['outlier'], linestyle='--', linewidth=2.5, label=f'Anomaly Threshold ({format_number(upper_bound)})')
    ax_dist.set_title(f"Density Distribution & Statistical Anomaly Threshold", weight='bold')
    ax_dist.xaxis.set_major_formatter(formatter)
    ax_dist.legend()

    # 4. Global Outlier Summary Table (Bottom)
    ax_tbl = fig.add_subplot(gs[2, :])
    ax_tbl.axis('tight')
    ax_tbl.axis('off')
    
    # Prep Table Data
    tbl_df = summary_df[['Metric', 'IQR Outliers', 'I-Forest Outliers', 'Max Outlier Value']].copy()
    tbl_df['Max Outlier Value'] = tbl_df['Max Outlier Value'].apply(lambda x: format_number(x))
    tbl_df['Outlier Impact %'] = (tbl_df['IQR Outliers'] / total_records * 100).apply(lambda x: f"{x:.2f}%")
    
    table = ax_tbl.table(cellText=tbl_df.values, colLabels=tbl_df.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 2.5)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor(COLORS['primary'])
        else:
            cell.set_facecolor('#ecf0f1' if row % 2 == 0 else 'white')

    plt.suptitle("EXECUTIVE ANOMALY & OUTLIER DETECTION DASHBOARD", fontsize=28, fontweight='bold', color=COLORS['primary'], y=0.97)
    plt.savefig(os.path.join(out_dir, "Executive_Outlier_Dashboard.png"), facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()

# =============================================================================
# EXCEL & TEXT REPORT GENERATION
# =============================================================================

def generate_excel_report(df, outlier_results, summary_df):
    """Export detailed anomaly records to Excel."""
    logging.info("Exporting Anomaly Data to Excel...")
    output_path = "sales_eda/excel/outliers/Outlier_Summary.xlsx"
    
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        # 1. Executive KPI
        summary_df.to_excel(writer, sheet_name='Executive KPI', index=False)
        
        # Save specific anomalous records for the primary net_sale_amt (if exists)
        main_col = 'net_sale_amt' if 'net_sale_amt' in outlier_results['IQR'] else list(outlier_results['IQR'].keys())[0]
        
        # 2. IQR Anomaly Set
        iqr_mask = outlier_results['IQR'][main_col]
        if iqr_mask.sum() > 0:
            df[iqr_mask].to_excel(writer, sheet_name='IQR', index=False)
        else:
            pd.DataFrame({'Message': ['No IQR outliers detected']}).to_excel(writer, sheet_name='IQR', index=False)
            
        # 3. Z-Score Anomaly Set
        z_mask = outlier_results['ZScore'][main_col]
        if z_mask.sum() > 0:
            df[z_mask].to_excel(writer, sheet_name='ZScore', index=False)
        
        # 4. Isolation Forest Anomaly Set
        if_mask = outlier_results['IsolationForest'][main_col]
        if if_mask.sum() > 0:
            df[if_mask].to_excel(writer, sheet_name='IsolationForest', index=False)

        # Formatting
        workbook = writer.book
        header_format = workbook.add_format({'bold': True, 'bg_color': COLORS['primary'], 'font_color': 'white'})
        for sheet in writer.sheets:
            writer.sheets[sheet].set_row(0, None, header_format)
            writer.sheets[sheet].set_column('A:Z', 15)

def generate_insights_report(summary_df):
    """Generate strategic business interpretations of outlier data."""
    logging.info("Generating Business Insights Text Report...")
    output_path = "reports/phase_2/Outlier_Business_Insights.txt"

    main_col = 'net_sale_amt' if 'net_sale_amt' in summary_df['Metric'].values else summary_df.iloc[0]['Metric']
    main_stats = summary_df[summary_df['Metric'] == main_col].iloc[0]
    
    total_recs = main_stats['Total Records']
    iqr_count = main_stats['IQR Outliers']
    outlier_pct = (iqr_count / total_recs) * 100
    
    high_spikes = summary_df.sort_values('Max Outlier Value', ascending=False).head(3)['Metric'].tolist()

    report_content = f"""====================================================
EXECUTIVE OUTLIER & ANOMALY INSIGHTS
====================================================

GLOBAL ANOMALY OVERVIEW ({main_col.upper()})
----------------------------------------------------
Total Records Evaluated:  {total_recs:,}
Total Outliers Detected:  {iqr_count:,}
System Outlier Ratio:     {outlier_pct:.2f}%
Highest Detected Value:   {format_number(main_stats['Max Outlier Value'])}

METHODOLOGY VARIANCE
----------------------------------------------------
IQR (Statistical Rule):          {main_stats['IQR Outliers']:,} anomalies
Z-Score (Gaussian Assumption):   {main_stats['Z-Score Outliers']:,} anomalies
Isolation Forest (ML AI):        {main_stats['I-Forest Outliers']:,} anomalies

Metrics containing the highest extreme spikes: {', '.join(high_spikes)}

====================================================
BUSINESS INTERPRETATION & RISK ANALYSIS
====================================================

1. Potential Data Quality Issues:
Z-Score identified {main_stats['Z-Score Outliers']} outliers while IQR found {main_stats['IQR Outliers']}. This wide disparity indicates the underlying data is heavily skewed and does NOT follow a normal bell curve. Extreme values ({format_number(main_stats['Max Outlier Value'])}) may be critical data entry errors (e.g., missed decimals) rather than actual sales.

2. Potential Business Events:
Legitimate outliers often represent institutional bulk purchases, seasonal holiday demand spikes, or emergency procurement during adverse weather conditions. Conversely, heavy outliers in "brkg_amt" (Breakage) indicate catastrophic logistics failures on specific shipments.

====================================================
STRATEGIC RECOMMENDATIONS
====================================================
* Model Protection: If proceeding to Phase 3 Predictive Forecasting (Machine Learning), these isolated outlier rows MUST be treated. Recommending Winsorization (capping values at the 99th percentile) to prevent these massive spikes from artificially inflating future demand forecasts.
* Commercial Audit: Export the 'IQR' sheet from the Excel output and dispatch it to the regional commercial heads to manually verify if the top 10 highest sales records were legitimate recognized revenue.
====================================================
"""
    with open(output_path, "w", encoding='utf-8') as file:
        file.write(report_content)

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    logging.info("Starting Executive Outlier Analysis Pipeline...")
    
    setup_directories()
    
    try:
        raw_data, valid_cols = load_data(INPUT_FILE)
        
        # Detect Outliers
        results_dict, results_summary = apply_outlier_methods(raw_data, valid_cols)
        
        # Outputs
        plot_anomaly_visualizations(raw_data, valid_cols, results_dict)
        plot_executive_dashboard(raw_data, results_summary, results_dict)
        generate_excel_report(raw_data, results_dict, results_summary)
        generate_insights_report(results_summary)
        
        logging.info("Outlier Analysis completed successfully.")
        
    except Exception as e:
        logging.error(f"Pipeline execution failed: {str(e)}")