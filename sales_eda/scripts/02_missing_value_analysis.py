"""
02_missing_value_analysis.py
=============================================================================
Project: Weather Driven Pharmaceutical Sales Analytics & Forecasting
Module:  Missing Value Analysis & Data Quality Assessment
Author:  Principal Data Scientist & BI Consultant
=============================================================================
Description:
    End-to-end pipeline for Executive Missing Value Analysis. Calculates 
    missing distributions, overall data completeness, and a proprietary Data 
    Quality Score. Generates presentation-grade visualizations, Excel reports,
    and automated text insights for C-suite stakeholders.
=============================================================================
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import PercentFormatter

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
    "reports/phase_1/"
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
    'secondary': '#95a5a6',    # Grey
    'danger': '#e74c3c',       # Red (Missing Data)
    'success': '#27ae60',      # Green (Complete Data)
    'warning': '#f39c12',      # Orange
    'background': '#f8f9fa'    # Light Background
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def setup_directories():
    """Create all required output directories safely."""
    for directory in BASE_DIRS:
        os.makedirs(directory, exist_ok=True)
    logging.info("Output directories verified and ready.")

def format_number(num):
    """Format large numbers with commas for executive readability."""
    return f"{int(num):,}"

def add_value_labels(ax, orient='v', is_pct=False):
    """Add precise value labels to bar charts."""
    for p in ax.patches:
        val = p.get_width() if orient == 'h' else p.get_height()
        if val == 0 or pd.isna(val): continue
        
        text = f"{val:.2f}%" if is_pct else f"{int(val):,}"
        
        if orient == 'h':
            x = p.get_width()
            y = p.get_y() + p.get_height() / 2
            ax.text(x, y, f' {text}', va='center', ha='left', fontsize=11, fontweight='bold', color=COLORS['primary'])
        else:
            x = p.get_x() + p.get_width() / 2
            y = p.get_height()
            ax.text(x, y, f'{text}\n', va='bottom', ha='center', fontsize=11, fontweight='bold', color=COLORS['primary'])

# =============================================================================
# DATA PROCESSING & METRICS GENERATION
# =============================================================================

def load_data(filepath):
    """Load dataset with error handling."""
    logging.info(f"Loading data from {filepath}...")
    try:
        df = pd.read_excel(filepath)
        logging.info(f"Successfully loaded {len(df)} records.")
        return df
    except FileNotFoundError:
        logging.error(f"File not found: {filepath}. Ensure data exists.")
        raise
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise

def analyze_missing_values(df):
    """Calculate comprehensive missing value metrics and DQ Score."""
    logging.info("Performing Missing Value Analysis...")
    
    total_rows = len(df)
    total_cols = len(df.columns)
    total_cells = total_rows * total_cols
    
    # Missing counts per column
    missing_counts = df.isna().sum()
    missing_df = pd.DataFrame({
        'Column': missing_counts.index,
        'Missing Count': missing_counts.values
    })
    
    missing_df['Missing Percentage'] = (missing_df['Missing Count'] / total_rows) * 100
    missing_df['Data Completeness'] = 100 - missing_df['Missing Percentage']
    
    # Sort descending by missing count
    missing_df = missing_df.sort_values(by='Missing Count', ascending=False).reset_index(drop=True)
    
    # Extract KPIs
    total_missing = int(missing_df['Missing Count'].sum())
    cols_with_missing = missing_df[missing_df['Missing Count'] > 0]
    num_cols_with_missing = len(cols_with_missing)
    
    overall_completeness_pct = ((total_cells - total_missing) / total_cells) * 100
    complete_records = len(df.dropna())
    
    most_incomplete_col = cols_with_missing.iloc[0]['Column'] if num_cols_with_missing > 0 else "None"
    most_complete_col = missing_df.iloc[-1]['Column']
    
    # Data Quality Score (Out of 100)
    # Penalty applies if missing rows > 0. We square the missing ratio to penalize heavily missing data.
    missing_ratio = total_missing / total_cells
    dq_score = round(100 * (1 - (missing_ratio ** 0.8)), 2)

    kpi_dict = {
        "Total Rows": total_rows,
        "Total Columns": total_cols,
        "Total Missing Values": total_missing,
        "Complete Records": complete_records,
        "Columns with Missing Data": num_cols_with_missing,
        "Most Incomplete Column": most_incomplete_col,
        "Most Complete Column": most_complete_col,
        "Overall Data Completeness %": round(overall_completeness_pct, 2),
        "Data Quality Score": dq_score
    }
    
    return missing_df, kpi_dict

# =============================================================================
# EXCEL & TEXT REPORT GENERATION
# =============================================================================

def generate_excel_report(missing_df, kpi_dict):
    """Generate Excel workbook with multiple missing value analytical sheets."""
    logging.info("Generating Excel Report...")
    output_path = "sales_eda/excel/business/Missing_Value_Analysis.xlsx"
    
    kpi_df = pd.DataFrame(list(kpi_dict.items()), columns=['Executive KPI', 'Value'])
    
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        kpi_df.to_excel(writer, sheet_name='Executive KPI', index=False)
        missing_df[['Column', 'Missing Count', 'Missing Percentage']].to_excel(writer, sheet_name='Column Summary', index=False)
        
        # Missing Percentage sorted
        miss_pct_df = missing_df[missing_df['Missing Percentage'] > 0].copy()
        if miss_pct_df.empty:
            pd.DataFrame({"Message": ["No missing values found"]}).to_excel(writer, sheet_name='Missing Percentage', index=False)
        else:
            miss_pct_df[['Column', 'Missing Percentage']].to_excel(writer, sheet_name='Missing Percentage', index=False)
        
        missing_df[['Column', 'Data Completeness']].to_excel(writer, sheet_name='Data Completeness', index=False)
        
        # Formatting
        workbook = writer.book
        header_format = workbook.add_format({'bold': True, 'bg_color': COLORS['primary'], 'font_color': 'white'})
        for sheet in writer.sheets:
            writer.sheets[sheet].set_row(0, None, header_format)
            writer.sheets[sheet].set_column('A:C', 25)

def generate_insights_report(kpi_dict, missing_df):
    """Generate professional text insights regarding missing data risks."""
    logging.info("Generating Business Insights Text Report...")
    output_path = "reports/phase_1/Missing_Value_Business_Insights.txt"

    dq_score = kpi_dict['Data Quality Score']
    missing_cols_count = kpi_dict['Columns with Missing Data']
    
    if dq_score >= 98:
        risk_level = "Low Risk"
        recommendation = "Data is highly complete. Proceed with analytical modeling without heavy imputation."
    elif dq_score >= 85:
        risk_level = "Moderate Risk"
        recommendation = "Impute missing values using mean/median for numericals and mode for categoricals before feeding into models."
    else:
        risk_level = "High Risk"
        recommendation = "Severe data leakage identified. Consult data engineering team to fix pipeline extraction logic before modeling."

    report_content = f"""====================================================
EXECUTIVE MISSING VALUE ANALYSIS & DATA QUALITY
====================================================

OVERVIEW
----------------------------------------------------
Data Quality Score:       {dq_score} / 100
Overall Completeness:     {kpi_dict['Overall Data Completeness %']}%
Total Missing Values:     {format_number(kpi_dict['Total Missing Values'])}

GRANULAR METRICS
----------------------------------------------------
Total Records:            {format_number(kpi_dict['Total Rows'])}
Complete Records:         {format_number(kpi_dict['Complete Records'])} (No missing fields)
Columns with Missing Data:{missing_cols_count} out of {kpi_dict['Total Columns']}
Most Incomplete Column:   {kpi_dict['Most Incomplete Column']}
Most Complete Column:     {kpi_dict['Most Complete Column']}

====================================================
BUSINESS IMPACT & RISK ANALYSIS
====================================================

Risk Level: {risk_level}

Business Impact:
Missing values in core categorical or continuous metrics degrade the integrity of historical sales reporting and inject bias into future predictive forecasting models. A DQ Score of {dq_score} indicates the baseline reliability of the provided dataset.

Risk Analysis:
Columns operating with significant missing data cannot be relied upon for segmentation or correlation analysis. If "{kpi_dict['Most Incomplete Column']}" is a primary business driver, analysis leveraging this feature will be compromised.

Recommendations:
1. {recommendation}
2. Flag records with missing critical fields in the final dashboard to warn end-users of potential reporting discrepancies.
3. Establish a data governance SLA to reduce missing inputs at the source system level.
====================================================
"""
    with open(output_path, "w", encoding='utf-8') as file:
        file.write(report_content)

# =============================================================================
# PRESENTATION-QUALITY VISUALIZATIONS
# =============================================================================

def generate_graphs(missing_df, kpi_dict):
    """Generate high-resolution corporate charts and an executive dashboard."""
    logging.info("Generating Presentation Quality Graphs...")
    
    out_dir = "sales_eda/graphs/business/"
    
    # Filter only columns with missing data for bar charts, or top 10 if none
    plot_df = missing_df[missing_df['Missing Count'] > 0].head(15)
    has_missing = not plot_df.empty

    # 1. Missing Value Bar Chart
    fig, ax = plt.subplots(figsize=(12, 7))
    if has_missing:
        ax.bar(plot_df['Column'], plot_df['Missing Count'], color=COLORS['danger'])
        ax.set_title("Missing Value Counts by Column")
        ax.set_ylabel("Count of Missing Values")
        ax.tick_params(axis='x', rotation=45)
        add_value_labels(ax, orient='v')
    else:
        ax.text(0.5, 0.5, "Dataset is 100% Complete\nNo Missing Values Detected", 
                ha='center', va='center', fontsize=20, color=COLORS['success'], weight='bold')
        ax.axis('off')
    plt.savefig(os.path.join(out_dir, "01_Missing_Value_Bar.png"))
    plt.close()

    # 2. Data Completeness Donut Chart
    fig, ax = plt.subplots(figsize=(8, 8))
    completeness_pct = kpi_dict['Overall Data Completeness %']
    missing_pct = 100 - completeness_pct
    
    wedges, texts, autotexts = ax.pie(
        [completeness_pct, missing_pct], 
        labels=['Complete Data', 'Missing Data'], 
        colors=[COLORS['success'], COLORS['danger']], 
        autopct='%1.2f%%', startangle=90, 
        textprops=dict(color="white", weight="bold", fontsize=12)
    )
    # Draw circle for Donut effect
    centre_circle = plt.Circle((0,0), 0.70, fc='white')
    fig.gca().add_artist(centre_circle)
    
    # Add label to texts outside
    for text in texts:
        text.set_color(COLORS['primary'])
        text.set_fontsize(14)
        
    ax.set_title("Overall Data Completeness Ratio")
    plt.savefig(os.path.join(out_dir, "02_Data_Completeness.png"))
    plt.close()

    # 3. Missing Percentage Horizontal Bar
    fig, ax = plt.subplots(figsize=(12, 7))
    if has_missing:
        ax.barh(plot_df['Column'][::-1], plot_df['Missing Percentage'][::-1], color=COLORS['warning'])
        ax.set_title("Percentage of Missing Data per Column")
        ax.set_xlabel("Missing Percentage (%)")
        ax.xaxis.set_major_formatter(PercentFormatter())
        add_value_labels(ax, orient='h', is_pct=True)
    else:
        ax.text(0.5, 0.5, "100% Complete\nZero Missing Percentage", 
                ha='center', va='center', fontsize=20, color=COLORS['success'], weight='bold')
        ax.axis('off')
    plt.savefig(os.path.join(out_dir, "03_Missing_Percentage.png"))
    plt.close()

    # 4. Executive Missing Value Dashboard
    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor(COLORS['background'])
    gs = GridSpec(3, 3, figure=fig, height_ratios=[0.5, 1.5, 1], wspace=0.3, hspace=0.4)

    # Top KPI Cards
    kpis = [
        ("DATA QUALITY SCORE", f"{kpi_dict['Data Quality Score']}/100", COLORS['primary']),
        ("OVERALL COMPLETENESS", f"{kpi_dict['Overall Data Completeness %']}%", COLORS['success']),
        ("TOTAL MISSING CELLS", format_number(kpi_dict['Total Missing Values']), COLORS['danger'])
    ]
    
    for i, (title, val, color) in enumerate(kpis):
        ax_kpi = fig.add_subplot(gs[0, i])
        ax_kpi.axis('off')
        rect = plt.Rectangle((0, 0.1), 1, 0.8, fill=True, color='white', 
                             edgecolor=color, linewidth=2, transform=ax_kpi.transAxes)
        ax_kpi.add_patch(rect)
        ax_kpi.text(0.5, 0.7, title, fontsize=14, fontweight='bold', color=COLORS['secondary'], ha='center')
        ax_kpi.text(0.5, 0.35, val, fontsize=32, fontweight='bold', color=color, ha='center')

    # Left: Missing Percentage Bar
    ax_bar = fig.add_subplot(gs[1, 0:2])
    if has_missing:
        ax_bar.bar(plot_df['Column'], plot_df['Missing Percentage'], color=COLORS['warning'])
        ax_bar.set_title("Missing Value % by Feature", fontsize=16, weight='bold')
        ax_bar.set_ylabel("Missing (%)")
        ax_bar.yaxis.set_major_formatter(PercentFormatter())
        ax_bar.tick_params(axis='x', rotation=30)
    else:
        ax_bar.text(0.5, 0.5, "No Missing Data to Display", ha='center', va='center', fontsize=18)
        ax_bar.axis('off')

    # Right: Completeness Donut
    ax_pie = fig.add_subplot(gs[1, 2])
    ax_pie.pie([completeness_pct, missing_pct], labels=['Complete', 'Missing'], 
               colors=[COLORS['success'], COLORS['danger']], autopct='%1.1f%%', 
               startangle=90, textprops={'weight': 'bold', 'color': COLORS['primary']})
    centre_circle_dash = plt.Circle((0,0), 0.65, fc='white')
    ax_pie.add_artist(centre_circle_dash)
    ax_pie.set_title("Completeness Ratio", fontsize=16, weight='bold')

    # Bottom: Business Insights Text Block
    ax_txt = fig.add_subplot(gs[2, :])
    ax_txt.axis('off')
    
    insight_text = f"EXECUTIVE BUSINESS INSIGHTS\n" \
                   f"----------------------------------------------------------------------\n" \
                   f"• Data Integrity Profile: With an overall completeness of {kpi_dict['Overall Data Completeness %']}%, the dataset achieved a DQ Score of {kpi_dict['Data Quality Score']}/100.\n" \
                   f"• Structural Risk: {kpi_dict['Columns with Missing Data']} out of {kpi_dict['Total Columns']} columns contain missing data architectures.\n" \
                   f"• Critical Finding: The most incomplete feature is '{kpi_dict['Most Incomplete Column']}'. If this feature governs pricing, volume, or region mappings, imputation strategies must be deployed.\n" \
                   f"• Actionable Recommendation: Establish a robust mean/mode imputation pipeline for analytics, but flag missing entries in downstream BI reports to prevent misinformed executive decision-making."
    
    ax_txt.text(0.02, 0.5, insight_text, fontsize=14, color=COLORS['primary'], 
                ha='left', va='center', wrap=True, family='monospace', 
                bbox=dict(facecolor='white', edgecolor=COLORS['secondary'], boxstyle='round,pad=1', alpha=0.9))

    plt.suptitle("EXECUTIVE DASHBOARD: MISSING VALUE & DATA QUALITY ASSESSMENT", fontsize=24, fontweight='bold', color=COLORS['primary'], y=0.98)
    plt.savefig(os.path.join(out_dir, "04_Missing_Dashboard.png"), facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    logging.info("Starting Executive Missing Value Analysis Pipeline...")
    
    setup_directories()
    
    try:
        raw_data = load_data(INPUT_FILE)
        
        missing_analysis_df, kpis = analyze_missing_values(raw_data)
        
        generate_excel_report(missing_analysis_df, kpis)
        generate_insights_report(kpis, missing_analysis_df)
        generate_graphs(missing_analysis_df, kpis)
        
        logging.info("Missing Value Analysis completed successfully.")
        
    except Exception as e:
        logging.error(f"Pipeline execution failed: {str(e)}")