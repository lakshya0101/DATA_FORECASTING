# import os
# import pandas as pd

# # =====================================================
# # CONFIGURATION
# # =====================================================

# INPUT_FILE = "data/Sale details.xlsx"

# OUTPUT_FOLDER = "sales_eda/phase_1_dataset_understanding/excel"

# os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# # =====================================================
# # LOAD DATA
# # =====================================================

# print("=" * 60)
# print("Loading Dataset...")
# print("=" * 60)

# df = pd.read_excel(INPUT_FILE)

# # =====================================================
# # BASIC INFORMATION
# # =====================================================

# total_rows = len(df)

# total_columns = len(df.columns)

# duplicate_rows = df.duplicated().sum()

# missing_values = df.isnull().sum().sum()

# memory_usage = round(
#     df.memory_usage(deep=True).sum() / 1024**2,
#     2
# )

# numeric_columns = len(
#     df.select_dtypes(include="number").columns
# )

# categorical_columns = len(
#     df.select_dtypes(include="object").columns
# )

# # =====================================================
# # UNIQUE COUNTS
# # =====================================================

# summary = []

# summary.append(["Total Records", total_rows])

# summary.append(["Total Columns", total_columns])

# summary.append(["Numeric Columns", numeric_columns])

# summary.append(["Categorical Columns", categorical_columns])

# summary.append(["Duplicate Records", duplicate_rows])

# summary.append(["Missing Values", missing_values])

# summary.append(["Memory Usage (MB)", memory_usage])

# if "matnr" in df.columns:
#     summary.append(
#         ["Unique Materials", df["matnr"].nunique()]
#     )

# if "item_name" in df.columns:
#     summary.append(
#         ["Unique Products", df["item_name"].nunique()]
#     )

# if "root_state_name" in df.columns:
#     summary.append(
#         ["Unique States", df["root_state_name"].nunique()]
#     )

# if "billing_month" in df.columns:
#     summary.append(
#         ["Unique Billing Months", df["billing_month"].nunique()]
#     )

# summary_df = pd.DataFrame(
#     summary,
#     columns=[
#         "Metric",
#         "Value"
#     ]
# )

# # =====================================================
# # COLUMN INFORMATION
# # =====================================================

# column_info = pd.DataFrame({

#     "Column":

#         df.columns,

#     "Data Type":

#         df.dtypes.astype(str),

#     "Non Null Values":

#         df.notnull().sum(),

#     "Missing Values":

#         df.isnull().sum(),

#     "Unique Values":

#         [
#             df[col].nunique()
#             for col in df.columns
#         ]

# })

# # =====================================================
# # SAVE
# # =====================================================

# output_file = os.path.join(

#     OUTPUT_FOLDER,

#     "Dataset_Summary.xlsx"

# )

# with pd.ExcelWriter(

#     output_file,

#     engine="openpyxl"

# ) as writer:

#     summary_df.to_excel(

#         writer,

#         sheet_name="Dataset Summary",

#         index=False

#     )

#     column_info.to_excel(

#         writer,

#         sheet_name="Column Details",

#         index=False

#     )

# print()

# print(summary_df)

# print()

# print("Saved Successfully")

# print(output_file)

# print("=" * 60)

"""
01_dataset_summary.py
=============================================================================
Project: Weather Driven Pharmaceutical Sales Analytics & Forecasting
Module:  Executive Dataset Summary & Data Quality Assessment
Author:  Principal Data Scientist
=============================================================================
Description:
    End-to-end profiling script to generate an Executive Dataset Summary.
    Computes dimensions, data types, memory usage, duplicates, and missing 
    values. Generates presentation-quality visuals, multi-sheet Excel 
    reports, and an executive text summary for BI stakeholders.
=============================================================================
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

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
    'primary': '#002060',    # Deep Corporate Blue
    'secondary': '#7f8c8d',  # Grey
    'accent': '#00b0f0',     # Light Blue
    'warning': '#c00000',    # Dark Red
    'success': '#00b050',    # Green
    'background': '#f2f2f2'  # Light Grey
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def setup_directories():
    """Create all required output directories."""
    for directory in BASE_DIRS:
        os.makedirs(directory, exist_ok=True)
    logging.info("Output directories verified and ready.")

def format_number(num):
    """Format large numbers with commas for executive readability."""
    return f"{num:,}"

def add_value_labels(ax, orient='v', is_pct=False):
    """Add precise value labels to bar charts."""
    for p in ax.patches:
        val = p.get_width() if orient == 'h' else p.get_height()
        if val == 0 or pd.isna(val): continue
        
        text = f"{val:.1f}%" if is_pct else f"{int(val):,}"
        
        if orient == 'h':
            x = p.get_width()
            y = p.get_y() + p.get_height() / 2
            ax.text(x, y, f' {text}', va='center', ha='left', fontsize=11, fontweight='bold', color=COLORS['primary'])
        else:
            x = p.get_x() + p.get_width() / 2
            y = p.get_height()
            ax.text(x, y, f'{text}\n', va='bottom', ha='center', fontsize=11, fontweight='bold', color=COLORS['primary'])

# =============================================================================
# DATA LOADING & PROFILING
# =============================================================================

def load_data(filepath):
    """Load dataset safely with error handling."""
    logging.info(f"Loading data from {filepath}...")
    try:
        df = pd.read_excel(filepath)
        logging.info(f"Successfully loaded {len(df)} records.")
        return df
    except FileNotFoundError:
        logging.error(f"File not found: {filepath}. Please ensure the data file exists.")
        raise
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise

def profile_dataset(df):
    """Generate comprehensive statistical summary of the dataset."""
    logging.info("Profiling dataset for executive KPIs...")
    
    # Identify key business columns based on standard naming conventions
    prod_col = 'item_name' if 'item_name' in df.columns else 'matnr' if 'matnr' in df.columns else None
    state_col = 'root_state_name' if 'root_state_name' in df.columns else 'state' if 'state' in df.columns else None
    month_col = 'billing_month' if 'billing_month' in df.columns else None

    # Calculate KPIs
    kpi_dict = {
        "Total Records": len(df),
        "Total Columns": len(df.columns),
        "Total Products": df[prod_col].nunique() if prod_col else "N/A",
        "Total States": df[state_col].nunique() if state_col else "N/A",
        "Total Billing Months": df[month_col].nunique() if month_col else "N/A",
        "Duplicate Records": df.duplicated().sum(),
        "Missing Values (Total)": df.isna().sum().sum(),
        "Memory Usage (MB)": np.round(df.memory_usage(deep=True).sum() / (1024**2), 2),
        "Numeric Columns": len(df.select_dtypes(include=[np.number]).columns),
        "Categorical/Text Columns": len(df.select_dtypes(exclude=[np.number]).columns)
    }

    # Column Information
    col_info = pd.DataFrame({
        'Column Name': df.columns,
        'Data Type': df.dtypes.astype(str),
        'Non-Null Count': df.notna().sum().values,
        'Missing Values': df.isna().sum().values,
        'Missing %': np.round((df.isna().sum().values / len(df)) * 100, 2),
        'Unique Values': df.nunique().values
    })

    # Data Types Summary
    dtype_summary = df.dtypes.value_counts().reset_index()
    dtype_summary.columns = ['Data Type', 'Count']
    dtype_summary['Data Type'] = dtype_summary['Data Type'].astype(str)

    # Unique Counts (for object/string columns)
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    unique_counts = pd.DataFrame({
        'Categorical Column': categorical_cols,
        'Unique Values': [df[col].nunique() for col in categorical_cols]
    }).sort_values('Unique Values', ascending=False)

    return kpi_dict, col_info, dtype_summary, unique_counts

# =============================================================================
# EXCEL & TEXT REPORT GENERATION
# =============================================================================

def generate_excel_report(kpi_dict, col_info, dtype_summary, unique_counts):
    """Export profiling results to a formatted Excel workbook."""
    logging.info("Generating Excel Dataset Summary...")
    output_path = "sales_eda/excel/business/Dataset_Summary.xlsx"
    
    kpi_df = pd.DataFrame(list(kpi_dict.items()), columns=['Executive KPI', 'Value'])
    
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        kpi_df.to_excel(writer, sheet_name='Executive KPI', index=False)
        col_info.to_excel(writer, sheet_name='Column Information', index=False)
        dtype_summary.to_excel(writer, sheet_name='Data Types', index=False)
        unique_counts.to_excel(writer, sheet_name='Unique Counts', index=False)
        
        # Format Excel widths for readability
        workbook = writer.book
        header_format = workbook.add_format({'bold': True, 'bg_color': COLORS['primary'], 'font_color': 'white'})
        
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            worksheet.set_row(0, None, header_format)
            worksheet.set_column('A:Z', 20)

def generate_insights_report(kpi_dict):
    """Generate professional text insights regarding dataset readiness."""
    logging.info("Generating Business Insights Text Report...")
    output_path = "reports/phase_1/Dataset_Executive_Insights.txt"

    dupe_status = "Excellent" if kpi_dict["Duplicate Records"] == 0 else "Requires Cleaning"
    missing_pct = np.round((kpi_dict["Missing Values (Total)"] / (kpi_dict["Total Records"] * kpi_dict["Total Columns"])) * 100, 2)
    missing_status = "High Integrity" if missing_pct < 5 else "Action Required"

    report_content = f"""====================================================
EXECUTIVE DATASET SUMMARY & ASSESSMENT
====================================================

DATASET SIZE & DIMENSIONS
----------------------------------------------------
Total Records:           {format_number(kpi_dict['Total Records'])}
Total Columns:           {format_number(kpi_dict['Total Columns'])}
Memory Usage:            {kpi_dict['Memory Usage (MB)']} MB

BUSINESS COVERAGE
----------------------------------------------------
Product Coverage:        {format_number(kpi_dict['Total Products'])} Unique Items
State Coverage:          {format_number(kpi_dict['Total States'])} Unique Geographies
Timeframe Coverage:      {kpi_dict['Total Billing Months']} Billing Months

DATA COMPLETENESS & QUALITY
----------------------------------------------------
Missing Values:          {format_number(kpi_dict['Missing Values (Total)'])} ({missing_pct}% of total cells)
Data Integrity Status:   {missing_status}
Duplicate Records:       {format_number(kpi_dict['Duplicate Records'])}
Duplication Status:      {dupe_status}

COLUMN ARCHITECTURE
----------------------------------------------------
Numeric/Quantitative:    {kpi_dict['Numeric Columns']} features
Categorical/Text:        {kpi_dict['Categorical/Text Columns']} features

====================================================
BUSINESS OBSERVATIONS & RECOMMENDATIONS
====================================================
1. Data Readiness: The dataset footprint ({kpi_dict['Memory Usage (MB)']} MB) is well within memory limits for in-memory analytics using Pandas.
2. Geographical & Product Depth: With {kpi_dict['Total States']} states and {kpi_dict['Total Products']} products, dimensional reduction or Pareto (80/20) analysis is highly recommended for phase 2.
3. Quality Action Items: Missing values and duplicates must be resolved through imputation or dropping before deploying machine learning forecasting models.
====================================================
"""
    with open(output_path, "w", encoding='utf-8') as file:
        file.write(report_content)

# =============================================================================
# PRESENTATION-QUALITY VISUALIZATIONS
# =============================================================================

def generate_graphs(dtype_summary, unique_counts, kpi_dict):
    """Generate high-resolution corporate charts and an executive dashboard."""
    logging.info("Generating Presentation Quality Graphs...")
    
    out_dir = "sales_eda/graphs/business/"

    # 1. Data Type Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(dtype_summary['Data Type'], dtype_summary['Count'], color=COLORS['accent'], edgecolor=COLORS['primary'], linewidth=1.5)
    ax.set_title("Dataset Column Architecture by Data Type")
    ax.set_ylabel("Number of Columns")
    ax.set_xlabel("Data Type")
    add_value_labels(ax, orient='v')
    plt.savefig(os.path.join(out_dir, "01_DataType_Distribution.png"))
    plt.close()

    # 2. Unique Values in Categorical Columns (Top 10)
    fig, ax = plt.subplots(figsize=(12, 7))
    top_uniques = unique_counts.head(10).sort_values('Unique Values', ascending=True)
    ax.barh(top_uniques['Categorical Column'], top_uniques['Unique Values'], color=COLORS['primary'])
    ax.set_title("Cardinality Analysis: Top Categorical Features")
    ax.set_xlabel("Count of Unique Values")
    add_value_labels(ax, orient='h')
    plt.savefig(os.path.join(out_dir, "02_Unique_Values.png"))
    plt.close()

    # 3. Executive KPI Dashboard Image
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor(COLORS['background'])
    gs = GridSpec(2, 3, figure=fig, wspace=0.1, hspace=0.1)

    cards = [
        ("TOTAL RECORDS", format_number(kpi_dict['Total Records']), COLORS['primary']),
        ("TOTAL COLUMNS", format_number(kpi_dict['Total Columns']), COLORS['primary']),
        ("BILLING MONTHS", str(kpi_dict['Total Billing Months']), COLORS['accent']),
        ("TOTAL PRODUCTS", format_number(kpi_dict['Total Products']), COLORS['success']),
        ("TOTAL STATES", format_number(kpi_dict['Total States']), COLORS['success']),
        ("DATA QUALITY ALERTS", f"{kpi_dict['Missing Values (Total)']} Nulls\n{kpi_dict['Duplicate Records']} Dupes", COLORS['warning'])
    ]

    for i, (title, val, color) in enumerate(cards):
        row = i // 3
        col = i % 3
        ax = fig.add_subplot(gs[row, col])
        ax.axis('off')
        
        # Draw card background
        rect = plt.Rectangle((0.05, 0.1), 0.9, 0.8, fill=True, color='white', 
                             edgecolor=color, linewidth=3, transform=ax.transAxes, zorder=1)
        ax.add_patch(rect)
        
        # Add Title
        ax.text(0.5, 0.75, title, fontsize=16, fontweight='bold', color=COLORS['secondary'], 
                ha='center', va='center', transform=ax.transAxes, zorder=2)
        
        # Add Value
        ax.text(0.5, 0.4, val, fontsize=36, fontweight='bold', color=color, 
                ha='center', va='center', transform=ax.transAxes, zorder=2)

    plt.suptitle("PHARMACEUTICAL SALES - DATASET EXECUTIVE SUMMARY", fontsize=28, fontweight='bold', color=COLORS['primary'], y=0.98)
    plt.savefig(os.path.join(out_dir, "03_Dataset_KPI_Dashboard.png"), facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    logging.info("Starting Executive Dataset Summary Pipeline...")
    
    # 1. Setup ecosystem
    setup_directories()
    
    try:
        # 2. Load Data
        raw_data = load_data(INPUT_FILE)
        
        # 3. Profile Dataset
        kpis, columns_info, dtypes_summary, uniques = profile_dataset(raw_data)
        
        # 4. Generate Output Artifacts
        generate_excel_report(kpis, columns_info, dtypes_summary, uniques)
        generate_insights_report(kpis)
        generate_graphs(dtypes_summary, uniques, kpis)
        
        logging.info("Phase 1 Profiling completed successfully.")
        
    except Exception as e:
        logging.error(f"Pipeline execution failed: {str(e)}")