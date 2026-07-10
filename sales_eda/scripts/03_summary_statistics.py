"""
03_summary_statistics.py
=============================================================================
Project: Weather Driven Pharmaceutical Sales Analytics & Forecasting
Module:  Executive Summary Statistics & Distribution Analysis
Author:  Principal Data Scientist & Analytics Consultant
=============================================================================
Description:
    End-to-end pipeline for calculating executive summary statistics across 
    all numerical features. Computes central tendency, dispersion, shape, 
    and missingness. Generates multi-sheet Excel reports, presentation-grade 
    visualizations, and automated statistical business insights.
=============================================================================
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats

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
    'primary': '#003366',      # Deep Navy
    'secondary': '#7f8c8d',    # Grey
    'accent': '#3498db',       # Light Blue
    'warning': '#e67e22',      # Orange
    'danger': '#c0392b',       # Red
    'success': '#27ae60',      # Green
    'background': '#f4f6f7'    # Light Grey
}

# Key Metrics to highlight if they exist
TARGET_METRICS = ['net_sale_amt', 'gross_sale_amt', 'net_sale_qty', 'total_returns_amt']

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def setup_directories():
    """Create necessary output directories safely."""
    for directory in BASE_DIRS:
        os.makedirs(directory, exist_ok=True)
    logging.info("Output directories verified and ready.")

def format_number(num):
    """Format numbers into human-readable strings (Lakhs/Crores if large)."""
    if pd.isna(num): return "0.00"
    is_neg = num < 0
    abs_num = abs(num)
    if abs_num >= 1e7: return f"{'-' if is_neg else ''}{abs_num/1e7:.2f} Cr"
    elif abs_num >= 1e5: return f"{'-' if is_neg else ''}{abs_num/1e5:.2f} L"
    elif abs_num >= 1e3: return f"{'-' if is_neg else ''}{abs_num/1e3:.2f} K"
    return f"{num:.2f}"

def add_value_labels(ax, orient='v', fmt='{:.2f}'):
    """Add precise value labels to bar charts."""
    for p in ax.patches:
        val = p.get_width() if orient == 'h' else p.get_height()
        if val == 0 or pd.isna(val): continue
        text = fmt.format(val)
        if orient == 'h':
            x, y = p.get_width(), p.get_y() + p.get_height() / 2
            ax.text(x, y, f' {text}', va='center', ha='left', fontsize=10, weight='bold', color=COLORS['primary'])
        else:
            x, y = p.get_x() + p.get_width() / 2, p.get_height()
            ax.text(x, y, f'{text}\n', va='bottom', ha='center', fontsize=10, weight='bold', color=COLORS['primary'])

# =============================================================================
# DATA PROCESSING & STATISTICAL CALCULATION
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

def calculate_summary_statistics(df):
    """Calculate deep statistical summary for all numerical columns."""
    logging.info("Calculating Executive Summary Statistics...")
    
    num_df = df.select_dtypes(include=[np.number])
    total_rows = len(df)
    
    stats_list = []
    
    for col in num_df.columns:
        s = num_df[col]
        
        # Calculate Mode safely
        mode_val = s.mode()
        mode_res = mode_val.iloc[0] if not mode_val.empty else np.nan
        
        # Calculations
        count = s.count()
        mean = s.mean()
        median = s.median()
        minimum = s.min()
        maximum = s.max()
        range_val = maximum - minimum
        variance = s.var()
        std_dev = s.std()
        cv = (std_dev / mean) if mean != 0 else np.nan
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        skewness = s.skew()
        kurt = s.kurt()
        missing_count = s.isna().sum()
        missing_pct = (missing_count / total_rows) * 100
        
        stats_list.append({
            'Feature': col,
            'Count': count,
            'Mean': mean,
            'Median': median,
            'Mode': mode_res,
            'Min': minimum,
            'Max': maximum,
            'Range': range_val,
            'Variance': variance,
            'Std Dev': std_dev,
            'CV': cv,
            '25% (Q1)': q1,
            '50% (Q2)': median,
            '75% (Q3)': q3,
            'IQR': iqr,
            'Skewness': skewness,
            'Kurtosis': kurt,
            'Missing Count': missing_count,
            'Missing %': missing_pct
        })
        
    stats_df = pd.DataFrame(stats_list)
    
    # Identify Highly Variable and Stable Features
    stats_df['Stability'] = pd.cut(
        stats_df['CV'].abs(),
        bins=[0, 0.5, 1.5, float('inf')],
        labels=['Stable (Low Variance)', 'Moderate Variance', 'Highly Variable']
    )
    
    # Identify Outlier Risk based on Skewness & Kurtosis
    stats_df['Outlier Risk'] = np.where(
        (stats_df['Skewness'].abs() > 2) | (stats_df['Kurtosis'] > 5), 
        'High Risk', 'Normal'
    )

    return stats_df

def extract_executive_kpis(df, stats_df):
    """Extract business specific KPIs from the dataset and stats."""
    
    # Attempt to locate net sales and quantity columns flexibly
    net_sales_col = 'net_sale_amt' if 'net_sale_amt' in df.columns else None
    qty_col = 'net_sale_qty' if 'net_sale_qty' in df.columns else None
    
    # Identify Most/Least variable features
    valid_cv = stats_df.dropna(subset=['CV'])
    if not valid_cv.empty:
        most_var_feat = valid_cv.loc[valid_cv['CV'].abs().idxmax(), 'Feature']
        least_var_feat = valid_cv.loc[valid_cv['CV'].abs().idxmin(), 'Feature']
    else:
        most_var_feat = least_var_feat = "N/A"

    kpis = {
        "Average Net Sales": df[net_sales_col].mean() if net_sales_col else np.nan,
        "Median Net Sales": df[net_sales_col].median() if net_sales_col else np.nan,
        "Highest Net Sales": df[net_sales_col].max() if net_sales_col else np.nan,
        "Lowest Net Sales": df[net_sales_col].min() if net_sales_col else np.nan,
        "Highest Quantity": df[qty_col].max() if qty_col else np.nan,
        "Lowest Quantity": df[qty_col].min() if qty_col else np.nan,
        "Most Variable Feature": most_var_feat,
        "Least Variable Feature": least_var_feat
    }
    return kpis

# =============================================================================
# EXCEL & TEXT REPORT GENERATION
# =============================================================================

def generate_excel_report(stats_df, kpis):
    """Generate Excel workbook with statistical profiling sheets."""
    logging.info("Generating Excel Report...")
    output_path = "sales_eda/excel/business/Summary_Statistics.xlsx"
    
    # Format KPIs
    kpi_list = [{"Executive Metric": k, "Value": v} for k, v in kpis.items()]
    kpi_df = pd.DataFrame(kpi_list)
    
    desc_cols = ['Feature', 'Count', 'Mean', 'Median', 'Mode', 'Min', 'Max', 'Missing %']
    dist_cols = ['Feature', 'CV', 'Skewness', 'Kurtosis', 'IQR', 'Stability', 'Outlier Risk']
    
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        kpi_df.to_excel(writer, sheet_name='Executive KPI', index=False)
        stats_df[desc_cols].to_excel(writer, sheet_name='Descriptive Statistics', index=False)
        stats_df[dist_cols].to_excel(writer, sheet_name='Distribution Summary', index=False)
        stats_df.to_excel(writer, sheet_name='Statistical Summary', index=False)
        
        workbook = writer.book
        header_format = workbook.add_format({'bold': True, 'bg_color': COLORS['primary'], 'font_color': 'white'})
        for sheet in writer.sheets:
            writer.sheets[sheet].set_row(0, None, header_format)
            writer.sheets[sheet].set_column('A:S', 18)

def generate_insights_report(stats_df, kpis):
    """Generate professional text insights regarding data distribution."""
    logging.info("Generating Business Insights Text Report...")
    output_path = "reports/phase_1/Summary_Statistics_Business_Insights.txt"

    high_var = stats_df[stats_df['Stability'] == 'Highly Variable']['Feature'].tolist()
    stable = stats_df[stats_df['Stability'] == 'Stable (Low Variance)']['Feature'].tolist()
    high_skew = stats_df[stats_df['Skewness'].abs() > 2]['Feature'].tolist()

    report_content = f"""====================================================
EXECUTIVE STATISTICAL SUMMARY & BUSINESS INSIGHTS
====================================================

CORE BUSINESS STATISTICS
----------------------------------------------------
Average Net Sales:        {format_number(kpis['Average Net Sales'])}
Median Net Sales:         {format_number(kpis['Median Net Sales'])}
Highest Net Sales:        {format_number(kpis['Highest Net Sales'])}
Highest Quantity Sold:    {format_number(kpis['Highest Quantity'])}

DATA STABILITY (Coefficient of Variation)
----------------------------------------------------
Most Variable Feature:    {kpis['Most Variable Feature']}
Least Variable Feature:   {kpis['Least Variable Feature']}
Highly Variable Features: {', '.join(high_var) if high_var else 'None'}
Highly Stable Features:   {', '.join(stable) if stable else 'None'}

DISTRIBUTION CHARACTERISTICS & OUTLIER RISK
----------------------------------------------------
Highly Skewed Features:   {', '.join(high_skew) if high_skew else 'None'}
(Features with Skewness > 2 or < -2 indicate heavy tails and potential outliers)

====================================================
BUSINESS INTERPRETATION & RECOMMENDATIONS
====================================================

Statistical Observations:
1. Mean vs Median Disparity: If the Average Net Sales is significantly higher than the Median, the sales distribution is positively skewed, meaning a small percentage of transactions or products drive the majority of the revenue.
2. Variance Interpretation: Features flagged as 'Highly Variable' (CV > 1.5) lack predictability. If sales volume is highly variable, standard mean-based forecasting will fail.

Risk Analysis:
Features identified with 'High Outlier Risk' due to kurtosis (>5) contain extreme spikes. These are likely anomalous bulk orders, institutional sales, or data entry errors. 

Recommendations:
1. Deploy robust scaler normalization on 'Highly Variable' features before applying machine learning models.
2. Isolate extreme values in the highly skewed features and conduct a root-cause analysis with the commercial team to verify if they are genuine business spikes (e.g., season-end dumps) or anomalies.
====================================================
"""
    with open(output_path, "w", encoding='utf-8') as file:
        file.write(report_content)

# =============================================================================
# PRESENTATION-QUALITY VISUALIZATIONS
# =============================================================================

def generate_graphs(df, stats_df, kpis):
    """Generate high-resolution corporate statistical charts."""
    logging.info("Generating Presentation Quality Graphs...")
    
    out_dir = "sales_eda/graphs/business/"
    
    # Ensure we only plot features with meaningful positive means for log comparison
    plot_df = stats_df.dropna(subset=['Mean', 'Median']).copy()
    plot_df = plot_df[(plot_df['Mean'] > 1) & (plot_df['Feature'].str.contains('amt|qty|sale|ret', case=False, na=False))].head(8)

    # 1. Mean vs Median
    fig, ax = plt.subplots()
    x = np.arange(len(plot_df))
    width = 0.35
    ax.bar(x - width/2, plot_df['Mean'], width, label='Mean', color=COLORS['primary'])
    ax.bar(x + width/2, plot_df['Median'], width, label='Median', color=COLORS['accent'])
    ax.set_title("Central Tendency: Mean vs Median Disparity")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df['Feature'], rotation=45, ha='right')
    ax.legend()
    # Log scale if max is very high to ensure visibility
    if plot_df['Mean'].max() > 1000 * plot_df['Median'].min():
        ax.set_yscale('log')
        ax.set_ylabel("Value (Log Scale)")
    else:
        ax.set_ylabel("Value")
    plt.savefig(os.path.join(out_dir, "01_Mean_vs_Median.png"))
    plt.close()

    # 2. Distribution Summary (Boxplots for key metrics)
    fig, ax = plt.subplots()
    key_cols = [c for c in TARGET_METRICS if c in df.columns]
    if key_cols:
        # Normalize data (Z-score) for boxplot comparison on same axis
        normalized_df = (df[key_cols] - df[key_cols].mean()) / df[key_cols].std()
        box = ax.boxplot([normalized_df[col].dropna() for col in key_cols], patch_artist=True, showfliers=False)
        for patch in box['boxes']: patch.set_facecolor(COLORS['accent'])
        for median in box['medians']: median.set_color(COLORS['danger'])
        ax.set_xticklabels(key_cols, rotation=15)
        ax.set_title("Standardized Distribution of Key Business Metrics (Excl. Outliers)")
        ax.set_ylabel("Z-Score (Standard Deviations)")
        plt.savefig(os.path.join(out_dir, "02_Distribution_Summary.png"))
    plt.close()

    # 3. Variance Comparison (Coefficient of Variation)
    fig, ax = plt.subplots()
    cv_df = stats_df.dropna(subset=['CV']).sort_values('CV', ascending=False).head(15)
    colors = [COLORS['danger'] if cv > 1.5 else COLORS['accent'] for cv in cv_df['CV']]
    ax.barh(cv_df['Feature'][::-1], cv_df['CV'][::-1], color=colors[::-1])
    ax.set_title("Volatility Analysis: Coefficient of Variation (CV)")
    ax.set_xlabel("CV (Standard Deviation / Mean)")
    ax.axvline(1.0, color='black', linestyle='--', label='High Variance Threshold (1.0)')
    ax.legend()
    add_value_labels(ax, orient='h', fmt='{:.2f}')
    plt.savefig(os.path.join(out_dir, "03_Variance_Comparison.png"))
    plt.close()

    # 4. Skewness Analysis
    fig, ax = plt.subplots()
    skew_df = stats_df.dropna(subset=['Skewness']).sort_values('Skewness', ascending=False).head(15)
    colors = [COLORS['primary'] if s > 0 else COLORS['warning'] for s in skew_df['Skewness']]
    ax.bar(skew_df['Feature'], skew_df['Skewness'], color=colors)
    ax.set_title("Distribution Shape: Feature Skewness")
    ax.set_ylabel("Skewness Score")
    ax.tick_params(axis='x', rotation=45)
    ax.axhline(0, color='black', linewidth=1)
    ax.axhline(2, color='red', linestyle='--', alpha=0.5, label='High Positive Skew')
    ax.legend()
    plt.savefig(os.path.join(out_dir, "04_Skewness_Analysis.png"))
    plt.close()

    # 5. Executive Statistics Dashboard
    fig = plt.figure(figsize=(22, 14))
    fig.patch.set_facecolor(COLORS['background'])
    gs = GridSpec(3, 3, figure=fig, height_ratios=[0.5, 1.2, 1.2], wspace=0.3, hspace=0.5)

    # Top KPI Cards
    kpi_cards = [
        ("AVERAGE NET SALES", format_number(kpis['Average Net Sales']), COLORS['primary']),
        ("MEDIAN NET SALES", format_number(kpis['Median Net Sales']), COLORS['accent']),
        ("MOST VARIABLE FEATURE", str(kpis['Most Variable Feature']), COLORS['danger'])
    ]
    
    for i, (title, val, color) in enumerate(kpi_cards):
        ax_kpi = fig.add_subplot(gs[0, i])
        ax_kpi.axis('off')
        rect = plt.Rectangle((0, 0.1), 1, 0.8, fill=True, color='white', 
                             edgecolor=color, linewidth=2, transform=ax_kpi.transAxes)
        ax_kpi.add_patch(rect)
        ax_kpi.text(0.5, 0.7, title, fontsize=14, fontweight='bold', color=COLORS['secondary'], ha='center')
        ax_kpi.text(0.5, 0.35, val, fontsize=30, fontweight='bold', color=color, ha='center')

    # Mean vs Median Chart (Mid Left)
    ax_mm = fig.add_subplot(gs[1, 0:2])
    x = np.arange(len(plot_df))
    ax_mm.bar(x - 0.2, plot_df['Mean'], 0.4, label='Mean', color=COLORS['primary'])
    ax_mm.bar(x + 0.2, plot_df['Median'], 0.4, label='Median', color=COLORS['accent'])
    ax_mm.set_title("Mean vs Median (Outlier Indicator)", weight='bold')
    ax_mm.set_xticks(x)
    ax_mm.set_xticklabels(plot_df['Feature'], rotation=25, ha='right')
    ax_mm.legend()

    # Skewness (Mid Right)
    ax_sk = fig.add_subplot(gs[1, 2])
    top_skew = skew_df.head(6)
    ax_sk.barh(top_skew['Feature'][::-1], top_skew['Skewness'][::-1], color=COLORS['warning'])
    ax_sk.set_title("Highest Skewness (Risk)", weight='bold')

    # Volatility / CV (Bottom Left)
    ax_cv = fig.add_subplot(gs[2, 0:2])
    top_cv = cv_df.head(8)
    ax_cv.bar(top_cv['Feature'], top_cv['CV'], color=COLORS['danger'])
    ax_cv.axhline(1.0, color='black', linestyle='--')
    ax_cv.set_title("Highest Volatility (Coefficient of Variation)", weight='bold')
    ax_cv.tick_params(axis='x', rotation=25)

    # Statistical Table (Bottom Right)
    ax_tbl = fig.add_subplot(gs[2, 2])
    ax_tbl.axis('tight')
    ax_tbl.axis('off')
    tbl_df = stats_df.dropna(subset=['Mean']).sort_values('Mean', ascending=False).head(5)[['Feature', 'Mean', 'Std Dev', 'Missing %']]
    # Format for display
    tbl_df['Mean'] = tbl_df['Mean'].apply(lambda x: f"{x:,.0f}")
    tbl_df['Std Dev'] = tbl_df['Std Dev'].apply(lambda x: f"{x:,.0f}")
    tbl_df['Missing %'] = tbl_df['Missing %'].apply(lambda x: f"{x:.1f}%")
    
    table = ax_tbl.table(cellText=tbl_df.values, colLabels=tbl_df.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor(COLORS['primary'])

    plt.suptitle("EXECUTIVE STATISTICAL OVERVIEW & DISTRIBUTION DASHBOARD", fontsize=26, fontweight='bold', color=COLORS['primary'], y=0.97)
    plt.savefig(os.path.join(out_dir, "05_Statistics_Dashboard.png"), facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    logging.info("Starting Summary Statistics Pipeline...")
    
    setup_directories()
    
    try:
        raw_data = load_data(INPUT_FILE)
        
        # Computations
        statistics_df = calculate_summary_statistics(raw_data)
        executive_kpis = extract_executive_kpis(raw_data, statistics_df)
        
        # Generation
        generate_excel_report(statistics_df, executive_kpis)
        generate_insights_report(statistics_df, executive_kpis)
        generate_graphs(raw_data, statistics_df, executive_kpis)
        
        logging.info("Statistical Analysis completed successfully.")
        
    except Exception as e:
        logging.error(f"Pipeline execution failed: {str(e)}")