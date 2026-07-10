"""
12_scatterplot_analysis.py
=============================================================================
Project: Weather Driven Pharmaceutical Sales Analytics & Forecasting
Module:  Executive Scatter Plot & Bivariate Analysis
Author:  Principal Data Scientist & Senior BI Consultant
=============================================================================
Description:
    End-to-end pipeline for performing bivariate analysis between key sales 
    and return metrics. Calculates Pearson Correlation (r), R-squared, Slope, 
    and Intercept. Generates regression scatter plots with confidence bands, 
    an executive correlation dashboard, and automated business insights.
=============================================================================
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
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
    "sales_eda/excel/scatter/",
    "sales_eda/graphs/scatter/",
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
    'regression': '#c00000',   # Dark Red
    'positive': '#00b050',     # Green
    'warning': '#ffc000',      # Golden Yellow
    'background': '#f8f9fa'    # Light Gray
}

# Define the pairs for Bivariate Analysis (X, Y)
SCATTER_PAIRS = [
    ('gross_sale_amt', 'net_sale_amt', 'Gross Sales', 'Net Sales'),
    ('gross_sale_qty', 'net_sale_amt', 'Gross Quantity', 'Net Sales'),
    ('fresh_ret_amt', 'net_sale_amt', 'Fresh Returns', 'Net Sales'),
    ('expiry_amt', 'net_sale_amt', 'Expiry Returns', 'Net Sales'),
    ('brkg_amt', 'net_sale_amt', 'Breakage Returns', 'Net Sales'),
    ('gross_sale_qty', 'net_sale_qty', 'Gross Quantity', 'Net Quantity'),
    ('gross_sale_qty', 'gross_sale_amt', 'Gross Quantity', 'Gross Sales')
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

def interpret_correlation(r_value):
    """Business interpretation of Pearson Correlation coefficient."""
    abs_r = abs(r_value)
    direction = "Positive" if r_value > 0 else "Negative"
    
    if abs_r > 0.8:
        strength = "Strong"
    elif abs_r > 0.5:
        strength = "Moderate"
    elif abs_r > 0.3:
        strength = "Weak"
    else:
        strength = "Negligible"
        direction = ""

    return f"{strength} {direction} Correlation".strip()

# =============================================================================
# DATA PROCESSING & STATISTICAL ANALYSIS
# =============================================================================

def load_data(filepath):
    """Load and validate dataset."""
    logging.info(f"Loading data from {filepath}...")
    try:
        df = pd.read_excel(filepath)
        df = df.fillna(0)  # Impute NaNs with 0 for numeric relationships
        logging.info(f"Successfully loaded {len(df)} records.")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise

def calculate_regression_stats(df):
    """Calculate deep bivariate statistics for defined pairs."""
    logging.info("Calculating Regression & Correlation Statistics...")
    
    stats_list = []
    
    for x_col, y_col, x_label, y_label in SCATTER_PAIRS:
        if x_col not in df.columns or y_col not in df.columns:
            logging.warning(f"Columns missing for pair: {x_col} vs {y_col}. Skipping.")
            continue
            
        x_data = df[x_col]
        y_data = df[y_col]
        
        # Calculate Linear Regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_data, y_data)
        r_squared = r_value ** 2
        
        stats_list.append({
            'X Variable': x_label,
            'Y Variable': y_label,
            'Pearson (r)': r_value,
            'R-Squared (R²)': r_squared,
            'Slope (m)': slope,
            'Intercept (b)': intercept,
            'P-Value': p_value,
            'Standard Error': std_err,
            'Interpretation': interpret_correlation(r_value)
        })
        
    return pd.DataFrame(stats_list)

# =============================================================================
# PRESENTATION-QUALITY VISUALIZATIONS
# =============================================================================

def plot_individual_scatters(df, stats_df):
    """Generate individual presentation-quality scatter plots with regression."""
    logging.info("Generating Individual Scatter Plots...")
    
    out_dir = "sales_eda/graphs/scatter/"
    formatter = FuncFormatter(format_number)
    
    for _, row in stats_df.iterrows():
        # Find original column names
        pair = next(p for p in SCATTER_PAIRS if p[2] == row['X Variable'] and p[3] == row['Y Variable'])
        x_col, y_col, x_label, y_label = pair
        
        fig, ax = plt.subplots(figsize=(16, 9))
        
        # Plot Scatter with Regression Line and 95% Confidence Interval using Seaborn
        sns.regplot(
            x=df[x_col], y=df[y_col], 
            ax=ax,
            scatter_kws={'alpha': 0.5, 'color': COLORS['primary'], 's': 50, 'edgecolor': 'white'},
            line_kws={'color': COLORS['regression'], 'linewidth': 2.5, 'label': 'Linear Trend'}
        )
        
        # Formatting
        ax.set_title(f"Bivariate Analysis: {y_label} vs {x_label}", pad=20)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
        
        # Annotations Box
        textstr = '\n'.join((
            f"Pearson (r): {row['Pearson (r)']:.4f}",
            f"R-Squared (R²): {row['R-Squared (R²)']:.4f}",
            f"Slope: {row['Slope (m)']:.4f}",
            f"Insight: {row['Interpretation']}"
        ))
        props = dict(boxstyle='round,pad=0.8', facecolor='white', alpha=0.9, edgecolor=COLORS['secondary'])
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=props)

        plt.savefig(os.path.join(out_dir, f"Scatter_{x_col}_vs_{y_col}.png"))
        plt.close()

def plot_executive_dashboards(df, stats_df):
    """Generate Correlation Heatmap and Executive Scatter Dashboard."""
    logging.info("Generating Executive Dashboards...")
    out_dir = "sales_eda/graphs/scatter/"
    formatter = FuncFormatter(format_number)
    
    # 1. Correlation Heatmap Dashboard
    numeric_cols = list(set([p[0] for p in SCATTER_PAIRS] + [p[1] for p in SCATTER_PAIRS]))
    valid_cols = [c for c in numeric_cols if c in df.columns]
    
    if valid_cols:
        corr_matrix = df[valid_cols].corr()
        
        fig, ax = plt.subplots(figsize=(14, 10))
        sns.heatmap(
            corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", 
            vmin=-1, vmax=1, center=0, square=True, linewidths=.5, 
            cbar_kws={"shrink": .8}, ax=ax
        )
        
        # Improve labels
        friendly_labels = [col.replace('_', ' ').title() for col in valid_cols]
        ax.set_xticklabels(friendly_labels, rotation=45, ha='right')
        ax.set_yticklabels(friendly_labels, rotation=0)
        
        ax.set_title("EXECUTIVE CORRELATION MATRIX", pad=20, fontsize=20, fontweight='bold')
        plt.savefig(os.path.join(out_dir, "Correlation_Dashboard.png"))
        plt.close()

    # 2. Combined Executive Scatter Dashboard (Top 4 Relationships)
    fig = plt.figure(figsize=(24, 16))
    fig.patch.set_facecolor(COLORS['background'])
    gs = GridSpec(2, 2, figure=fig, wspace=0.2, hspace=0.3)
    
    # Select top 4 most important pairs for the dashboard
    dashboard_pairs = SCATTER_PAIRS[:4]
    
    for idx, pair in enumerate(dashboard_pairs):
        x_col, y_col, x_label, y_label = pair
        row_idx = idx // 2
        col_idx = idx % 2
        
        ax = fig.add_subplot(gs[row_idx, col_idx])
        
        sns.regplot(
            x=df[x_col], y=df[y_col], ax=ax,
            scatter_kws={'alpha': 0.4, 'color': COLORS['primary'], 's': 30},
            line_kws={'color': COLORS['regression'], 'linewidth': 2}
        )
        
        ax.set_title(f"{y_label} vs {x_label}", fontsize=16, weight='bold')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
        
        # Mini annotation
        stat_row = stats_df[(stats_df['X Variable'] == x_label) & (stats_df['Y Variable'] == y_label)].iloc[0]
        ax.annotate(f"R²: {stat_row['R-Squared (R²)']:.2f}", xy=(0.05, 0.9), xycoords='axes fraction', 
                    fontsize=14, weight='bold', color=COLORS['regression'],
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    plt.suptitle("EXECUTIVE BIVARIATE RELATIONSHIP DASHBOARD", fontsize=28, fontweight='bold', color=COLORS['primary'], y=0.96)
    plt.savefig(os.path.join(out_dir, "Executive_Scatter_Dashboard.png"), facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()

# =============================================================================
# EXCEL & TEXT REPORT GENERATION
# =============================================================================

def generate_excel_report(stats_df):
    """Export bivariate statistics to an Excel workbook."""
    logging.info("Exporting data to Excel...")
    output_path = "sales_eda/excel/scatter/Scatter_Analysis.xlsx"
    
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        stats_df.to_excel(writer, sheet_name='Bivariate Analysis', index=False)
        
        workbook = writer.book
        header_format = workbook.add_format({'bold': True, 'bg_color': COLORS['primary'], 'font_color': 'white'})
        worksheet = writer.sheets['Bivariate Analysis']
        worksheet.set_row(0, None, header_format)
        worksheet.set_column('A:B', 20)
        worksheet.set_column('C:H', 15)
        worksheet.set_column('I:I', 30)

def generate_insights_report(stats_df):
    """Generate executive text summary mapping statistical outputs to business logic."""
    logging.info("Generating Business Insights Text Report...")
    output_path = "reports/phase_2/Scatter_Business_Insights.txt"

    strong_pos = stats_df[stats_df['Pearson (r)'] > 0.8]['X Variable'].tolist()
    strong_neg = stats_df[stats_df['Pearson (r)'] < -0.8]['X Variable'].tolist()
    weak_corrs = stats_df[stats_df['Pearson (r)'].abs() < 0.3]['X Variable'].tolist()

    report_content = f"""====================================================
EXECUTIVE SCATTER PLOT & CORRELATION INSIGHTS
====================================================

LINEAR RELATIONSHIPS
----------------------------------------------------
Strong Positive Drivers (Pearson r > 0.8):
{', '.join(strong_pos) if strong_pos else 'None identified'}

Strong Negative Impactors (Pearson r < -0.8):
{', '.join(strong_neg) if strong_neg else 'None identified'}

Weak/Non-Linear Variables (|r| < 0.3):
{', '.join(weak_corrs) if weak_corrs else 'None identified'}

====================================================
BUSINESS INTERPRETATION
====================================================

1. Sales Conversion Efficiency:
The R² score between Gross Sales and Net Sales acts as a direct proxy for "Sales Conversion Efficiency." A lower R² indicates heavy volatility introduced by discounts, returns, and breakage. 

2. Return Volatility (Fresh vs Expiry):
Analyzing the scatter plot for Returns vs Net Sales reveals whether high sales volumes naturally dictate high return volumes (linear trend) or if returns operate independently (cluster formation). If Expiry Returns show weak correlation to current Net Sales, it suggests expiry is tied to historical channel stuffing rather than current market velocity.

3. Outliers & Clusters:
Any data points residing far outside the 95% Confidence Band of the regression line represent extreme business anomalies. These are typically institutional mega-orders, catastrophic batch recalls, or data entry errors.

====================================================
STRATEGIC RECOMMENDATIONS
====================================================
* Variables demonstrating Strong Correlation can be reliably used as independent variables in Phase 3 Linear Regression forecasting.
* Variables with Weak Correlation require non-linear machine learning models (e.g., Random Forest, XGBoost) to extract predictive value, as simple trend lines fail to capture their underlying patterns.
====================================================
"""
    with open(output_path, "w", encoding='utf-8') as file:
        file.write(report_content)

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    logging.info("Starting Executive Scatter Plot Analysis Pipeline...")
    
    setup_directories()
    
    try:
        raw_data = load_data(INPUT_FILE)
        
        # Computations
        regression_stats = calculate_regression_stats(raw_data)
        
        # Generation
        plot_individual_scatters(raw_data, regression_stats)
        plot_executive_dashboards(raw_data, regression_stats)
        generate_excel_report(regression_stats)
        generate_insights_report(regression_stats)
        
        logging.info("Scatter Plot Analysis completed successfully.")
        
    except Exception as e:
        logging.error(f"Pipeline execution failed: {str(e)}")