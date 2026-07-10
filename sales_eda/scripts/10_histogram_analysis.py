"""
10_histogram_analysis.py
=============================================================================
Project: Weather Driven Pharmaceutical Sales Analytics & Forecasting
Module:  Executive Distribution & Histogram Analysis
Author:  Principal Data Scientist & Analytics Consultant
=============================================================================
Description:
    End-to-end pipeline for performing deep distribution analysis on numeric 
    sales and return metrics. Computes measures of central tendency, dispersion, 
    and shape (Skewness/Kurtosis). Generates presentation-grade histograms 
    with KDE and Normal Distribution overlays, along with automated business 
    insights and Excel summaries.
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
    "sales_eda/excel/distribution/",
    "sales_eda/graphs/distribution/",
    "reports/phase_2/"
]

# Numeric Columns for Analysis
NUMERIC_COLUMNS = [
    "gross_sale_amt", "gross_sale_qty", 
    "fresh_ret_amt", "fresh_ret_qty", 
    "expiry_amt", "expiry_qty", 
    "brkg_amt", "brkg_qty", 
    "net_sale_amt", "net_sale_qty"
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
    'mean': '#c00000',         # Dark Red
    'median': '#00b050',       # Green
    'normal': '#ffc000',       # Golden Yellow
    'background': '#f8f9fa'    # Light Gray
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
    """Format large numbers for executive readability."""
    if pd.isna(num): return "0.00"
    is_neg = num < 0
    abs_num = abs(num)
    if abs_num >= 1e7: return f"{'-' if is_neg else ''}{abs_num/1e7:.2f} Cr"
    elif abs_num >= 1e5: return f"{'-' if is_neg else ''}{abs_num/1e5:.2f} L"
    elif abs_num >= 1e3: return f"{'-' if is_neg else ''}{abs_num/1e3:.2f} K"
    return f"{num:.2f}"

def interpret_skewness(skew_val):
    """Business interpretation of statistical skewness."""
    if skew_val > 1:
        return "Highly Right Skewed (Concentration of small values, long tail of high outliers)"
    elif 0.5 < skew_val <= 1:
        return "Moderately Right Skewed"
    elif -0.5 <= skew_val <= 0.5:
        return "Approximately Symmetrical (Normal/Bell-Curve distribution)"
    elif -1 <= skew_val < -0.5:
        return "Moderately Left Skewed"
    else:
        return "Highly Left Skewed (Concentration of high values, long tail of low outliers)"

def interpret_kurtosis(kurt_val):
    """Business interpretation of statistical kurtosis."""
    if kurt_val > 3:
        return "Heavy Tails (Leptokurtic) - High frequency of extreme outliers/spikes"
    elif -3 <= kurt_val <= 3:
        return "Normal Tails (Mesokurtic) - Standard outlier frequency"
    else:
        return "Light Tails (Platykurtic) - Highly uniform data, lack of outliers"

# =============================================================================
# DATA PROCESSING & STATISTICAL ANALYSIS
# =============================================================================

def load_data(filepath):
    """Load and validate dataset."""
    logging.info(f"Loading data from {filepath}...")
    try:
        df = pd.read_excel(filepath)
        missing_cols = [col for col in NUMERIC_COLUMNS if col not in df.columns]
        if missing_cols:
            logging.warning(f"Missing columns in dataset: {missing_cols}")
        
        # Keep only existing requested columns
        valid_cols = [col for col in NUMERIC_COLUMNS if col in df.columns]
        df = df[valid_cols].copy()
        
        # Fill missing values with 0 for numeric distribution logic
        df = df.fillna(0)
        logging.info(f"Successfully loaded {len(df)} records.")
        return df, valid_cols
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise

def calculate_distribution_stats(df, columns):
    """Calculate deep statistical summary for the distribution analysis."""
    logging.info("Calculating Distribution Statistics...")
    
    stats_list = []
    
    for col in columns:
        data = df[col]
        
        # Central Tendency
        mean = data.mean()
        median = data.median()
        mode_val = data.mode()
        mode = mode_val.iloc[0] if not mode_val.empty else np.nan
        
        # Dispersion
        std_dev = data.std()
        variance = data.var()
        min_val = data.min()
        max_val = data.max()
        
        # Quartiles
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        
        # Shape
        skewness = data.skew()
        kurt = data.kurt()
        
        stats_list.append({
            'Metric': col,
            'Count': data.count(),
            'Mean': mean,
            'Median': median,
            'Mode': mode,
            'Std Dev': std_dev,
            'Variance': variance,
            'Min': min_val,
            '25% (Q1)': q1,
            '50% (Median)': median,
            '75% (Q3)': q3,
            'Max': max_val,
            'Skewness': skewness,
            'Kurtosis': kurt,
            'Distribution Shape': interpret_skewness(skewness),
            'Tail Behavior': interpret_kurtosis(kurt)
        })
        
    return pd.DataFrame(stats_list)

# =============================================================================
# PRESENTATION-QUALITY VISUALIZATIONS
# =============================================================================

def plot_individual_histograms(df, columns, stats_df):
    """Generate individual presentation-quality histograms with overlays."""
    logging.info("Generating Individual Histograms & Distribution Overlays...")
    
    out_dir = "sales_eda/graphs/distribution/"
    formatter = FuncFormatter(lambda x, pos: format_number(x))
    
    for col in columns:
        data = df[col]
        col_stats = stats_df[stats_df['Metric'] == col].iloc[0]
        
        # Remove extreme 1% outliers ONLY for visual clarity in the histogram
        # (Stats are still calculated on 100% of the data)
        p1, p99 = data.quantile(0.01), data.quantile(0.99)
        plot_data = data[(data >= p1) & (data <= p99)]
        
        if plot_data.empty or plot_data.nunique() <= 1:
            logging.warning(f"Skipping plot for {col} due to lack of variance.")
            continue

        fig, ax = plt.subplots(figsize=(16, 9))
        
        # Histogram & KDE
        sns.histplot(
            plot_data, 
            kde=True, 
            stat='density', 
            color=COLORS['primary'], 
            edgecolor='white',
            linewidth=1.2,
            alpha=0.7,
            ax=ax,
            label='Data Distribution (KDE)'
        )
        
        # Normal Distribution Overlay
        mu, std = col_stats['Mean'], col_stats['Std Dev']
        if std > 0:
            xmin, xmax = ax.get_xlim()
            x = np.linspace(xmin, xmax, 100)
            p = stats.norm.pdf(x, mu, std)
            ax.plot(x, p, color=COLORS['normal'], linewidth=3, linestyle='-', label='Theoretical Normal Dist.')

        # Mean and Median Lines
        ax.axvline(col_stats['Mean'], color=COLORS['mean'], linestyle='--', linewidth=2.5, label=f"Mean: {format_number(col_stats['Mean'])}")
        ax.axvline(col_stats['Median'], color=COLORS['median'], linestyle='-', linewidth=2.5, label=f"Median: {format_number(col_stats['Median'])}")
        
        # Formatting
        ax.set_title(f"Distribution Analysis: {col.replace('_', ' ').title()}", pad=20)
        ax.set_xlabel(f"{col.replace('_', ' ').title()} Value")
        ax.set_ylabel("Density")
        ax.xaxis.set_major_formatter(formatter)
        ax.legend(loc='upper right', frameon=True, shadow=True, facecolor='white')
        
        # Annotations Box
        textstr = '\n'.join((
            f"Skewness: {col_stats['Skewness']:.2f}",
            f"Kurtosis: {col_stats['Kurtosis']:.2f}",
            f"Shape: {col_stats['Distribution Shape'].split('(')[0].strip()}"
        ))
        props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor=COLORS['secondary'])
        ax.text(0.95, 0.5, textstr, transform=ax.transAxes, fontsize=12,
                verticalalignment='center', horizontalalignment='right', bbox=props)

        plt.savefig(os.path.join(out_dir, f"{col}_histogram.png"))
        plt.close()

def plot_executive_dashboard(df, columns):
    """Generate combined Distribution Comparison Dashboard."""
    logging.info("Generating Combined Executive Dashboard...")
    
    # Select top 6 most important metrics for the dashboard
    dashboard_cols = [c for c in ['gross_sale_amt', 'net_sale_amt', 'fresh_ret_amt', 'expiry_amt', 'net_sale_qty', 'brkg_amt'] if c in columns]
    if not dashboard_cols: return
    
    fig = plt.figure(figsize=(24, 14))
    fig.patch.set_facecolor(COLORS['background'])
    gs = GridSpec(2, 3, figure=fig, wspace=0.3, hspace=0.4)
    
    for idx, col in enumerate(dashboard_cols):
        if idx >= 6: break
        
        row = idx // 3
        col_idx = idx % 3
        ax = fig.add_subplot(gs[row, col_idx])
        
        # Remove extreme outliers for dashboard clarity
        p99 = df[col].quantile(0.99)
        plot_data = df[df[col] <= p99][col]
        
        sns.histplot(plot_data, bins=30, color=COLORS['primary'], edgecolor='white', ax=ax)
        
        ax.axvline(df[col].mean(), color=COLORS['mean'], linestyle='--', linewidth=2)
        ax.axvline(df[col].median(), color=COLORS['median'], linestyle='-', linewidth=2)
        
        ax.set_title(col.replace('_', ' ').title(), fontsize=16, weight='bold')
        ax.set_xlabel("")
        ax.set_ylabel("Frequency")
        
        if idx == 0: # Add legend only to first plot to save space
            ax.legend(['Mean', 'Median'], loc='upper right')

    plt.suptitle("EXECUTIVE DISTRIBUTION & FREQUENCY DASHBOARD", fontsize=28, fontweight='bold', color=COLORS['primary'], y=0.98)
    plt.savefig("sales_eda/graphs/distribution/Combined_Distribution_Dashboard.png", facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()

# =============================================================================
# EXCEL & TEXT REPORT GENERATION
# =============================================================================

def generate_excel_report(stats_df):
    """Export calculated distribution statistics to Excel."""
    logging.info("Exporting data to Excel...")
    output_path = "sales_eda/excel/distribution/Histogram_Summary.xlsx"
    
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        stats_df.to_excel(writer, sheet_name='Distribution Metrics', index=False)
        
        # Formatting
        workbook = writer.book
        header_format = workbook.add_format({'bold': True, 'bg_color': COLORS['primary'], 'font_color': 'white'})
        worksheet = writer.sheets['Distribution Metrics']
        worksheet.set_row(0, None, header_format)
        worksheet.set_column('A:A', 20)
        worksheet.set_column('B:P', 15)

def generate_insights_report(stats_df):
    """Generate professional text insights regarding distribution shapes."""
    logging.info("Generating Business Insights Text Report...")
    output_path = "reports/phase_2/Histogram_Business_Insights.txt"

    # Isolate key findings
    highly_skewed = stats_df[stats_df['Skewness'] > 1.5]['Metric'].tolist()
    heavy_tails = stats_df[stats_df['Kurtosis'] > 3]['Metric'].tolist()
    
    try:
        net_sales = stats_df[stats_df['Metric'] == 'net_sale_amt'].iloc[0]
        net_sales_str = f"Mean: {format_number(net_sales['Mean'])} | Median: {format_number(net_sales['Median'])} | Shape: {net_sales['Distribution Shape']}"
    except IndexError:
        net_sales_str = "Data not available."

    report_content = f"""====================================================
EXECUTIVE DISTRIBUTION & HISTOGRAM INSIGHTS
====================================================

CORE DISTRIBUTION BEHAVIOR
----------------------------------------------------
Net Sales Amount Profile:
{net_sales_str}

Skewness Analysis (Data Concentration):
Highly Right Skewed Metrics: {', '.join(highly_skewed) if highly_skewed else 'None'}
*Business Interpretation:* A strong right skew indicates that the majority of transactional occurrences are small in value/volume, heavily weighted toward the left side of the histogram. The mean is being artificially pulled higher by a small number of massive institutional or bulk transactions (the "long tail").

Kurtosis Analysis (Outlier Risk):
Heavy Tailed (Leptokurtic) Metrics: {', '.join(heavy_tails) if heavy_tails else 'None'}
*Business Interpretation:* Leptokurtic distributions possess "fat tails", meaning extreme values, spikes, and anomalies occur much more frequently than a standard normal bell curve predicts. For supply chain and forecasting, standard deviation will under-predict risk.

====================================================
BUSINESS IMPACT & RECOMMENDATIONS
====================================================

1. Sales Concentration:
Because Net Sales and Gross Sales do not follow a perfect Normal (Gaussian) Distribution, executive planning should not rely exclusively on "Average (Mean) Sales". The Median represents the true "typical" performance much more accurately.

2. Inventory & Return Risk:
Metrics such as Expiry and Breakage typically exhibit extreme right skewness and heavy kurtosis. This indicates that while daily breakage is low, catastrophic batch failures or massive return events (spikes) dictate the total financial loss.

3. Forecasting Adjustments:
Standard regression models assume normally distributed variables. The high skewness observed requires logarithmic or Box-Cox transformations prior to feeding this data into Phase 3 predictive machine learning models.
====================================================
"""
    with open(output_path, "w", encoding='utf-8') as file:
        file.write(report_content)

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    logging.info("Starting Executive Distribution Analysis Pipeline...")
    
    setup_directories()
    
    try:
        # Load and clean data
        raw_data, valid_columns = load_data(INPUT_FILE)
        
        if not valid_columns:
            logging.error("No valid numeric columns found to analyze. Terminating.")
            exit()
            
        # Computations
        statistics_df = calculate_distribution_stats(raw_data, valid_columns)
        
        # Generation
        plot_individual_histograms(raw_data, valid_columns, statistics_df)
        plot_executive_dashboard(raw_data, valid_columns)
        generate_excel_report(statistics_df)
        generate_insights_report(statistics_df)
        
        logging.info("Distribution Analysis completed successfully.")
        
    except Exception as e:
        logging.error(f"Pipeline execution failed: {str(e)}")