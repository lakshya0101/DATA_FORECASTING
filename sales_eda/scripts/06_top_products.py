"""
06_top_products.py
=============================================================================
Project: Weather Driven Pharmaceutical Sales Analytics & Forecasting
Module:  Executive Product Performance Analysis & Scorecard
=============================================================================
Description:
    End-to-end pipeline for granular product-level analytics. Calculates
    ABC classifications, Pareto (80/20) distributions, and proprietary 
    product performance scores. Generates Fortune 500 presentation-grade 
    visualizations, multidimensional Excel reports, and executive summaries.
=============================================================================
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter, PercentFormatter

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

# Required Columns
REQUIRED_COLUMNS = [
    "matnr", "item_name", "root_state_name", "billing_month",
    "gross_sale_amt", "gross_sale_qty", "fresh_ret_amt", "fresh_ret_qty",
    "expiry_amt", "expiry_qty", "brkg_amt", "brkg_qty", "net_sale_amt", "net_sale_qty"
]

METRIC_COLUMNS = [
    "gross_sale_amt", "gross_sale_qty", "fresh_ret_amt", "fresh_ret_qty",
    "expiry_amt", "expiry_qty", "brkg_amt", "brkg_qty", "net_sale_amt", "net_sale_qty"
]

# Matplotlib Global Parameters for Presentation Quality
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'sans-serif'],
    'axes.titlesize': 22,
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
    'primary': '#2c3e50',    # Navy
    'secondary': '#bdc3c7',  # Silver
    'top_perf': '#27ae60',   # Emerald Green
    'low_perf': '#c0392b',   # Dark Red
    'returns': '#8e44ad',    # Purple
    'pareto': '#f39c12',     # Orange
    'qty': '#2980b9'         # Blue
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def setup_directories():
    """Create necessary output directories if they do not exist."""
    for directory in BASE_DIRS:
        os.makedirs(directory, exist_ok=True)
    logging.info("Output directories verified and ready.")

def format_indian_currency(num, pos=None):
    """Format large numbers into Indian currency system (Lakhs, Crores)."""
    if pd.isna(num) or num == 0:
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

def add_value_labels(ax, orient='v', format_func=None, is_pct=False):
    """Add precise value labels to bar charts."""
    for p in ax.patches:
        val = p.get_width() if orient == 'h' else p.get_height()
        if val == 0 or pd.isna(val): continue
        
        text = f"{val:.1f}%" if is_pct else format_func(val) if format_func else str(int(val))
        
        if orient == 'h':
            x = p.get_width()
            y = p.get_y() + p.get_height() / 2
            ax.text(x, y, f' {text}', va='center', ha='left', fontsize=10, fontweight='bold')
        else:
            x = p.get_x() + p.get_width() / 2
            y = p.get_height()
            ax.text(x, y, f'{text}\n', va='bottom', ha='center', fontsize=10, fontweight='bold')

# =============================================================================
# DATA PROCESSING & METRICS GENERATION
# =============================================================================

def load_and_validate(filepath):
    """Load dataset, validate columns, and handle missing values."""
    logging.info(f"Loading data from {filepath}...")
    try:
        df = pd.read_excel(filepath)
    except FileNotFoundError:
        logging.error(f"File not found: {filepath}")
        raise

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df[METRIC_COLUMNS] = df[METRIC_COLUMNS].fillna(0)
    df['item_name'] = df['item_name'].astype(str).str.strip().str.upper()
    return df

def aggregate_product_metrics(df):
    """Generate product-level business metrics, ABC, Pareto, and Scorecard."""
    logging.info("Calculating Product Business Metrics...")
    
    # Base Aggregation
    prod_df = df.groupby(['matnr', 'item_name'])[METRIC_COLUMNS].sum().reset_index()
    
    # Advanced Metrics
    prod_df['total_returns_amt'] = prod_df['fresh_ret_amt'] + prod_df['expiry_amt'] + prod_df['brkg_amt']
    prod_df['total_returns_qty'] = prod_df['fresh_ret_qty'] + prod_df['expiry_qty'] + prod_df['brkg_qty']
    
    prod_df['return_pct'] = np.where(
        prod_df['gross_sale_amt'] > 0,
        (prod_df['total_returns_amt'] / prod_df['gross_sale_amt']) * 100, 0
    )
    prod_df['net_gross_ratio'] = np.where(
        prod_df['gross_sale_amt'] > 0,
        (prod_df['net_sale_amt'] / prod_df['gross_sale_amt']) * 100, 0
    )
    prod_df['qty_efficiency'] = np.where(
        prod_df['gross_sale_qty'] > 0,
        (prod_df['net_sale_qty'] / prod_df['gross_sale_qty']) * 100, 0
    )

    # Sort & Rank by Net Sales
    prod_df = prod_df.sort_values(by='net_sale_amt', ascending=False).reset_index(drop=True)
    prod_df['product_rank'] = prod_df.index + 1
    
    # Pareto & Contribution
    total_net = prod_df['net_sale_amt'].sum()
    prod_df['contribution_pct'] = (prod_df['net_sale_amt'] / total_net) * 100
    prod_df['cumulative_pct'] = prod_df['contribution_pct'].cumsum()
    
    # ABC Classification
    prod_df['abc_category'] = pd.cut(
        prod_df['cumulative_pct'],
        bins=[0, 80, 95, 100],
        labels=['A (Top 80%)', 'B (Next 15%)', 'C (Bottom 5%)']
    )
    
    # Monthly Average (Assuming uniform months across dataset for all products)
    total_months = df['billing_month'].nunique()
    prod_df['avg_monthly_sales'] = prod_df['net_sale_amt'] / total_months if total_months > 0 else 0

    return build_product_scorecard(prod_df)

def build_product_scorecard(df):
    """Generate proprietary 0-100 Product Performance Scorecard."""
    logging.info("Generating Product Scorecards...")
    
    max_sales = df['net_sale_amt'].max()
    max_qty = df['net_sale_qty'].max()
    max_contrib = df['contribution_pct'].max()
    
    # 1. Sales Volume Score (max 25 pts)
    sales_score = (df['net_sale_amt'] / max_sales) * 25
    
    # 2. Efficiency Score (max 25 pts) - based on Net/Gross ratio
    eff_score = (df['net_gross_ratio'] / 100) * 25
    
    # 3. Return Penalty/Bonus (max 25 pts) - 0% return gets 25 pts
    ret_score = np.maximum(0, (1 - (df['return_pct'] / 100))) * 25
    
    # 4. Market Contribution Score (max 25 pts)
    contrib_score = (df['contribution_pct'] / max_contrib) * 25
    
    df['business_score_100'] = np.round(sales_score + eff_score + ret_score + contrib_score, 1)
    
    return df

# =============================================================================
# PRESENTATION-QUALITY VISUALIZATIONS
# =============================================================================

def generate_individual_graphs(df):
    """Generate the 9 specified individual high-resolution charts."""
    logging.info("Generating Presentation Quality Graphs...")
    
    formatter = FuncFormatter(format_indian_currency)
    out_dir = "sales_eda/graphs/business/"
    
    def _save_plot(filename):
        plt.savefig(os.path.join(out_dir, f"{filename}.png"))
        plt.savefig(os.path.join(out_dir, f"{filename}.pdf"))
        plt.close()

    # 1. Top 10 Net Sales
    top10 = df.head(10).sort_values('net_sale_amt', ascending=True)
    fig, ax = plt.subplots()
    colors = [COLORS['primary']] * 9 + [COLORS['top_perf']]
    ax.barh(top10['item_name'], top10['net_sale_amt'], color=colors)
    ax.set_title("Top 10 Products by Net Sales")
    ax.xaxis.set_major_formatter(formatter)
    add_value_labels(ax, orient='h', format_func=format_indian_currency)
    _save_plot("01_Top10_NetSales")

    # 2. Top 20 Net Sales
    top20 = df.head(20).sort_values('net_sale_amt', ascending=True)
    fig, ax = plt.subplots(figsize=(16, 12)) # Taller for 20 items
    colors = [COLORS['secondary']] * 10 + [COLORS['primary']] * 9 + [COLORS['top_perf']]
    ax.barh(top20['item_name'], top20['net_sale_amt'], color=colors)
    ax.set_title("Top 20 Products by Net Sales")
    ax.xaxis.set_major_formatter(formatter)
    add_value_labels(ax, orient='h', format_func=format_indian_currency)
    _save_plot("02_Top20_NetSales")

    # 3. Product Contribution % (Top 10 vs Rest)
    fig, ax = plt.subplots()
    top10_sum = df.head(10)['contribution_pct'].sum()
    rest_sum = 100 - top10_sum
    ax.pie([top10_sum, rest_sum], labels=['Top 10 Products', 'All Other Products'], 
           autopct='%1.1f%%', colors=[COLORS['top_perf'], COLORS['secondary']], 
           startangle=90, explode=(0.1, 0), textprops={'fontsize': 14, 'weight': 'bold'})
    ax.set_title("Top 10 Products Sales Contribution %")
    _save_plot("03_Product_Contribution")

    # 4. Return Analysis (Scatter)
    fig, ax = plt.subplots()
    top50 = df.head(50)
    ax.scatter(top50['net_sale_amt'], top50['total_returns_amt'], 
               s=top50['return_pct']*20, c=COLORS['returns'], alpha=0.7)
    ax.set_title("Return Analysis: Sales vs Returns (Top 50)")
    ax.set_xlabel("Net Sales (INR)")
    ax.set_ylabel("Total Returns (INR)")
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    for i, txt in enumerate(top50.head(5)['item_name']):
        ax.annotate(txt, (top50['net_sale_amt'].iloc[i], top50['total_returns_amt'].iloc[i]), fontsize=10)
    _save_plot("04_Product_Return_Analysis")

    # 5. Product Quantity
    top10_qty = df.sort_values('net_sale_qty', ascending=False).head(10).sort_values('net_sale_qty')
    fig, ax = plt.subplots()
    ax.barh(top10_qty['item_name'], top10_qty['net_sale_qty'], color=COLORS['qty'])
    ax.set_title("Top 10 Products by Net Quantity")
    ax.set_xlabel("Net Quantity (Units)")
    add_value_labels(ax, orient='h')
    _save_plot("05_Product_Quantity")

    # 6. Pareto Chart
    fig, ax1 = plt.subplots()
    pareto_df = df.head(30)
    ax1.bar(pareto_df['item_name'], pareto_df['net_sale_amt'], color=COLORS['primary'])
    ax1.set_ylabel("Net Sales (INR)", color=COLORS['primary'])
    ax1.yaxis.set_major_formatter(formatter)
    ax1.tick_params(axis='x', rotation=90)
    
    ax2 = ax1.twinx()
    ax2.plot(pareto_df['item_name'], pareto_df['cumulative_pct'], color=COLORS['pareto'], marker='D', lw=3)
    ax2.set_ylabel("Cumulative %", color=COLORS['pareto'])
    ax2.yaxis.set_major_formatter(PercentFormatter())
    ax2.axhline(80, color='red', linestyle='--', label='80% Threshold')
    plt.title("Product Sales Pareto Analysis (Top 30)")
    fig.tight_layout()
    _save_plot("06_Product_Pareto")

    # 7. ABC Analysis
    fig, ax = plt.subplots()
    abc_counts = df['abc_category'].value_counts().sort_index()
    ax.bar(abc_counts.index, abc_counts.values, color=[COLORS['top_perf'], COLORS['primary'], COLORS['secondary']])
    ax.set_title("ABC Classification: Product Count per Category")
    ax.set_ylabel("Number of Products")
    add_value_labels(ax, orient='v')
    _save_plot("07_Product_ABC_Analysis")

    # 8. Sales Efficiency (Net/Gross Ratio)
    fig, ax = plt.subplots()
    top10_eff = df.head(10)
    ax.bar(top10_eff['item_name'], top10_eff['net_gross_ratio'], color=COLORS['top_perf'])
    ax.set_title("Sales Efficiency (Net/Gross Ratio) - Top 10 Revenue Products")
    ax.set_ylabel("Efficiency (%)")
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylim(0, 110)
    add_value_labels(ax, orient='v', is_pct=True)
    fig.tight_layout()
    _save_plot("08_Product_Sales_Efficiency")

    # 9. Return Percentage
    fig, ax = plt.subplots()
    worst_ret = df[df['gross_sale_amt'] > df['gross_sale_amt'].median()].sort_values('return_pct', ascending=False).head(10)
    ax.bar(worst_ret['item_name'], worst_ret['return_pct'], color=COLORS['low_perf'])
    ax.set_title("Highest Return Percentage (Above Median Sales Volume)")
    ax.set_ylabel("Return %")
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.tick_params(axis='x', rotation=45)
    add_value_labels(ax, orient='v', is_pct=True)
    fig.tight_layout()
    _save_plot("09_Product_Return_Percentage")

def generate_executive_dashboard(df):
    """Combine key metrics into a singular presentation-ready dashboard image."""
    logging.info("Generating Executive Product Dashboard...")
    
    fig = plt.figure(figsize=(24, 16))
    gs = GridSpec(3, 3, height_ratios=[0.5, 1.2, 1], hspace=0.5, wspace=0.3)
    formatter = FuncFormatter(format_indian_currency)

    # 1. KPI Table (Spans Top Row)
    ax_kpi = fig.add_subplot(gs[0, :])
    ax_kpi.axis('tight')
    ax_kpi.axis('off')
    
    kpi_data = [
        ["Total Products", str(len(df))],
        ["Total Net Sales", format_indian_currency(df['net_sale_amt'].sum())],
        ["Total Returns", format_indian_currency(df['total_returns_amt'].sum())],
        ["Top Product", f"{df.iloc[0]['item_name']} ({format_indian_currency(df.iloc[0]['net_sale_amt'])})"],
        ["Top 10 Contribution", f"{df.head(10)['contribution_pct'].sum():.1f}%"],
        ["Products driving 80%", str(len(df[df['cumulative_pct'] <= 80]) + 1)]
    ]
    
    table = ax_kpi.table(cellText=[[x[1] for x in kpi_data]], 
                         colLabels=[x[0] for x in kpi_data], 
                         loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 3)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor(COLORS['primary'])
        else:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#ecf0f1')

    # 2. Top 10 Bar Chart
    ax_top = fig.add_subplot(gs[1, 0:2])
    top10 = df.head(10).sort_values('net_sale_amt')
    ax_top.barh(top10['item_name'], top10['net_sale_amt'], color=COLORS['primary'])
    ax_top.patches[-1].set_facecolor(COLORS['top_perf'])
    ax_top.set_title("Top 10 Revenue Generating Products")
    ax_top.xaxis.set_major_formatter(formatter)
    add_value_labels(ax_top, orient='h', format_func=format_indian_currency)

    # 3. ABC Classification Pie
    ax_abc = fig.add_subplot(gs[1, 2])
    abc_sums = df.groupby('abc_category', observed=False)['net_sale_amt'].sum()
    ax_abc.pie(abc_sums, labels=abc_sums.index, autopct='%1.1f%%', 
               colors=[COLORS['top_perf'], COLORS['primary'], COLORS['secondary']], 
               startangle=90, textprops={'fontsize': 12, 'weight': 'bold'})
    ax_abc.set_title("Sales by ABC Classification")

    # 4. Pareto Line
    ax_par = fig.add_subplot(gs[2, 0])
    p_df = df.head(20)
    ax_par.plot(p_df['item_name'], p_df['cumulative_pct'], color=COLORS['pareto'], marker='o', lw=3)
    ax_par.axhline(80, color='red', linestyle='--', alpha=0.7)
    ax_par.set_title("Cumulative Sales Contribution (Top 20)")
    ax_par.yaxis.set_major_formatter(PercentFormatter())
    ax_par.tick_params(axis='x', rotation=90)

    # 5. Returns vs Net
    ax_ret = fig.add_subplot(gs[2, 1])
    ax_ret.scatter(df['net_sale_amt'], df['total_returns_amt'], alpha=0.6, color=COLORS['returns'])
    ax_ret.set_title("Net Sales vs Total Returns")
    ax_ret.xaxis.set_major_formatter(formatter)
    ax_ret.yaxis.set_major_formatter(formatter)
    ax_ret.set_xlabel("Net Sales")

    # 6. Top Scorecard Table
    ax_score = fig.add_subplot(gs[2, 2])
    ax_score.axis('tight')
    ax_score.axis('off')
    score_df = df.head(5)[['item_name', 'business_score_100', 'return_pct']]
    score_table = ax_score.table(cellText=score_df.values, 
                                 colLabels=['Product', 'BI Score (0-100)', 'Return %'], 
                                 loc='center', cellLoc='center')
    score_table.auto_set_font_size(False)
    score_table.set_fontsize(12)
    score_table.scale(1, 2)
    for (row, col), cell in score_table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor(COLORS['primary'])

    plt.suptitle("EXECUTIVE PRODUCT PERFORMANCE DASHBOARD", fontsize=28, fontweight='bold', y=0.96)
    plt.savefig("sales_eda/graphs/business/10_Executive_Product_Dashboard.png", dpi=600, bbox_inches='tight')
    plt.close()

# =============================================================================
# EXCEL & TEXT REPORT GENERATION
# =============================================================================

def generate_excel_report(df):
    """Generate multi-dimensional Excel workbook."""
    logging.info("Exporting data to Excel...")
    out_path = "sales_eda/excel/business/Product_Performance_Summary.xlsx"
    
    kpi_df = pd.DataFrame({
        "Metric": ["Total Products", "Total Net Sales", "Total Gross Sales", "Total Returns",
                   "Avg Product Sales", "Median Product Sales", "Top 10 Contribution %", "Top 20 Contribution %", "Top 50 Contribution %"],
        "Value": [len(df), df['net_sale_amt'].sum(), df['gross_sale_amt'].sum(), df['total_returns_amt'].sum(),
                  df['net_sale_amt'].mean(), df['net_sale_amt'].median(), df.head(10)['contribution_pct'].sum(),
                  df.head(20)['contribution_pct'].sum(), df.head(50)['contribution_pct'].sum()]
    })

    with pd.ExcelWriter(out_path, engine='xlsxwriter') as writer:
        kpi_df.to_excel(writer, sheet_name='Executive KPI', index=False)
        df.head(10).to_excel(writer, sheet_name='Top10 Products', index=False)
        df.head(20).to_excel(writer, sheet_name='Top20 Products', index=False)
        df.head(50).to_excel(writer, sheet_name='Top50 Products', index=False)
        
        # Select columns for summary
        summary_cols = ['product_rank', 'matnr', 'item_name', 'abc_category', 'business_score_100', 
                        'net_sale_amt', 'net_sale_qty', 'return_pct', 'contribution_pct']
        df[summary_cols].to_excel(writer, sheet_name='Product Summary', index=False)
        
        df[['item_name', 'net_sale_amt', 'contribution_pct', 'cumulative_pct']].to_excel(writer, sheet_name='Contribution', index=False)
        df.sort_values('return_pct', ascending=False)[['item_name', 'gross_sale_amt', 'total_returns_amt', 'return_pct']].to_excel(writer, sheet_name='Returns', index=False)
        
        abc_summary = df.groupby('abc_category', observed=False).agg(
            Count=('matnr', 'count'),
            Net_Sales=('net_sale_amt', 'sum'),
            Returns=('total_returns_amt', 'sum')
        ).reset_index()
        abc_summary.to_excel(writer, sheet_name='ABC Classification', index=False)
        
        df.head(50)[['item_name', 'cumulative_pct']].to_excel(writer, sheet_name='Pareto Analysis', index=False)
        df[['item_name', 'net_gross_ratio', 'qty_efficiency']].to_excel(writer, sheet_name='Sales Efficiency', index=False)

def generate_insights_report(df):
    """Generate executive text summary mapping product business interpretations."""
    logging.info("Generating Business Insights Text Report...")
    out_path = "reports/phase_2/Product_Business_Insights.txt"

    top_prod = df.iloc[0]
    bottom_prod = df[df['net_sale_amt'] > 0].iloc[-1]
    high_ret_prod = df[df['gross_sale_amt'] > df['gross_sale_amt'].median()].sort_values('return_pct', ascending=False).iloc[0]
    high_qty_prod = df.sort_values('net_sale_qty', ascending=False).iloc[0]
    
    count_80_pct = len(df[df['cumulative_pct'] <= 80]) + 1
    abc_a_count = len(df[df['abc_category'] == 'A (Top 80%)'])

    report_content = f"""====================================================
EXECUTIVE PRODUCT BUSINESS INSIGHTS & SUMMARY
====================================================

CATALOG OVERVIEW
----------------------------------------------------
Total Unique Products:     {len(df)}
Products for 80% Sales:    {count_80_pct} (Hyper-concentration)

PERFORMANCE EXTREMES
----------------------------------------------------
Top Selling Product:       {top_prod['item_name']} 
  -> Net Sales:            {format_indian_currency(top_prod['net_sale_amt'])}
  -> BI Score (0-100):     {top_prod['business_score_100']}

Lowest Selling (Active):   {bottom_prod['item_name']}
  -> Net Sales:            {format_indian_currency(bottom_prod['net_sale_amt'])}

Highest Volume Product:    {high_qty_prod['item_name']} ({high_qty_prod['net_sale_qty']} units)

CONTRIBUTION METRICS
----------------------------------------------------
Top 10 Contribution:       {df.head(10)['contribution_pct'].sum():.2f}% of Total Revenue
Top 20 Contribution:       {df.head(20)['contribution_pct'].sum():.2f}% of Total Revenue

EFFICIENCY & LEAKAGE
----------------------------------------------------
Highest Return Product:    {high_ret_prod['item_name']}
  -> Return Ratio:         {high_ret_prod['return_pct']:.2f}%
  -> Lost Revenue:         {format_indian_currency(high_ret_prod['total_returns_amt'])}

ABC SUMMARY
----------------------------------------------------
Category A (80% Rev):      {abc_a_count} Products
Category B (15% Rev):      {len(df[df['abc_category'] == 'B (Next 15%)'])} Products
Category C ( 5% Rev):      {len(df[df['abc_category'] == 'C (Bottom 5%)'])} Products

====================================================
BUSINESS OBSERVATIONS & RECOMMENDATIONS
====================================================

Top Performing Products:
The portfolio relies heavily on {top_prod['item_name']} and the top 10 SKUs, which drive {df.head(10)['contribution_pct'].sum():.1f}% of revenue. Ensure uncompromising supply chain priority and safety stock for these Category A items.

Poor Performing Products:
Category C contains {len(df[df['abc_category'] == 'C (Bottom 5%)'])} products contributing just 5% of revenue. A strategic SKU rationalization review is highly recommended to eliminate catalog bloat, reduce warehousing costs, and improve cash flow.

Products with High Returns:
{high_ret_prod['item_name']} is bleeding margin with a {high_ret_prod['return_pct']:.1f}% return rate. Investigate batch quality, shelf-life parameters, or potential transit damage immediately.

Inventory Attention Required:
The disparity between the Highest Volume Product ({high_qty_prod['item_name']}) and Top Revenue Product ({top_prod['item_name']}) indicates varying margin profiles. High-volume, low-margin goods consume logistics bandwidth; optimize shipping routes for these specific SKUs.
====================================================
"""

    with open(out_path, "w", encoding='utf-8') as file:
        file.write(report_content)

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    logging.info("Starting Product Performance Analytics Pipeline...")
    
    setup_directories()
    
    try:
        raw_data = load_and_validate(INPUT_FILE)
        
        product_metrics = aggregate_product_metrics(raw_data)
        
        generate_individual_graphs(product_metrics)
        generate_executive_dashboard(product_metrics)
        generate_excel_report(product_metrics)
        generate_insights_report(product_metrics)
        
        logging.info("Product pipeline execution completed successfully.")
        
    except Exception as e:
        logging.error(f"Pipeline execution failed: {str(e)}")