code = '''
"""
05_statewise_sales.py
=============================================================================
Project: Weather Driven Pharmaceutical Sales Analytics & Forecasting
Module:  Statewise Sales Performance & Executive Dashboard
=============================================================================
Description:
    End-to-end pipeline for geographic sales analysis. Calculates state-wise
    KPIs, Pareto contributions (80/20 rule), and return ratios using the 
    root_state_name. Generates high-resolution visualizations and comprehensive 
    business reports.
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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

INPUT_FILE = "data/Sale_Details.xlsx"
BASE_DIRS = [
    "sales_eda/excel/business/",
    "sales_eda/graphs/business/",
    "reports/phase_2/"
]

METRIC_COLUMNS = [
    "gross_sale_amt", "gross_sale_qty", "fresh_ret_amt", "fresh_ret_qty", 
    "expiry_amt", "expiry_qty", "brkg_amt", "brkg_qty", "net_sale_amt", "net_sale_qty"
]

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

COLORS = {
    'primary': '#2c3e50',    
    'secondary': '#95a5a6',  
    'top_perf': '#27ae60',   
    'low_perf': '#e74c3c',   
    'returns': '#8e44ad',    
    'pareto': '#f39c12'      
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def setup_directories():
    for directory in BASE_DIRS:
        os.makedirs(directory, exist_ok=True)
    logging.info("Output directories verified and ready.")

def format_indian_currency(num, pos=None):
    if pd.isna(num): return "0.00"
    is_negative = num < 0
    abs_num = abs(num)
    if abs_num >= 1e7: formatted = f"{abs_num/1e7:.2f} Cr"
    elif abs_num >= 1e5: formatted = f"{abs_num/1e5:.2f} Lakh"
    elif abs_num >= 1e3: formatted = f"{abs_num/1e3:.2f} K"
    else: formatted = f"{abs_num:.2f}"
    return f"-{formatted}" if is_negative else formatted

def add_value_labels(ax, orient='v', format_func=None, is_pct=False):
    for p in ax.patches:
        val = p.get_width() if orient == 'h' else p.get_height()
        if val == 0 or pd.isna(val): continue
        text = f"{val:.1f}%" if is_pct else format_func(val) if format_func else str(val)
        if orient == 'h':
            x, y = p.get_width(), p.get_y() + p.get_height() / 2
            ax.text(x, y, f' {text}', va='center', ha='left', fontsize=10, fontweight='bold')
        else:
            x, y = p.get_x() + p.get_width() / 2, p.get_height()
            ax.text(x, y, f'{text}\\n', va='bottom', ha='center', fontsize=10, fontweight='bold')

# =============================================================================
# DATA PROCESSING & METRICS GENERATION
# =============================================================================

def load_and_preprocess(filepath):
    logging.info(f"Loading data from {filepath}...")
    try:
        df = pd.read_excel(filepath)
    except FileNotFoundError:
        logging.error(f"File not found: {filepath}")
        raise

    if 'root_state_name' not in df.columns:
        raise ValueError("Geographic column missing. Expected: 'root_state_name'")

    df['root_state_name'] = df['root_state_name'].astype(str).str.strip().str.title()
    
    for col in METRIC_COLUMNS:
        df[col] = df[col].fillna(0) if col in df.columns else 0
            
    return df

def calculate_state_metrics(df):
    logging.info("Calculating State Business Metrics...")
    
    state_df = df.groupby('root_state_name')[METRIC_COLUMNS].sum().reset_index()

    state_df['total_returns_amt'] = (
        state_df['fresh_ret_amt'] + state_df['expiry_amt'] + state_df['brkg_amt']
    )
    
    state_df['return_pct'] = np.where(
        state_df['gross_sale_amt'] > 0,
        (state_df['total_returns_amt'] / state_df['gross_sale_amt']) * 100, 0
    )

    state_df = state_df.sort_values(by='net_sale_amt', ascending=False).reset_index(drop=True)
    state_df['rank'] = state_df.index + 1

    total_net_sales = state_df['net_sale_amt'].sum()
    state_df['contribution_pct'] = (state_df['net_sale_amt'] / total_net_sales) * 100
    state_df['cumulative_contribution_pct'] = state_df['contribution_pct'].cumsum()

    state_df['tier'] = pd.cut(
        state_df['cumulative_contribution_pct'],
        bins=[0, 80, 95, 100],
        labels=['Tier 1 (Top 80%)', 'Tier 2 (Next 15%)', 'Tier 3 (Bottom 5%)']
    )

    return state_df

# =============================================================================
# PRESENTATION-QUALITY VISUALIZATIONS
# =============================================================================

def generate_individual_graphs(df):
    logging.info("Generating Presentation Quality Graphs...")
    formatter = FuncFormatter(format_indian_currency)
    
    # 1. Top 10
    top_10 = df.head(10).sort_values('net_sale_amt', ascending=True)
    fig, ax = plt.subplots()
    colors = [COLORS['primary']] * 9 + [COLORS['top_perf']]
    ax.barh(top_10['root_state_name'], top_10['net_sale_amt'], color=colors, edgecolor='none')
    ax.set_title("Top 10 States by Net Sales Contribution")
    ax.set_xlabel("Net Sales (INR)")
    ax.xaxis.set_major_formatter(formatter)
    add_value_labels(ax, orient='h', format_func=format_indian_currency)
    plt.savefig("sales_eda/graphs/business/1_Top_10_States.png")
    plt.close()

    # 2. Bottom 10
    bottom_10 = df[df['net_sale_amt'] > 0].tail(10).sort_values('net_sale_amt', ascending=True)
    fig, ax = plt.subplots()
    colors = [COLORS['low_perf']] * 3 + [COLORS['secondary']] * 7 
    ax.barh(bottom_10['root_state_name'], bottom_10['net_sale_amt'], color=colors)
    ax.set_title("Bottom 10 Active States by Net Sales")
    ax.set_xlabel("Net Sales (INR)")
    ax.xaxis.set_major_formatter(formatter)
    add_value_labels(ax, orient='h', format_func=format_indian_currency)
    plt.savefig("sales_eda/graphs/business/2_Bottom_10_States.png")
    plt.close()

    # 3. Pareto
    fig, ax1 = plt.subplots()
    pareto_df = df.head(15)
    ax1.bar(pareto_df['root_state_name'], pareto_df['net_sale_amt'], color=COLORS['primary'])
    ax1.set_ylabel("Net Sales (INR)", color=COLORS['primary'])
    ax1.yaxis.set_major_formatter(formatter)
    ax1.tick_params(axis='x', rotation=45)
    
    ax2 = ax1.twinx()
    ax2.plot(pareto_df['root_state_name'], pareto_df['cumulative_contribution_pct'], color=COLORS['pareto'], marker='D', ms=7, lw=3)
    ax2.set_ylabel("Cumulative Contribution (%)", color=COLORS['pareto'])
    ax2.yaxis.set_major_formatter(PercentFormatter())
    ax2.axhline(80, color='red', linestyle='--', alpha=0.5, label='80% Threshold')
    
    plt.title("State-wise Sales Pareto Analysis (Top 15)")
    fig.tight_layout()
    plt.savefig("sales_eda/graphs/business/3_State_Pareto_Analysis.png")
    plt.close()

    # 4. Return Ratio
    top_vol = df.head(10)
    fig, ax = plt.subplots()
    ax.bar(top_vol['root_state_name'], top_vol['return_pct'], color=COLORS['returns'])
    ax.set_title("Return Ratio Analysis (% of Gross Sales) - Top 10 States")
    ax.set_ylabel("Return Percentage (%)")
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.tick_params(axis='x', rotation=45)
    ax.axhline(top_vol['return_pct'].mean(), color='red', linestyle='--', label='Average Return %')
    add_value_labels(ax, orient='v', is_pct=True)
    ax.legend()
    plt.savefig("sales_eda/graphs/business/4_State_Return_Ratio.png")
    plt.close()

def generate_executive_dashboard(df):
    logging.info("Generating Executive State Dashboard...")
    fig = plt.figure(figsize=(22, 14))
    gs = GridSpec(3, 2, height_ratios=[1, 1, 0.4], hspace=0.4, wspace=0.2)
    formatter = FuncFormatter(format_indian_currency)

    # 1. Top 10
    ax1 = fig.add_subplot(gs[0, 0])
    top_10 = df.head(10).sort_values('net_sale_amt')
    ax1.barh(top_10['root_state_name'], top_10['net_sale_amt'], color=COLORS['primary'])
    ax1.patches[-1].set_facecolor(COLORS['top_perf'])
    ax1.set_title("Top 10 Revenue Generating States")
    ax1.xaxis.set_major_formatter(formatter)
    add_value_labels(ax1, orient='h', format_func=format_indian_currency)

    # 2. Pareto Line
    ax2 = fig.add_subplot(gs[0, 1])
    p_df = df.head(15)
    ax2.plot(p_df['root_state_name'], p_df['cumulative_contribution_pct'], color=COLORS['pareto'], marker='o', lw=3)
    ax2.fill_between(p_df['root_state_name'], p_df['cumulative_contribution_pct'], alpha=0.1, color=COLORS['pareto'])
    ax2.axhline(80, color='red', linestyle='--', alpha=0.7)
    ax2.set_title("Cumulative Contribution (Pareto Principle)")
    ax2.yaxis.set_major_formatter(PercentFormatter())
    ax2.tick_params(axis='x', rotation=45)

    # 3. Returns
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.bar(p_df['root_state_name'], p_df['total_returns_amt'], color=COLORS['returns'])
    ax3.set_title("Total Returns Impact by State")
    ax3.yaxis.set_major_formatter(formatter)
    ax3.tick_params(axis='x', rotation=45)

    # 4. Bubble
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.scatter(p_df['net_sale_qty'], p_df['net_sale_amt'], s=p_df['return_pct']*100, alpha=0.6, color=COLORS['primary'])
    for i, txt in enumerate(p_df['root_state_name']):
        ax4.annotate(txt, (p_df['net_sale_qty'].iloc[i], p_df['net_sale_amt'].iloc[i]), fontsize=9)
    ax4.set_title("Sales Amount vs Quantity (Bubble Size = Return %)")
    ax4.set_xlabel("Net Quantity")
    ax4.yaxis.set_major_formatter(formatter)

    # 5. KPI
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('tight')
    ax5.axis('off')
    
    top_state = df.iloc[0]
    tier_1_count = len(df[df['tier'] == 'Tier 1 (Top 80%)'])
    
    kpi_data = [
        ["Total States Active", str(len(df[df['net_sale_amt'] > 0]))],
        ["Top Performing State", f"{top_state['root_state_name']} ({format_indian_currency(top_state['net_sale_amt'])})"],
        ["Top State Contribution", f"{top_state['contribution_pct']:.1f}%"],
        ["States Driving 80% Revenue", str(tier_1_count)],
        ["Highest Return Ratio State", df.sort_values('return_pct', ascending=False).iloc[0]['root_state_name']]
    ]
    
    table = ax5.table(cellText=kpi_data, colLabels=["Geographic Metric", "Business Value"], loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 2.5)
    
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#34495e')
        else:
            cell.set_facecolor('#f2f5f8')

    plt.suptitle("EXECUTIVE GEOGRAPHIC PERFORMANCE DASHBOARD", fontsize=24, fontweight='bold', y=0.98)
    plt.savefig("sales_eda/graphs/business/Executive_Statewise_Dashboard.png", dpi=600, bbox_inches='tight')
    plt.close()

# =============================================================================
# EXCEL & TEXT REPORT GENERATION
# =============================================================================

def generate_excel_report(df):
    logging.info("Exporting data to Excel...")
    output_path = "sales_eda/excel/business/Statewise_Sales_Summary.xlsx"
    
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        exec_cols = ['rank', 'root_state_name', 'net_sale_amt', 'contribution_pct', 'tier']
        df[exec_cols].to_excel(writer, sheet_name='Executive KPI', index=False)
        df.to_excel(writer, sheet_name='Complete State Metrics', index=False)
        
        ret_cols = ['root_state_name', 'gross_sale_amt', 'total_returns_amt', 'return_pct']
        ret_df = df.sort_values('return_pct', ascending=False)
        ret_df[ret_cols].to_excel(writer, sheet_name='Return Analysis', index=False)

def generate_insights_report(df):
    logging.info("Generating Business Insights Text Report...")
    output_path = "reports/phase_2/Statewise_Business_Insights.txt"

    top_state = df.iloc[0]
    bottom_state = df[df['net_sale_amt'] > 0].iloc[-1]
    tier_1 = df[df['tier'] == 'Tier 1 (Top 80%)']
    high_return_state = df[df['gross_sale_amt'] > 100000].sort_values('return_pct', ascending=False).iloc[0]

    report_content = f"""====================================================
GEOGRAPHIC BUSINESS INSIGHTS & SUMMARY
====================================================

MARKET PENETRATION
----------------------------------------------------
Total Active Regions:    {len(df[df['net_sale_amt'] > 0])}
States Driving 80% Rev:  {len(tier_1)} states out of {len(df)}

TOP & BOTTOM PERFORMERS
----------------------------------------------------
Top Performing State:    {top_state['root_state_name']}
  -> Net Sales:          {format_indian_currency(top_state['net_sale_amt'])}
  -> National Share:     {top_state['contribution_pct']:.2f}%

Lagging Market (Active): {bottom_state['root_state_name']}
  -> Net Sales:          {format_indian_currency(bottom_state['net_sale_amt'])}

EFFICIENCY & LEAKAGE
----------------------------------------------------
Highest Return Margin:   {high_state := high_return_state['root_state_name']}
  -> Return Ratio:       {high_return_state['return_pct']:.2f}% of Gross Sales
  -> Absolute Return:    {format_indian_currency(high_return_state['total_returns_amt'])}

====================================================
BUSINESS INTERPRETATION & RECOMMENDATIONS
====================================================

Geographic Concentration:
The business exhibits heavy geographic dependency. {len(tier_1)} states generate 80% of the total corporate revenue. Resource allocation (marketing, sales force) should be highly indexed to these regions.

Growth Opportunity:
Secondary markets present high upside potential. Tier 2 states need targeted promotional strategies to increase market penetration and reduce reliance on {top_state['root_state_name']}.

Supply Chain Concern:
{high_state} operates at an unsustainably high return ratio ({high_return_state['return_pct']:.2f}%). This indicates a severe mismatch in local demand forecasting, logistics damage, or aggressive channel stuffing by regional sales teams. 

Action Items:
1. Conduct an immediate audit of distribution partners in {high_state} regarding expired/broken inventory.
2. Rebalance Q3/Q4 logistics budgets to ensure the top {len(tier_1)} states face zero stock-outs.
3. Overlay weather pattern data specifically on the Tier 1 states to build a predictive demand model for upcoming quarters.
====================================================
"""
    with open(output_path, "w", encoding='utf-8') as file:
        file.write(report_content)

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    logging.info("Starting Statewise Geographic Analytics Pipeline...")
    setup_directories()
    
    try:
        raw_data = load_and_preprocess(INPUT_FILE)
        state_metrics = calculate_state_metrics(raw_data)
        
        generate_individual_graphs(state_metrics)
        generate_executive_dashboard(state_metrics)
        generate_excel_report(state_metrics)
        generate_insights_report(state_metrics)
        
        logging.info("Geographic pipeline execution completed successfully.")
    except Exception as e:
        logging.error(f"Pipeline execution failed: {str(e)}")
'''

with open('05_statewise_sales-v2.py', 'w') as f:
    f.write(code)

print("[file-tag: 05_statewise_sales-v2.py]")