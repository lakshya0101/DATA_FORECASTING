import pandas as pd
import numpy as np
import sys
sys.path.insert(0, 'c:/Users/lakshya.dogra/Desktop/IDSP_Disease_Project')

df = pd.read_csv('mahacef200_analysis/data/mahacef200_master_dataset.csv')
print('Shape:', df.shape)
print('Columns:', list(df.columns))

# Missing analysis
weather_cols = ['avg_temperature', 'avg_humidity', 'total_rainfall_mm']
print('\n=== MISSING WEATHER ANALYSIS ===')
for col in weather_cols:
    print(f'{col}: {df[col].isnull().sum()} missing')

# Which months are missing (unique)?
missing_months = sorted(df[df['avg_temperature'].isnull()]['billing_month'].unique())
present_months = sorted(df[~df['avg_temperature'].isnull()]['billing_month'].unique())
print('\nMissing months:', missing_months)
print('Present months:', present_months)

# Build monthly climatology from raw weather
weather = pd.read_excel('data/WEATHER_DATASET.xlsx')
weather['datetime'] = pd.to_datetime(weather['datetime'])
weather['billing_month'] = weather['datetime'].dt.year * 100 + weather['datetime'].dt.month
weather['month_num'] = weather['datetime'].dt.month

monthly = weather.groupby('billing_month').agg(
    avg_temperature=('temp', 'mean'),
    avg_humidity=('humidity', 'mean'),
    total_rainfall_mm=('precip', 'sum'),
).reset_index()
monthly['month_num'] = monthly['billing_month'] % 100

print('\n=== AVAILABLE MONTHLY WEATHER ===')
for _, row in monthly.iterrows():
    print(f"  {int(row['billing_month'])}  temp={row['avg_temperature']:.2f}  hum={row['avg_humidity']:.2f}  rain={row['total_rainfall_mm']:.2f}")

clim = monthly.groupby('month_num').agg(
    clim_temp=('avg_temperature', 'mean'),
    clim_hum=('avg_humidity', 'mean'),
    clim_rain=('total_rainfall_mm', 'mean'),
    n_obs=('billing_month', 'count'),
).reset_index()
print('\n=== CLIMATOLOGY BY CALENDAR MONTH ===')
print(clim.to_string())

# Check which missing months can be filled by climatology
month_names = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
               7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
print('\n=== IMPUTATION PLAN ===')
for m in missing_months:
    mon = m % 100
    if mon in clim['month_num'].values:
        row = clim[clim['month_num'] == mon].iloc[0]
        print(f"  {m} ({month_names[mon]}): climatology available (n={int(row['n_obs'])}) -> temp={row['clim_temp']:.2f}, hum={row['clim_hum']:.2f}, rain={row['clim_rain']:.2f}")
    else:
        print(f"  {m} ({month_names[mon]}): NO CLIMATOLOGY - need interpolation")
