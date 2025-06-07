

import pandas as pd
import numpy as np
from datetime import timedelta, datetime

# Load CSV
df = pd.read_csv(r'D:\Internship\Muhurat\nifty50_muhurat_data.csv')

# Convert timestamp and muhurat_date
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
df['muhurat_date'] = pd.to_datetime(df['muhurat_date'], format="%Y-%m-%d", errors='coerce')

df.dropna(subset=['timestamp', 'muhurat_date'], inplace=True)
df.sort_values(by='timestamp', inplace=True)
df.reset_index(drop=True, inplace=True)

# Debug info
print("\n🧪 Sample timestamp vs muhurat_date comparison:")
print(df[['timestamp', 'muhurat_date']].drop_duplicates().head(10))

# Sets for comparison
timestamp_dates = set(df['timestamp'].dt.date.unique())
muhurat_dates = set(df['muhurat_date'].dt.date.unique())
common_dates = timestamp_dates & muhurat_dates

print(f"\n📅 Unique timestamp dates: {sorted(timestamp_dates)}")
print(f"🪔 Muhurat dates: {sorted(muhurat_dates)}")
print(f"✅ Common dates: {sorted(common_dates)}")

print("\n🔍 Row count per Muhurat date:\n", df.groupby('muhurat_date').size())

# Split data
train_df = df[(df['muhurat_date'].dt.year >= 2012) & (df['muhurat_date'].dt.year <= 2021)]
test_df = df[(df['muhurat_date'].dt.year >= 2022) & (df['muhurat_date'].dt.year <= 2023)]

train_muhurat_dates = train_df['muhurat_date'].drop_duplicates().sort_values()
test_muhurat_dates = test_df['muhurat_date'].drop_duplicates().sort_values()

# Function to compute average return by time
def compute_intraday_return_by_time(df, muhurat_dates):
    returns_by_time = {}

    for date in muhurat_dates:
        day_data = df[df['muhurat_date'] == date].copy()
        if day_data.empty:
            print(f"[❌] No data for: {date.date()}")
            continue

        day_data['time'] = day_data['timestamp'].dt.time
        day_data.set_index('timestamp', inplace=True)

        for t in day_data['time'].unique():
            prices = day_data[day_data['time'] == t]['close']
            if len(prices) < 2:
                continue
            ret = prices.iloc[-1] - prices.iloc[0]
            returns_by_time.setdefault(t, []).append(ret)

    if not returns_by_time:
        print("[ERROR] Could not compute intraday returns.")

    avg_returns = {t: np.mean(r) for t, r in returns_by_time.items()}
    sorted_avg_returns = dict(sorted(avg_returns.items(), key=lambda x: x[1], reverse=True))
    return sorted_avg_returns

# Analyze training data
returns_by_time = compute_intraday_return_by_time(train_df, train_muhurat_dates)

if not returns_by_time:
    print("❌ No return statistics. Exiting.")
    exit(1)

best_time = list(returns_by_time.keys())[0]
worst_time = list(returns_by_time.keys())[-1]

print(f"\n✅ Best Time (Train): {best_time}")
print(f"✅ Worst Time (Train): {worst_time}")

# Nearest time finder
def find_nearest_time(target_time, time_list):
    target_dt = datetime.strptime(str(target_time), '%H:%M:%S')
    available_dt = [datetime.strptime(str(t), '%H:%M:%S') for t in time_list]
    nearest_dt = min(available_dt, key=lambda t: abs(t - target_dt))
    return nearest_dt.time()

# Predict on test data
predictions = []

for date in test_muhurat_dates:
    buffer_start = date - timedelta(days=5)
    buffer_end = date + timedelta(days=5)

    muhurat_data = test_df[(test_df['timestamp'].dt.date >= buffer_start.date()) &
                           (test_df['timestamp'].dt.date <= buffer_end.date())]

    muhurat_day_data = muhurat_data[muhurat_data['muhurat_date'] == date].copy()
    if muhurat_day_data.empty:
        print(f"[WARNING] No test data found for Muhurat date: {date.date()}")
        continue

    muhurat_day_data['time'] = muhurat_day_data['timestamp'].dt.time

    # Match or find nearest best time
    best_prices = muhurat_day_data[muhurat_day_data['time'] == best_time]['close']
    if best_prices.empty:
        nearest_best = find_nearest_time(best_time, muhurat_day_data['time'].unique())
        print(f"[⚠️] Using nearest best_time for {date.date()}: {nearest_best}")
        best_time = nearest_best
        best_prices = muhurat_day_data[muhurat_day_data['time'] == best_time]['close']

    # Match or find nearest worst time
    worst_prices = muhurat_day_data[muhurat_day_data['time'] == worst_time]['close']
    if worst_prices.empty:
        nearest_worst = find_nearest_time(worst_time, muhurat_day_data['time'].unique())
        print(f"[⚠️] Using nearest worst_time for {date.date()}: {nearest_worst}")
        worst_time = nearest_worst
        worst_prices = muhurat_day_data[muhurat_day_data['time'] == worst_time]['close']

    result = {
        'date': date.strftime('%d-%m-%Y'),
        'best_time': best_time.strftime('%H:%M:%S'),
        'best_price': best_prices.iloc[0] if not best_prices.empty else None,
        'best_exit_price': best_prices.iloc[-1] if len(best_prices) > 1 else None,
        'worst_time': worst_time.strftime('%H:%M:%S'),
        'worst_price': worst_prices.iloc[0] if not worst_prices.empty else None,
        'worst_exit_price': worst_prices.iloc[-1] if len(worst_prices) > 1 else None,
    }
    predictions.append(result)

# Save predictions
pred_df = pd.DataFrame(predictions)
pred_df.to_csv('muhurat_entry_exit_predictions_final.csv', index=False)
print("\n📁 Predictions saved as 'muhurat_entry_exit_predictions_final.csv'")
