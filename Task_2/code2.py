import pandas as pd
import numpy as np
from datetime import datetime, time

# Load data
df = pd.read_csv("C:/Users/gauri/Documents/git_clones/python_codes/BATS_NVDA, 15.csv")
df.set_index('t', inplace=True)
df.index = pd.to_datetime(df.index)
df.index = df.index.map(lambda x: x.tz_localize(None) if x.tzinfo else x)
print(df.head())

capital = 10000
max_trades = 2
tradecount_L = 0
tradecount_S = 0

in_position = False
position_type = None
entry_price = None
entry_time = None
trades = []

# PARAMETERS
LEPctTrailingPct = 0.05
LEProfitSlopemultiple = 3
SEPctTrailingPct = 0.05
SEProfitSlopemultiple = 3
LEPricePerc = 0.01
SEPricePerc = 0.01
LEExitMA_period = 12
SEExitMA_period = 12

# Calculate Indicators
def calculate_rsi(df, period=14):
    delta = df['c'].diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_bollinger_bands(df, period=20, num_std_dev=2):
    sma = df['c'].rolling(period).mean()
    std = df['c'].rolling(period).std()
    upper = sma + num_std_dev * std
    lower = sma - num_std_dev * std
    return sma, upper, lower

df['rsi'] = calculate_rsi(df)
df['sma'], df['upper_band'], df['lower_band'] = calculate_bollinger_bands(df)
df['LEExitMA'] = df['c'].rolling(LEExitMA_period).mean()
df['SEExitMA'] = df['c'].rolling(SEExitMA_period).mean()

df.index = df.index.tz_localize(None) 
df.to_excel("UPdated_BATS_NVDA_15min.xlsx", index=True)

# Helper: Check time in range
def is_market_time(t):
    return time(10,0) <= t.time() <= time(15,0)

def is_eod_exit_time(t):
    return t.time() >= time(15,30)

current_date = df.index[0].date()

# Backtest Loop
for idx in range(1, len(df)):
    row = df.iloc[idx]
    now_time = row.name
    price = row['c']

    row_date = now_time.date()
    if row_date != current_date:
        tradecount_L = 0
        tradecount_S = 0
        current_date = row_date

    if is_eod_exit_time(now_time):
        if in_position:
            pnl = (price - entry_price) if position_type == 'Long' else (entry_price - price)
            capital += pnl
            trades.append({
                'entry_time': entry_time, 'exit_time': now_time,
                'position': position_type, 'entry_price': entry_price,
                'exit_price': price, 'exit_reason': 'EOD Exit',
                'pnl': pnl
            })
            in_position = False
            print(f"[EOD EXIT] {position_type} at {price} | PnL: {pnl}")
        tradecount_L, tradecount_S = 0, 0
        continue

    if not in_position and is_market_time(now_time):
        if row['rsi'] < 40 and price < row['lower_band'] and tradecount_L < max_trades:
            in_position = True
            position_type = 'Long'
            entry_price = price
            entry_time = now_time
            tradecount_L += 1
            highest_price = price
            print(f"Long Entry at {price} on {now_time}")
        
        elif row['rsi'] > 60 and price > row['upper_band'] and tradecount_S < max_trades:
            in_position = True
            position_type = 'Short'
            entry_price = price
            entry_time = now_time
            tradecount_S += 1
            lowest_price = price
            print(f"Short Entry at {price} on {now_time}")

    elif in_position:  
        if position_type == 'Long':
            highest_price = max(highest_price, price)
            trail_stop = highest_price * (1 - LEPctTrailingPct)
            ma_exit = row['LEExitMA']
            slope = (row['LEExitMA'] - df['LEExitMA'].iloc[idx-1]) if idx > 1 else 0
            dyn_exit = ma_exit - slope * LEProfitSlopemultiple
            min_exit = entry_price * (1 + LEPricePerc)

            if price <= trail_stop or price <= dyn_exit or price >= min_exit:
                if now_time != entry_time:
                    pnl = price - entry_price
                    capital += pnl
                    trades.append({
                        'entry_time': entry_time, 'exit_time': now_time,
                        'position': position_type, 'entry_price': entry_price,
                        'exit_price': price, 'exit_reason': 'Trailing/MA Exit',
                        'pnl': pnl
                    })
                    in_position = False
                    print(f"Long Exit at {price} | PnL: {pnl}")
        
        elif position_type == 'Short':
            lowest_price = min(lowest_price, price)
            trail_stop = lowest_price * (1 + SEPctTrailingPct)
            ma_exit = row['SEExitMA']
            slope = (row['SEExitMA'] - df['SEExitMA'].iloc[idx-1]) if idx > 1 else 0
            dyn_exit = ma_exit + slope * SEProfitSlopemultiple
            min_exit = entry_price * (1 - SEPricePerc)

            if price >= trail_stop or price >= dyn_exit or price <= min_exit:
                if now_time != entry_time:
                    pnl = entry_price - price
                    capital += pnl
                    trades.append({
                        'entry_time': entry_time, 'exit_time': now_time,
                        'position': position_type, 'entry_price': entry_price,
                        'exit_price': price, 'exit_reason': 'Trailing/MA Exit',
                        'pnl': pnl
                    })
                    in_position = False
                    print(f"Short Exit at {price} | PnL: {pnl}")

# Convert trades to DataFrame
trades_df = pd.DataFrame(trades)

trades_df["rsi"] = df.loc[trades_df['entry_time'], 'rsi'].values
trades_df["sma"] = df.loc[trades_df['entry_time'], 'sma'].values
trades_df["upper_band"] = df.loc[trades_df['entry_time'], 'upper_band'].values
trades_df["lower_band"] = df.loc[trades_df['entry_time'], 'lower_band'].values
trades_df["LEExitMA"] = df.loc[trades_df['entry_time'], 'LEExitMA'].values
trades_df["SEExitMA"] = df.loc[trades_df['entry_time'], 'SEExitMA'].values

trades_df['entry_time'] = trades_df['entry_time'].dt.tz_localize(None)
trades_df['exit_time'] = trades_df['exit_time'].dt.tz_localize(None)

# Add 'win' column: 1 if pnl > 0 else 0
trades_df['win'] = trades_df['pnl'].apply(lambda x: 1 if x > 0 else 0)

# Export to Excel
trades_df.to_excel("trades_summary.xlsx", index=False)

# Calculate and print average pnl
if not trades_df.empty:
    avg_pnl = trades_df['pnl'].mean()
    print("\nAverage PnL per trade:", avg_pnl)
else:
    print("\nNo trades executed.")

print("\nFinal Capital:", capital)
print("\nTotal Trades Executed:", len(trades_df))
print("\n Average win rate:", trades_df['win'].mean() * 100, "%") 