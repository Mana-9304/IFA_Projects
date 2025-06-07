import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import pytz

IST = pytz.timezone('Asia/Kolkata')

# Configuration
DATA_FOLDER = "btcusdt_1m_2020_2025.csv"
TRADEBOOK_CSV = "tradebook_btc.csv"

START_DATE = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
END_DATE = datetime.now().strftime('%Y-%m-%d')

# Markets to test
MARKETS = ["BINANCE:BTCUSDT"]

# Strategy variations from the image
VARIATIONS = [
    {"entry": "price_above_sma10", "exit": "fixed_target", "tp_pct": 10, "sl_pct": 4},
    {"entry": "price_above_sma10", "exit": "fixed_target", "tp_pct": 9, "sl_pct": 3},
    {"entry": "sma5_above_sma10", "exit": "fixed_target", "tp_pct": 7.5, "sl_pct": 3},
    {"entry": "sma5_above_sma10", "exit": "fixed_target", "tp_pct": 10, "sl_pct": 4},
]

def fetch_historical_data(symbol):
    csv_file = DATA_FOLDER
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"File {csv_file} not found.")
    print(f"Loading data for {symbol} from {csv_file}")
    try:
        df = pd.read_csv(csv_file)
        if "Open time" in df.columns:
            df.rename(columns={
                "Open time": "Datetime",
                "Open": "Open",
                "High": "High",
                "Low": "Low",
                "Close": "Close",
                "Volume": "Volume"
            }, inplace=True)

        if df["Datetime"].dtype == "int64" or df["Datetime"].dtype == "float64":
            df["Datetime"] = pd.to_datetime(df["Datetime"], unit="ms")
        else:
            df["Datetime"] = pd.to_datetime(df["Datetime"])

        df["Datetime"] = df["Datetime"].dt.tz_localize("UTC").dt.tz_convert(IST)

        return df[["Datetime", "Open", "High", "Low", "Close", "Volume"]]
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")

def calculate_sma(df, window=10):
    df["sma_10"] = df["Close"].rolling(window=window).mean()
    return df

def calculate_sma5(df, window=5):
    df["sma_5"] = df["Close"].rolling(window=window).mean()
    return df

def backtest_strategy(df, variation):
    tradebook = []
    open_position = None
    df = calculate_sma(df)
    df = calculate_sma5(df)

    for row in df.itertuples(index=False):
        close = row.Close
        sma_10 = row.sma_10
        sma_5 = row.sma_5

        # Entry conditions
        price_above_sma10 = close > sma_10
        sma5_above_sma10 = sma_5 > sma_10

        if open_position:
            entry_price = open_position["Entry Price"]
            sl_price = entry_price * (1 - variation["sl_pct"] / 100)
            tp_price = entry_price * (1 + variation["tp_pct"] / 100)

            # Check for stop loss or take profit
            if row.Low <= sl_price:
                tradebook.append({
                    "Entry Time": open_position["Entry Time"],
                    "Entry Price": entry_price,
                    "Exit Time": row.Datetime,
                    "Exit Price": sl_price,
                    "PnL": sl_price - entry_price,
                    "Exit Reason": "Stop Loss"
                })
                open_position = None
            elif row.High >= tp_price:
                tradebook.append({
                    "Entry Time": open_position["Entry Time"],
                    "Entry Price": entry_price,
                    "Exit Time": row.Datetime,
                    "Exit Price": tp_price,
                    "PnL": tp_price - entry_price,
                    "Exit Reason": "Take Profit"
                })
                open_position = None

        # Check for new entry
        if not open_position:
            if variation["entry"] == "price_above_sma10" and price_above_sma10:
                open_position = {
                    "Entry Time": row.Datetime,
                    "Entry Price": close
                }
            elif variation["entry"] == "sma5_above_sma10" and sma5_above_sma10:
                open_position = {
                    "Entry Time": row.Datetime,
                    "Entry Price": close
                }

    return pd.DataFrame(tradebook)

def calculate_performance(df):
    if df.empty:
        return {"Total Trades": 0, "Total PnL": 0, "Win Rate": 0, "Avg PnL": 0}
    total_pnl = df["PnL"].sum()
    win_rate = len(df[df["PnL"] > 0]) / len(df) * 100
    avg_pnl = df["PnL"].mean()
    return {"Total Trades": len(df), "Total PnL": total_pnl, "Win Rate": win_rate, "Avg PnL": avg_pnl}

def run_all_backtests():
    slice_step = 50000
    for market in MARKETS:
        print(f"\nFetching data for {market}")
        try:
            df = fetch_historical_data(market)
        except Exception as e:
            print(f"Error loading data for {market}: {e}")
            continue

        total_rows = len(df)

        for idx, variation in enumerate(VARIATIONS):
            variation_results = []

            for start_idx in range(0, total_rows - 1000, slice_step):
                df_slice = df.iloc[start_idx:].copy()
                slice_start_time = df_slice.iloc[0]["Datetime"]

                print(f"Running variation {idx+1} for {market} from index {start_idx}")
                tradebook_df = backtest_strategy(df_slice, variation)
                perf = calculate_performance(tradebook_df)

                variation_results.append({
                    "Market": market,
                    "Entry Type": variation["entry"],
                    "Exit Type": variation["exit"],
                    "Take Profit (%)": variation["tp_pct"],
                    "Stop Loss (%)": variation["sl_pct"],
                    "Start Time": slice_start_time,
                    "Total Trades": perf["Total Trades"],
                    "Total PnL": round(perf["Total PnL"], 2),
                    "Win Rate (%)": round(perf["Win Rate"], 2),
                    "Avg PnL per Trade": round(perf["Avg PnL"], 2)
                })

            # Save to individual variation CSV
            var_df = pd.DataFrame(variation_results)
            var_df.to_csv(f"variation_{idx+1}_results.csv", index=False)
            print(f"✅ Saved variation {idx+1} results to variation_{idx+1}_results.csv")

if __name__ == "__main__":
    run_all_backtests()