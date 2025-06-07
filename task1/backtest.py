import pandas as pd
import numpy as np
from fyers_apiv3 import fyersModel
from datetime import datetime, timedelta
import pytz
import time
import os

# Configuration
SYMBOL = "NSE:RELIANCE-EQ"  # Stock symbol (e.g., Reliance Industries)
ACCESS_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhdWQiOlsiZDoxIiwiZDoyIiwieDowIiwieDoxIiwieDoyIl0sImF0X2hhc2giOiJnQUFBQUFCb0hZUGRxNE9qNjBmZTEtMWJRWWVROU9CdGd6S3VtUG5BOHRXVExaOUVVbm9BX19CQmwwY1lUMUhZV1duTFZvb2dub0tyQ3YwczJyUDUwdENuanViaW1NaGgtakNKNi0tWXcxUXNJZzhUNHBOd1dJTT0iLCJkaXNwbGF5X25hbWUiOiIiLCJvbXMiOiJLMSIsImhzbV9rZXkiOiI4MTFmYjNjYWI2NzRhNDRiYjYzYzRlYTEyMjlkOTA1YmI3MWRkY2M5Zjk4ZGMzZmE4YWU2MzU4NCIsImlzRGRwaUVuYWJsZWQiOiJOIiwiaXNNdGZFbmFibGVkIjoiTiIsImZ5X2lkIjoiWU4xNDA2OCIsImFwcFR5cGUiOjEwMCwiZXhwIjoxNzQ2ODM3MDAwLCJpYXQiOjE3NDY3NjQ3NjUsImlzcyI6ImFwaS5meWVycy5pbiIsIm5iZiI6MTc0Njc2NDc2NSwic3ViIjoiYWNjZXNzX3Rva2VuIn0.NJpjdsyFBBI6OT2HC3G9p1ZIZDUmMQUPs3oTgPj_lw4"  # Replace with your Fyers API access token
APP_ID = "Y93EC2KB25-100"  # Replace with your Fyers API app ID
DATA_FOLDER = "historical_data"  # Folder to store data
TRADEBOOK_CSV = "tradebook.csv"  # Tradebook file
IST = pytz.timezone('Asia/Kolkata')
START_DATE = (datetime.now(IST) - timedelta(days=5*365)).strftime('%Y-%m-%d')  # 5 years ago
END_DATE = datetime.now(IST).strftime('%Y-%m-%d')  # Today
CHUNK_DAYS = 100  # Fyers API limit per request
STOP_LOSS_PCT = 0.02  # 2% stop loss
TAKE_PROFIT_PCT = 0.03  # 3% take profit

# Initialize Fyers API client
fyers = fyersModel.FyersModel(client_id=APP_ID, token=ACCESS_TOKEN, is_async=False)

def fetch_historical_data(symbol, start_date, end_date):
    """Fetch 1-minute historical data in 100-day chunks and save to CSV."""
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
    
    csv_file = os.path.join(DATA_FOLDER, f"{symbol.replace(':', '_')}.csv")
    if os.path.exists(csv_file):
        print(f"Loading cached data for {symbol} from {csv_file}")
        try:
            df = pd.read_csv(csv_file)
            # Convert to datetime without timezone first
            df['Datetime'] = pd.to_datetime(df['Datetime'])
            # Then localize to IST
            df['Datetime'] = df['Datetime'].dt.tz_localize(IST)
            return df
        except Exception as e:
            print(f"Error reading cached data: {e}")
            # If there's an error reading the cached file, we'll fetch new data
    
    all_data = []
    current_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    while current_date <= end_date:
        chunk_end = min(current_date + timedelta(days=CHUNK_DAYS-1), end_date)
        print(f"Fetching data for {symbol} from {current_date} to {chunk_end}")
        
        try:
            data = {
                "symbol": symbol,
                "resolution": "1",  # 1-minute data
                "date_format": "1",
                "range_from": current_date.strftime('%Y-%m-%d'),
                "range_to": chunk_end.strftime('%Y-%m-%d'),
                "cont_flag": "1"
            }
            response = fyers.history(data)
            
            if response.get('code') != 200 or not response.get('candles'):
                print(f"No data for {current_date} to {chunk_end}")
                current_date += timedelta(days=CHUNK_DAYS)
                continue
            
            candles = response['candles']
            df_chunk = pd.DataFrame(candles, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            
            # Convert timestamp to datetime and handle timezone properly
            df_chunk['Datetime'] = pd.to_datetime(df_chunk['Timestamp'], unit='s')
            df_chunk['Datetime'] = df_chunk['Datetime'].dt.tz_localize('UTC').dt.tz_convert(IST)
            
            # Select and reorder columns
            df_chunk = df_chunk[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]
            all_data.append(df_chunk)
            
        except Exception as e:
            print(f"Error fetching data for {current_date} to {chunk_end}: {e}")
            time.sleep(1)  # Avoid API rate limits
        
        current_date += timedelta(days=CHUNK_DAYS)
    
    if not all_data:
        raise ValueError(f"No data retrieved for {symbol}")
    
    # Combine all chunks
    df = pd.concat(all_data, ignore_index=True)
    
    # Sort by datetime and remove duplicates
    df = df.sort_values('Datetime').reset_index(drop=True)
    df = df.drop_duplicates(subset=['Datetime'], keep='last')
    
    # Save to CSV
    try:
        # Convert to string format before saving to avoid timezone issues
        df_to_save = df.copy()
        df_to_save['Datetime'] = df_to_save['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S%z')
        df_to_save.to_csv(csv_file, index=False)
        print(f"Saved data to {csv_file}")
    except Exception as e:
        print(f"Error saving data to CSV: {e}")
    
    return df

# def initialize_emas():
#     """Initialize EMA state for live-like processing."""
#     return {
#         'ema_20': {'value': None, 'alpha': 2 / (20 + 1), 'prev': None},
#         'ema_50': {'value': None, 'alpha': 2 / (50 + 1), 'prev': None},
#         'ema_100': {'value': None, 'alpha': 2 / (100 + 1), 'prev': None}
#     }

# def update_ema(ema_state, close):
#     """Update EMA values for a new close price."""
#     for key in ['ema_20', 'ema_50', 'ema_100']:
#         if ema_state[key]['value'] is None:
#             ema_state[key]['value'] = close
#         else:
#             ema_state[key]['value'] = (close * ema_state[key]['alpha']) + (ema_state[key]['value'] * (1 - ema_state[key]['alpha']))
#     return ema_state

def backtest_strategy(df):
    """Backtest the EMA strategy minute-by-minute, simulating live market."""
    tradebook = []
    open_position = None

    # Filter data for market hours (9:15 AM to 3:30 PM IST)
    df['Time'] = df['Datetime'].dt.time
    market_open = datetime.strptime("09:15:00", "%H:%M:%S").time()
    market_close = datetime.strptime("15:30:00", "%H:%M:%S").time()
    df = df[df['Time'].between(market_open, market_close)]
    df = df.drop(columns=['Time'])
    df = df.reset_index(drop=True)

    df['ema_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['ema_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['ema_100'] = df['Close'].ewm(span=100, adjust=False).mean()
    df.to_csv('df.csv', index=False)
    
    for i in range(len(df)):
        row = df.iloc[i]
        close = row['Close']
        
        # Skip if EMAs are not yet initialized (need ~100 periods for stability)
        if i < 100:
            continue
        
        ema_20 = df['ema_20'].iloc[i]
        ema_50 = df['ema_50'].iloc[i]
        ema_100 = df['ema_100'].iloc[i]
        
        # Calculate signals
        buy_signal = (ema_20 > ema_50 > ema_100 and close > ema_50)
        sell_signal = (ema_100 > ema_50 > ema_20 and 
                       ema_20 >= 0.975 * ema_100 and 
                       (close < ema_100 or close < ema_50 or close < ema_20))
        
        # Check open position for stop loss or take profit
        if open_position:
            entry_price = open_position['Entry Price']
            stop_loss = entry_price * (1 - STOP_LOSS_PCT)
            take_profit = entry_price * (1 + TAKE_PROFIT_PCT)
            
            if row['Low'] <= stop_loss:
                trade = {
                    'Symbol': SYMBOL,
                    'Entry Time': open_position['Entry Time'],
                    'Entry Price': entry_price,
                    'Exit Time': row['Datetime'],
                    'Exit Price': stop_loss,
                    'PnL': stop_loss - entry_price,
                    'Exit Reason': 'Stop Loss'
                }
                tradebook.append(trade)
                open_position = None
            elif row['High'] >= take_profit:
                trade = {
                    'Symbol': SYMBOL,
                    'Entry Time': open_position['Entry Time'],
                    'Entry Price': entry_price,
                    'Exit Time': row['Datetime'],
                    'Exit Price': take_profit,
                    'PnL': take_profit - entry_price,
                    'Exit Reason': 'Take Profit'
                }
                tradebook.append(trade)
                open_position = None
            elif sell_signal:
                trade = {
                    'Symbol': SYMBOL,
                    'Entry Time': open_position['Entry Time'],
                    'Entry Price': entry_price,
                    'Exit Time': row['Datetime'],
                    'Exit Price': row['Close'],
                    'PnL': row['Close'] - entry_price,
                    'Exit Reason': 'Sell Signal'
                }
                tradebook.append(trade)
                open_position = None
        
        # Enter new position if no open position and buy signal
        if not open_position and buy_signal:
            open_position = {
                'Entry Time': row['Datetime'],
                'Entry Price': row['Close']
            }
    
    # Convert tradebook to DataFrame
    tradebook_df = pd.DataFrame(tradebook)
    if not tradebook_df.empty:
        tradebook_df.to_csv(TRADEBOOK_CSV, index=False)
        print(f"Saved tradebook to {TRADEBOOK_CSV}")
    return tradebook_df

def calculate_performance(tradebook_df):
    """Calculate performance metrics."""
    if tradebook_df.empty:
        return {
            'Total Trades': 0,
            'Total PnL': 0,
            'Win Rate': 0,
            'Average PnL': 0,
            'Wins': 0,
            'Losses': 0
        }
    
    total_trades = len(tradebook_df)
    total_pnl = tradebook_df['PnL'].sum()
    wins = len(tradebook_df[tradebook_df['PnL'] > 0])
    losses = len(tradebook_df[tradebook_df['PnL'] <= 0])
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
    average_pnl = tradebook_df['PnL'].mean() if total_trades > 0 else 0
    
    return {
        'Total Trades': total_trades,
        'Total PnL': total_pnl,
        'Win Rate': win_rate,
        'Average PnL': average_pnl,
        'Wins': wins,
        'Losses': losses
    }

def main():
    """Main function to run the backtest."""
    print(f"Backtesting strategy for {SYMBOL} from {START_DATE} to {END_DATE}")
    
    # Fetch and save historical data
    df = fetch_historical_data(SYMBOL, START_DATE, END_DATE)
    
    # Run backtest
    tradebook_df = backtest_strategy(df)
    
    # Calculate performance
    performance = calculate_performance(tradebook_df)
    
    # Display results
    print("\nTradebook:")
    if not tradebook_df.empty:
        print(tradebook_df[['Symbol', 'Entry Time', 'Entry Price', 'Exit Time', 'Exit Price', 'PnL', 'Exit Reason']])
    else:
        print("No trades executed.")
    
    print("\nPerformance Metrics:")
    print(f"Total Trades: {performance['Total Trades']}")
    print(f"Total PnL: {performance['Total PnL']:.2f}")
    print(f"Win Rate: {performance['Win Rate']:.2f}%")
    print(f"Average PnL per Trade: {performance['Average PnL']:.2f}")
    print(f"Wins: {performance['Wins']}")
    print(f"Losses: {performance['Losses']}")

def load_tradebook():
    """Load existing tradebook from CSV file and ensure timestamps are in IST."""
    if not os.path.exists(TRADEBOOK_CSV):
        return pd.DataFrame(columns=['Symbol', 'Entry Time', 'Entry Price', 'Exit Time', 'Exit Price', 'PnL'])
    try:
        df = pd.read_csv(TRADEBOOK_CSV)
        # Convert to datetime without timezone first
        df['Entry Time'] = pd.to_datetime(df['Entry Time'])
        df['Exit Time'] = pd.to_datetime(df['Exit Time'])
        # Then localize to IST
        df['Entry Time'] = df['Entry Time'].dt.tz_localize(IST)
        df['Exit Time'] = df['Exit Time'].dt.tz_localize(IST)
        return df
    except Exception as e:
        print(f"Error reading tradebook CSV: {e}")
        return pd.DataFrame(columns=['Symbol', 'Entry Time', 'Entry Price', 'Exit Time', 'Exit Price', 'PnL'])

if __name__ == "__main__":
    main()