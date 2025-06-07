import pandas as pd
import time
from datetime import datetime, timedelta
from fyers_apiv3 import fyersModel
import os
import pytz  # Add pytz for timezone handling

# === Configuration ===
CLIENT_ID = "Y93EC2KB25-100"  # e.g., KQ8GN33BLF-100
ACCESS_TOKEN_FILE = "access_token.txt"
INPUT_CSV = "lists.csv"
OUTPUT_FOLDER = "output_data"
LOOKBACK_DAYS = 7
TIMEFRAME = "15"  # 15 minutes
IST = pytz.timezone('Asia/Kolkata')  # Define IST timezone

# Create output folder
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Read access token
try:
    ACCESS_TOKEN = open(ACCESS_TOKEN_FILE, 'r').read().strip()
except FileNotFoundError:
    raise Exception(f"Access token file '{ACCESS_TOKEN_FILE}' not found.")

# Initialize Fyers instance
fyers = fyersModel.FyersModel(client_id=CLIENT_ID, token=ACCESS_TOKEN, log_path=".")

def get_from_to_dates():
    end_date = datetime.now(IST)
    # Adjust end_date to last trading day if market is closed
    if end_date.weekday() >= 5:  # Saturday (5) or Sunday (6)
        days_to_subtract = end_date.weekday() - 4  # Go back to Friday
        end_date = end_date - timedelta(days=days_to_subtract)
    elif end_date.time() > datetime.strptime("15:30:00", "%H:%M:%S").time():
        # If after market close, set to market close time
        end_date = end_date.replace(hour=15, minute=30, second=0, microsecond=0)
    start_date = end_date - timedelta(days=LOOKBACK_DAYS)
    return (
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d")
    )

def fetch_and_save_data(symbol):
    try:
        print(f"[{datetime.now(IST).strftime('%H:%M:%S')}] Fetching data for {symbol}...")

        # Get the date range for fetching data
        from_date, to_date = get_from_to_dates()

        # Fetch data from Fyers API
        data = {
            "symbol": symbol,
            "resolution": TIMEFRAME,
            "date_format": "1",
            "range_from": from_date,
            "range_to": to_date,
            "cont_flag": "1"
        }

        response = fyers.history(data)

        if response.get("code") != 200:
            print(f"Error fetching data for {symbol}: {response.get('message', 'Unknown error')}")
            return

        df = pd.DataFrame(response["candles"], columns=["Datetime", "Open", "High", "Low", "Close", "Volume"])

        if df.empty:
            print(f"No data returned for {symbol}.")
            return

        # Convert timestamp to datetime and localize to IST
        df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')
        df['Datetime'] = df['Datetime'].dt.tz_localize('UTC').dt.tz_convert(IST)

        # Filter for market hours (9:15 AM to 3:30 PM IST)
        df['Time'] = df['Datetime'].dt.time
        market_open = datetime.strptime("09:15:00", "%H:%M:%S").time()
        market_close = datetime.strptime("15:30:00", "%H:%M:%S").time()
        df = df[df['Time'].between(market_open, market_close)]
        df = df.drop(columns=['Time'])

        # Add symbol column
        df['Symbol'] = symbol

        # Save to CSV
        safe_symbol = symbol.replace(":", "_").replace("-", "_")
        filename = f"{safe_symbol}.csv"
        filepath = os.path.join(OUTPUT_FOLDER, filename)

        # If file exists, load existing data and append new data
        if os.path.exists(filepath):
            try:
                existing_df = pd.read_csv(filepath)
                existing_df['Datetime'] = pd.to_datetime(existing_df['Datetime'], errors='coerce', utc=True).dt.tz_convert(IST)
                # Combine existing and new data
                combined_df = pd.concat([existing_df, df], ignore_index=True)
                # Sort by datetime and remove duplicates
                combined_df = combined_df.sort_values('Datetime').drop_duplicates(subset=['Datetime'], keep='last')
                df = combined_df
            except Exception as e:
                print(f"Error combining data for {symbol}: {e}")

        # Save updated DataFrame to CSV with timezone
        df_to_save = df.copy()
        df_to_save['Datetime'] = df_to_save['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S%z')
        df_to_save.to_csv(filepath, index=False)
        print(f"Saved {len(df)} rows for ({symbol}) to {filename}")

    except Exception as e:
        print(f"Error processing {symbol}: {e}")
        
def main():
    try:
        symbols_df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"Input CSV file '{INPUT_CSV}' not found.")
        return

    for _, row in symbols_df.iterrows():
        symbol = row['symbol']
        fetch_and_save_data(symbol)
        time.sleep(1)  # Prevent hitting rate limits

    print("✅ Data fetching completed successfully.")

if __name__ == "__main__":
    main()