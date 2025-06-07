import pandas as pd
import streamlit as st
import os
import time
from datetime import datetime, timedelta
import pytz
from fyers_apiv3 import fyersModel

# Configuration
INPUT_CSV = "lists.csv"  # CSV file containing Fyers symbols
OUTPUT_FOLDER = "output_data"  # Folder where Fyers CSV files are saved
TRADE_LOG_CSV = "trade_log.csv"  # File to store trade log
IST = pytz.timezone('Asia/Kolkata')
POLL_INTERVAL = 10  # Seconds between CSV file checks

# Fyers API Configuration
ACCESS_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhdWQiOlsiZDoxIiwiZDoyIiwieDowIiwieDoxIiwieDoyIl0sImF0X2hhc2giOiJnQUFBQUFCb0hlNmRnSy0xb284NG01dm1GSE9XbTBuU19jVDh6bG1lYnphR1JUTkFXSFd3UHlxNFJzejRlbnNvRjV6Z3Y5cy1hNXptV1ZkUndfeWJRWUoxVUZCM3oxYm1qM0tUR3N5NDhnVVdMNGRwaHQtSENycz0iLCJkaXNwbGF5X25hbWUiOiIiLCJvbXMiOiJLMSIsImhzbV9rZXkiOiI4MTFmYjNjYWI2NzRhNDRiYjYzYzRlYTEyMjlkOTA1YmI3MWRkY2M5Zjk4ZGMzZmE4YWU2MzU4NCIsImlzRGRwaUVuYWJsZWQiOiJOIiwiaXNNdGZFbmFibGVkIjoiTiIsImZ5X2lkIjoiWU4xNDA2OCIsImFwcFR5cGUiOjEwMCwiZXhwIjoxNzQ2ODM3MDAwLCJpYXQiOjE3NDY3OTIwOTMsImlzcyI6ImFwaS5meWVycy5pbiIsIm5iZiI6MTc0Njc5MjA5Mywic3ViIjoiYWNjZXNzX3Rva2VuIn0.hKtc7l8SzFoZzP7f5O2vWYXMGgG-9c1yRTnyXwzATuQ"
APP_ID = 'Y93EC2KB25-100'

# Initialize Fyers API client
try:
    fyers = fyersModel.FyersModel(client_id=APP_ID, token=ACCESS_TOKEN, is_async=False)
except Exception as e:
    st.error(f"Error initializing Fyers API client: {e}")
    fyers = None

def get_ohlc_file(symbol):
    """Get the OHLC CSV file path for a symbol."""
    safe_symbol = symbol.replace(":", "_").replace("-", "_")
    return os.path.join(OUTPUT_FOLDER, f"{safe_symbol}.csv")

def get_signal_file(symbol):
    """Get the signal CSV file path for a symbol."""
    safe_symbol = symbol.replace(":", "_").replace("-", "_")
    return os.path.join(OUTPUT_FOLDER, f"{safe_symbol}_signals.csv")

def load_ohlc_data(symbol):
    """Load OHLC data from CSV file and ensure Datetime is in IST."""
    file_path = get_ohlc_file(symbol)
    if not os.path.exists(file_path):
        return None
    try:
        df = pd.read_csv(file_path)
        # Parse Datetime with timezone information
        df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce', utc=True).dt.tz_convert(IST)
        return df
    except Exception as e:
        print(f"Error reading OHLC CSV for {symbol}: {e}")
        return None

def load_signal_data(symbol):
    """Load existing signal data from CSV file and ensure Datetime is in IST."""
    file_path = get_signal_file(symbol)
    if not os.path.exists(file_path):
        return pd.DataFrame(columns=['Datetime', 'Symbol', 'Close', 'ema_20', 'ema_50', 'ema_100', 'signal'])
    try:
        df = pd.read_csv(file_path)
        # Parse Datetime with timezone information
        df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce', utc=True).dt.tz_convert(IST)
        return df
    except Exception as e:
        print(f"Error reading signal CSV for {symbol}: {e}")
        return pd.DataFrame(columns=['Datetime', 'Symbol', 'Close', 'ema_20', 'ema_50', 'ema_100', 'signal'])

def load_trade_log():
    """Load existing trade log from CSV file."""
    if not os.path.exists(TRADE_LOG_CSV):
        return pd.DataFrame(columns=['Timestamp', 'Symbol', 'Action', 'Price', 'Reason'])
    try:
        df = pd.read_csv(TRADE_LOG_CSV)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce', utc=True).dt.tz_convert(IST)
        return df
    except Exception as e:
        print(f"Error reading trade log CSV: {e}")
        return pd.DataFrame(columns=['Timestamp', 'Symbol', 'Action', 'Price', 'Reason'])

def save_trade_log(trade_log):
    """Save trade log to CSV file."""
    try:
        trade_log.to_csv(TRADE_LOG_CSV, index=False)
    except Exception as e:
        print(f"Error saving trade log: {e}")

def calculate_ema_and_signals(symbol, df):
    """Calculate EMAs and generate buy/sell signals."""
    if df is None or len(df) < 100:
        return None
    
    df['ema_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['ema_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['ema_100'] = df['Close'].ewm(span=100, adjust=False).mean()
    
    latest = df.iloc[-1]
    timestamp = latest['Datetime']
    
    # Check if timestamp is within market hours (9:15 AM to 3:30 PM IST)
    time_of_day = timestamp.time()
    market_open = datetime.strptime("09:15:00", "%H:%M:%S").time()
    market_close = datetime.strptime("15:30:00", "%H:%M:%S").time()
    if not (market_open <= time_of_day <= market_close):
        print(f"Skipping signal for {symbol} at {timestamp} - outside market hours")
        return None
    
    close = latest['Close']
    ema_20 = latest['ema_20']
    ema_50 = latest['ema_50']
    ema_100 = latest['ema_100']
    time_str = latest['Datetime']
    
    # Buy condition: 20 EMA > 50 EMA > 100 EMA and Close > 50 EMA
    if ema_20 > ema_50 > ema_100 and close > ema_50:
        signal = 'Buy'
    # Sell condition: 100 EMA > 50 EMA > 20 EMA and 20 EMA >= 0.975 * 100 EMA and (Close < 100 EMA or Close < 50 EMA or Close < 20 EMA)
    elif (ema_100 > ema_50 > ema_20 and ema_20 >= 0.975 * ema_100 and 
          (close < ema_100 or close < ema_50 or close < ema_20)):
        signal = 'Sell'
    else:
        signal = 'None'
    
    return {
        'Datetime': time_str,
        'Symbol': symbol,
        'Close': close,
        'ema_20': ema_20,
        'ema_50': ema_50,
        'ema_100': ema_100,
        'signal': signal
    }

def save_signal_to_csv(signal):
    """Save a signal to the symbol's signal CSV file."""
    if signal['signal'] == 'None':
        return
    
    file_path = get_signal_file(signal['Symbol'])
    signal_df = pd.DataFrame([signal])
    
    try:
        existing_df = load_signal_data(signal['Symbol'])
        combined_df = pd.concat([existing_df, signal_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=['Datetime', 'Symbol'], keep='last')
        combined_df.to_csv(file_path, index=False)
    except Exception as e:
        print(f"Error saving signal to CSV for {signal['Symbol']}: {e}")

def update_trade_log(signal, trade_log, open_positions):
    """Update the trade log based on a new signal."""
    symbol = signal['Symbol']
    signal_type = signal['signal']
    timestamp = signal['Datetime']
    
    # Check if timestamp is within market hours
    time_of_day = timestamp.time()
    market_open = datetime.strptime("09:15:00", "%H:%M:%S").time()
    market_close = datetime.strptime("15:30:00", "%H:%M:%S").time()
    if not (market_open <= time_of_day <= market_close):
        print(f"Skipping trade log update for {symbol} at {timestamp} - outside market hours")
        return trade_log, open_positions
    
    price = signal['Close']
    
    # Check if we have an open position for this symbol
    if symbol in open_positions:
        if signal_type == 'Sell':
            # Close the position
            entry_data = open_positions.pop(symbol)
            entry_trade = {
                'Timestamp': entry_data['Entry Time'],
                'Symbol': symbol,
                'Action': 'BUY',
                'Price': entry_data['Entry Price'],
                'Reason': 'EMA Strategy Buy Signal'
            }
            exit_trade = {
                'Timestamp': timestamp,
                'Symbol': symbol,
                'Action': 'SELL',
                'Price': price,
                'Reason': 'EMA Strategy Sell Signal'
            }
            trade_log = pd.concat([trade_log, pd.DataFrame([entry_trade, exit_trade])], ignore_index=True)
            save_trade_log(trade_log)
    else:
        if signal_type == 'Buy':
            # Open a new position
            open_positions[symbol] = {
                'Entry Time': timestamp,
                'Entry Price': price
            }
    
    return trade_log, open_positions

def streamlit_app():
    """Streamlit frontend to display buy/sell signals and trade log."""
    st.set_page_config(page_title="Fyers Trading Signals", layout="wide")
    st.title("📈 Fyers Trading Signals")
    st.markdown("""
        Real-time *Buy* and *Sell* signals for stocks based on EMA strategy.
        - *Buy*: 20 EMA > 50 EMA > 100 EMA and Close > 50 EMA
        - *Sell*: 100 EMA > 50 EMA > 20 EMA and 20 EMA >= 0.975 * 100 EMA and (Close < 100 EMA or Close < 50 EMA or Close < 20 EMA)
        - Only actionable signals (Buy/Sell) are displayed, sorted by time (newest first).
        - Trade log tracks trades with no repurchase before selling.
    """)
    
    try:
        symbols_df = pd.read_csv(INPUT_CSV)
        SYMBOLS = symbols_df['symbol'].tolist()
    except FileNotFoundError:
        st.error(f"Input CSV file '{INPUT_CSV}' not found.")
        return
    
    signal_placeholder = st.empty()
    trade_log_placeholder = st.empty()
    last_updated = st.empty()
    last_modified = {symbol: 0 for symbol in SYMBOLS}
    open_positions = {}
    trade_log = load_trade_log()
    
    while True:
        all_signals = []
        current_time = datetime.now(IST)
        for symbol in SYMBOLS:
            file_path = get_ohlc_file(symbol)
            if os.path.exists(file_path):
                mtime = os.path.getmtime(file_path)
                if mtime > last_modified[symbol]:
                    df = load_ohlc_data(symbol)
                    if df is not None:
                        # Only process recent data (within the last 15 minutes)
                        recent_data = df[df['Datetime'] >= current_time - timedelta(minutes=15)]
                        if not recent_data.empty:
                            signal = calculate_ema_and_signals(symbol, recent_data)
                            if signal and signal['signal'] != 'None':
                                save_signal_to_csv(signal)
                                trade_log, open_positions = update_trade_log(signal, trade_log, open_positions)
                    last_modified[symbol] = mtime
                
                signal_df = load_signal_data(symbol)
                if not signal_df.empty:
                    signal_df = signal_df[signal_df['signal'].isin(['Buy', 'Sell'])]
                    all_signals.append(signal_df)
        
        if all_signals:
            combined_df = pd.concat(all_signals, ignore_index=True)
            combined_df = combined_df.sort_values(by='Datetime', ascending=False)
            display_df = combined_df[['Datetime', 'Symbol', 'Close', 'ema_20', 'ema_50', 'ema_100', 'signal']].copy()
            display_df.columns = ['Time (IST)', 'Symbol', 'Close', '20 EMA', '50 EMA', '100 EMA', 'Signal']
            display_df['Time (IST)'] = display_df['Time (IST)'].dt.strftime('%Y-%m-%d %H:%M:%S')
            display_df['Close'] = display_df['Close'].apply(lambda x: f"{x:.2f}")
            display_df['20 EMA'] = display_df['20 EMA'].apply(lambda x: f"{x:.2f}")
            display_df['50 EMA'] = display_df['50 EMA'].apply(lambda x: f"{x:.2f}")
            display_df['100 EMA'] = display_df['100 EMA'].apply(lambda x: f"{x:.2f}")
            
            def style_signal(val):
                color = 'green' if val == 'Buy' else 'red' if val == 'Sell' else 'black'
                return f'color: {color}; font-weight: bold;'
            
            styled_df = display_df.style.applymap(style_signal, subset=['Signal'])
            
            with signal_placeholder.container():
                st.subheader("Buy/Sell Signals")
                st.dataframe(styled_df, use_container_width=True, height=400)
        else:
            with signal_placeholder.container():
                st.info("No Buy or Sell signals available yet. Waiting for data...")
        
        if not trade_log.empty:
            display_trade_log = trade_log.copy()
            display_trade_log['Timestamp'] = display_trade_log['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            display_trade_log['Price'] = display_trade_log['Price'].apply(lambda x: f"{x:.2f}")
            display_trade_log = display_trade_log.sort_values('Timestamp', ascending=False)
            
            with trade_log_placeholder.container():
                st.subheader("Trade Log")
                st.dataframe(display_trade_log, use_container_width=True, height=400)
        else:
            with trade_log_placeholder.container():
                st.info("No trades recorded yet.")
        
        last_updated.text(f"Last Updated: {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')}")
        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    streamlit_app()