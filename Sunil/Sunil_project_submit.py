# Forward testing code snippet (uncomment to use): for live trading


# import asyncio
# import websockets
# import json
# import pandas as pd
# import numpy as np
# import talib
# from datetime import datetime, timedelta, timezone
# import signal
# import sys

# # Global trackers
# last_timestamp_3m = None
# last_timestamp_30m = None
# trade_log = []
# total_trades = 0

# # Strategy parameters
# RISK_REWARD_RATIO = 2
# POSITION_SIZE = 0.01
# CAPITAL = 1000

# current_position = None  # {'side', 'entry_price', 'stop_loss', 'take_profit', 'entry_time'}

# df_columns = ['Time (IST)', 'Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'ATR']
# df_dtypes = {col: (str if col == 'Time (IST)' else float) for col in df_columns}

# df = pd.DataFrame({col: pd.Series(dtype=dt) for col, dt in df_dtypes.items()})    # 3m
# df1 = pd.DataFrame({col: pd.Series(dtype=dt) for col, dt in df_dtypes.items()})   # 30m

# def calculate_indicators(dataframe):
#     if len(dataframe) >= 20:
#         close = dataframe['Close'].astype(float).values
#         high = dataframe['High'].astype(float).values
#         low = dataframe['Low'].astype(float).values
#         dataframe['RSI'] = talib.RSI(close, timeperiod=20)
#         dataframe['ATR'] = talib.ATR(high, low, close, timeperiod=14)
#     else:
#         dataframe['RSI'] = np.nan
#         dataframe['ATR'] = np.nan
#     return dataframe

# def check_entry_signal(df, is_long=True):
#     if len(df) < 2 or pd.isna(df['RSI'].iloc[-1]) or pd.isna(df['RSI'].iloc[-2]):
#         return False
#     if is_long:
#         return df['RSI'].iloc[-1] >= 27 and df['RSI'].iloc[-1] >= df['RSI'].iloc[-2]
#     else:
#         return df['RSI'].iloc[-1] <= 73 and df['RSI'].iloc[-1] <= df['RSI'].iloc[-2]

# def check_exit_signal(price, time_str):
#     global current_position, trade_log, total_trades

#     if not current_position:
#         return

#     exit_reason = None
#     if current_position['side'] == 'long':
#         if price <= current_position['stop_loss']:
#             exit_reason = 'stop_loss'
#         elif price >= current_position['take_profit']:
#             exit_reason = 'take_profit'

#     elif current_position['side'] == 'short':
#         if price >= current_position['stop_loss']:
#             exit_reason = 'stop_loss'
#         elif price <= current_position['take_profit']:
#             exit_reason = 'take_profit'

#     if exit_reason:
#         exit_price = price
#         entry_price = current_position['entry_price']
#         pnl = (exit_price - entry_price) if current_position['side'] == 'long' else (entry_price - exit_price)
#         trade_duration = (datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S') - datetime.strptime(current_position['entry_time'], '%Y-%m-%d %H:%M:%S')).seconds
#         total_trades += 1

#         trade_log.append({
#             'Entry Time': current_position['entry_time'],
#             'Entry Price': entry_price,
#             'Stop Loss': current_position['stop_loss'],
#             'Take Profit': current_position['take_profit'],
#             'Exit Time': time_str,
#             'Exit Price': exit_price,
#             'Exit Reason': exit_reason,
#             'PnL': pnl,
#             'Trade Duration (sec)': trade_duration,
#             'Total Trade': total_trades
#         })

#         print(f"EXIT {current_position['side'].upper()} at {exit_price:.2f} | Reason: {exit_reason.upper()} | PnL: {pnl:.2f}")
#         current_position = None

# def handle_socket_message(msg, interval):
#     global last_timestamp_3m, last_timestamp_30m, df, df1, current_position

#     if msg['e'] != 'kline':
#         return

#     kline = msg['k']
#     timestamp = kline['t'] / 1000
#     utc_time = datetime.fromtimestamp(timestamp, tz=timezone.utc)
#     ist_time = utc_time + timedelta(hours=5, minutes=30)
#     formatted_time = ist_time.strftime('%Y-%m-%d %H:%M:%S')

#     row = pd.DataFrame([{
#         'Time (IST)': formatted_time,
#         'Open': float(kline['o']),
#         'High': float(kline['h']),
#         'Low': float(kline['l']),
#         'Close': float(kline['c']),
#         'Volume': float(kline['v']),
#         'RSI': np.nan,
#         'ATR': np.nan
#     }], columns=df_columns)

#     if interval == '3m':
#         if last_timestamp_3m is None or (timestamp - last_timestamp_3m) >= 180:
#             df = pd.concat([df, row], ignore_index=True)
#             df = calculate_indicators(df)
#             last_timestamp_3m = timestamp
#             print(df)  # Print the DataFrame after each update

#             price = df['Close'].iloc[-1]
#             check_exit_signal(price, formatted_time)

#             if not current_position:
#                 if check_entry_signal(df, is_long=True) and check_entry_signal(df1, is_long=True):
#                     entry_price = price
#                     atr_30m = df1['ATR'].iloc[-1]
#                     stop_loss = entry_price - atr_30m * 2
#                     take_profit = entry_price + atr_30m * 2 * RISK_REWARD_RATIO
#                     position_size = CAPITAL * POSITION_SIZE
#                     current_position = {'side': 'long', 'entry_price': entry_price, 'stop_loss': stop_loss,
#                                         'take_profit': take_profit, 'size': position_size, 'entry_time': formatted_time}
#                     print(f"ENTER LONG at {entry_price:.2f}, SL: {stop_loss:.2f}, TP: {take_profit:.2f}, Size: ${position_size:.2f}")

#                 elif check_entry_signal(df, is_long=False) and check_entry_signal(df1, is_long=False):
#                     entry_price = price
#                     atr_30m = df1['ATR'].iloc[-1]
#                     stop_loss = entry_price + atr_30m * 2
#                     take_profit = entry_price - atr_30m * 2 * RISK_REWARD_RATIO
#                     position_size = CAPITAL * POSITION_SIZE
#                     current_position = {'side': 'short', 'entry_price': entry_price, 'stop_loss': stop_loss,
#                                         'take_profit': take_profit, 'size': position_size, 'entry_time': formatted_time}
#                     print(f"ENTER SHORT at {entry_price:.2f}, SL: {stop_loss:.2f}, TP: {take_profit:.2f}, Size: ${position_size:.2f}")

#     elif interval == '30m':
#         if last_timestamp_30m is None or (timestamp - last_timestamp_30m) >= 1800:
#             df1 = pd.concat([df1, row], ignore_index=True)
#             df1 = calculate_indicators(df1)
#             last_timestamp_30m = timestamp
#             print(df1)  # Print the DataFrame after each update

# async def stream_data(symbol, interval):
#     uri = f"wss://stream.binance.com:9443/ws/{symbol}@kline_{interval}"
#     async with websockets.connect(uri) as websocket:
#         while True:
#             try:
#                 msg = await websocket.recv()
#                 msg = json.loads(msg)
#                 handle_socket_message(msg, interval)
#             except Exception as e:
#                 print(f"WebSocket error: {e}")

# def save_trade_log():
#     if trade_log:
#         df_log = pd.DataFrame(trade_log)
#         df_log.to_excel('trade_log.xlsx', index=False)
#         print('Trade log saved to trade_log.xlsx')

# def signal_handler(sig, frame):
#     print('\nTerminating program... Saving trade log.')
#     save_trade_log()
#     sys.exit(0)

# async def main():
#     signal.signal(signal.SIGINT, signal_handler)
#     signal.signal(signal.SIGTERM, signal_handler)
#     await asyncio.gather(
#         stream_data('btcusdt', '3m'),
#         stream_data('btcusdt', '30m')
#     )

# if __name__ == "__main__":
#     asyncio.run(main())





# Data downloading code snippet (uncomment to use): for backfilling historical data



# import ccxt
# import pandas as pd
# import time

# exchange = ccxt.binance()
# symbol = 'BTC/USDT'
# timeframes = ['3m', '30m']

# for tf in timeframes:
#     ohlcv = []
#     since = exchange.parse8601('2023-01-01T00:00:00Z')  # or your desired start date
#     while True:
#         data = exchange.fetch_ohlcv(symbol, timeframe=tf, since=since, limit=1000)
#         if not data:
#             break
#         ohlcv += data
#         since = data[-1][0] + 1  # avoid overlap
#         time.sleep(exchange.rateLimit / 1000)  # be kind to API

#     df = pd.DataFrame(ohlcv, columns=['Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
#     df['Time'] = pd.to_datetime(df['Time'], unit='ms')
#     df.to_csv(f'btcusdt_{tf}.csv', index=False)
#     print(f'Saved btcusdt_{tf}.csv')




# For backtesting and strategy development, you can use the following code snippet to read the saved CSV files and implement your trading strategy.


# import pandas as pd
# import numpy as np
# import talib
# from datetime import datetime, timedelta

# # Parameters
# RISK_REWARD_RATIO = 2
# POSITION_SIZE = 0.01
# CAPITAL = 1000

# current_position = None
# trade_log = []
# total_trades = 0

# # Load historical data
# df_3m = pd.read_csv("C:/Users/91815/OneDrive/Desktop/DESKTOP-D6T8ITB/NAYAN/NAYAN/Desktop/Insight_Fusion_Internship_task/Task_18_Sunil/btcusdt_3m.csv", parse_dates=['Time'])
# df_30m = pd.read_csv("C:/Users/91815/OneDrive/Desktop/DESKTOP-D6T8ITB/NAYAN/NAYAN/Desktop/Insight_Fusion_Internship_task/Task_18_Sunil/btcusdt_30m.csv", parse_dates=['Time'])

# def calculate_indicators(df):
#     close = df['Close'].astype(float).values
#     high = df['High'].astype(float).values
#     low = df['Low'].astype(float).values
#     df['RSI'] = talib.RSI(close, timeperiod=20)
#     df['ATR'] = talib.ATR(high, low, close, timeperiod=14)
#     return df

# def check_entry_signal(df, is_long=True):
#     if len(df) < 2 or pd.isna(df['RSI'].iloc[-1]) or pd.isna(df['RSI'].iloc[-2]):
#         return False
#     if is_long:
#         return df['RSI'].iloc[-1] >= 27 and df['RSI'].iloc[-1] >= df['RSI'].iloc[-2]
#     else:
#         return df['RSI'].iloc[-1] <= 73 and df['RSI'].iloc[-1] <= df['RSI'].iloc[-2]

# def check_exit_signal(price, time_str):
#     global current_position, trade_log, total_trades

#     if not current_position:
#         return

#     exit_reason = None
#     if current_position['side'] == 'long':
#         if price <= current_position['stop_loss']:
#             exit_reason = 'stop_loss'
#         elif price >= current_position['take_profit']:
#             exit_reason = 'take_profit'
#     elif current_position['side'] == 'short':
#         if price >= current_position['stop_loss']:
#             exit_reason = 'stop_loss'
#         elif price <= current_position['take_profit']:
#             exit_reason = 'take_profit'

#     if exit_reason:
#         exit_price = price
#         entry_price = current_position['entry_price']
#         pnl = (exit_price - entry_price) if current_position['side'] == 'long' else (entry_price - exit_price)
#         entry_time = datetime.strptime(current_position['entry_time'], '%Y-%m-%d %H:%M:%S')
#         exit_time = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
#         trade_duration = (exit_time - entry_time).seconds
#         total_trades += 1

#         trade_log.append({
#             'Entry Time': current_position['entry_time'],
#             'Entry Price': entry_price,
#             'Stop Loss': current_position['stop_loss'],
#             'Take Profit': current_position['take_profit'],
#             'Exit Time': time_str,
#             'Exit Price': exit_price,
#             'Exit Reason': exit_reason,
#             'PnL': pnl,
#             'Trade Duration (sec)': trade_duration,
#             'Total Trade': total_trades
#         })

#         print(f"EXIT {current_position['side'].upper()} at {exit_price:.2f} | Reason: {exit_reason.upper()} | PnL: {pnl:.2f}")
#         current_position = None

# # Pre-calculate indicators
# df_3m = calculate_indicators(df_3m)
# df_30m = calculate_indicators(df_30m)

# for idx in range(len(df_3m)):
#     row_3m = df_3m.iloc[:idx+1]
#     time_str = row_3m['Time'].iloc[-1].strftime('%Y-%m-%d %H:%M:%S')
#     price = row_3m['Close'].iloc[-1]

#     # Get matching 30m row
#     row_30m = df_30m[df_30m['Time'] <= row_3m['Time'].iloc[-1]]
#     if row_30m.empty:
#         continue
#     df1 = row_30m

#     check_exit_signal(price, time_str)

#     if not current_position:
#         if check_entry_signal(row_3m, is_long=True) and check_entry_signal(df1, is_long=True):
#             entry_price = price
#             atr_30m = df1['ATR'].iloc[-1]
#             stop_loss = entry_price - atr_30m * 2
#             take_profit = entry_price + atr_30m * 2 * RISK_REWARD_RATIO
#             position_size = CAPITAL * POSITION_SIZE
#             current_position = {'side': 'long', 'entry_price': entry_price, 'stop_loss': stop_loss,
#                                 'take_profit': take_profit, 'size': position_size, 'entry_time': time_str}
#             print(f"ENTER LONG at {entry_price:.2f}, SL: {stop_loss:.2f}, TP: {take_profit:.2f}, Size: ${position_size:.2f}")

#         elif check_entry_signal(row_3m, is_long=False) and check_entry_signal(df1, is_long=False):
#             entry_price = price
#             atr_30m = df1['ATR'].iloc[-1]
#             stop_loss = entry_price + atr_30m * 2
#             take_profit = entry_price - atr_30m * 2 * RISK_REWARD_RATIO
#             position_size = CAPITAL * POSITION_SIZE
#             current_position = {'side': 'short', 'entry_price': entry_price, 'stop_loss': stop_loss,
#                                 'take_profit': take_profit, 'size': position_size, 'entry_time': time_str}
#             print(f"ENTER SHORT at {entry_price:.2f}, SL: {stop_loss:.2f}, TP: {take_profit:.2f}, Size: ${position_size:.2f}")

# # Save trade log
# if trade_log:
#     df_log = pd.DataFrame(trade_log)
#     df_log.to_excel('trade_log.xlsx', index=False)
#     print('Trade log saved to trade_log.xlsx')





# Win calculation code snippet (uncomment to use): for trade log analysis



import pandas as pd

df = pd.read_excel('trade_log.xlsx')


df["Win"] = df["PnL"].apply(lambda x: 1 if x > 0 else 0)

df.to_excel('trade_log.xlsx', index=False)

print(df)