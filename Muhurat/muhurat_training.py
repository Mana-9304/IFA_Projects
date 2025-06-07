import pandas as pd
import numpy as np
from datetime import timedelta, datetime
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def load_and_validate_data(file_path='nifty50_muhurat_data.csv'):
    """Load and validate CSV data."""
    if not os.path.exists(file_path):
        logger.error(f"Input file {file_path} does not exist.")
        raise FileNotFoundError(f"Input file {file_path} not found")
    
    try:
        df = pd.read_csv(file_path)
        required_columns = ['timestamp', 'muhurat_date', 'close']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            raise ValueError(f"CSV missing required columns: {missing_cols}")
        
        # Convert timestamps and dates, handling errors
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
        df['muhurat_date'] = pd.to_datetime(df['muhurat_date'], format="%Y-%m-%d", errors='coerce')
        df = df.dropna(subset=['timestamp', 'muhurat_date', 'close']).sort_values('timestamp')
        df = df.set_index('muhurat_date', drop=False)  # Keep muhurat_date as column and index
        
        logger.info(f"Loaded {len(df)} valid rows from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        raise

def explore_data(df):
    """Log basic data exploration details."""
    timestamp_dates = set(df['timestamp'].dt.date)
    muhurat_dates = set(np.unique(df.index.date))  # Fixed: Use np.unique for NumPy array
    common_dates = timestamp_dates & muhurat_dates
    
    logger.info(f"Unique timestamp dates: {len(timestamp_dates)}")
    logger.info(f"Muhurat dates: {len(muhurat_dates)}")
    logger.info(f"Common dates: {len(common_dates)}")
    logger.debug(f"Row count per Muhurat date:\n{df.groupby(df.index).size()}")

def compute_intraday_return_by_time(df, muhurat_dates):
    """Compute average intraday returns by time with statistics."""
    returns_by_time = {}
    skipped_dates = 0
    single_price_times = 0
    
    for date in muhurat_dates:
        day_data = df.loc[df.index == date] if date in df.index else pd.DataFrame()
        if day_data.empty:
            logger.warning(f"No data for Muhurat date: {date.date()}")
            skipped_dates += 1
            continue
        
        # Extract time component
        day_data = day_data.assign(time=day_data['timestamp'].dt.time)
        for t in day_data['time'].unique():
            prices = day_data[day_data['time'] == t]['close']
            if len(prices) < 2:
                logger.debug(f"Skipping time {t} on {date.date()} due to insufficient data (prices: {len(prices)})")
                single_price_times += 1 if len(prices) == 1 else 0
                continue
            # Handle duplicates by taking the last price
            if prices.index.duplicated().any():
                logger.warning(f"Duplicate timestamps for time {t} on {date.date()}, using last price")
                prices = prices.groupby(prices.index).last()
            ret = round(prices.iloc[-1] - prices.iloc[0], 2)
            returns_by_time.setdefault(t, []).append(ret)
    
    if not returns_by_time:
        logger.error("No intraday returns computed")
        return None, skipped_dates, single_price_times
    
    # Compute statistics for each time
    avg_returns = {t: {'mean': np.mean(r), 'std': np.std(r), 'count': len(r)} 
                   for t, r in returns_by_time.items()}
    for t, stats in avg_returns.items():
        if stats['std'] == 0:
            logger.warning(f"Time {t} has zero variance (mean: {stats['mean']:.2f}, samples: {stats['count']})")
        logger.info(f"Time {t}: Mean return = {stats['mean']:.2f}, Std = {stats['std']:.2f}, Samples = {stats['count']}")
    
    # Sort by mean return
    sorted_avg_returns = dict(sorted(avg_returns.items(), key=lambda x: x[1]['mean'], reverse=True))
    return sorted_avg_returns, skipped_dates, single_price_times

def find_nearest_time(target_time, time_list, max_diff=timedelta(minutes=15)):
    """Find nearest time within a maximum difference."""
    if not time_list:
        return None
    try:
        # Convert target_time to datetime.time if it isn't already
        if isinstance(target_time, str):
            target_dt = pd.to_datetime(target_time, format='%H:%M:%S').time()
        else:
            target_dt = target_time
        
        available_dt = []
        for t in time_list:
            try:
                # Convert time to datetime for comparison
                t_str = str(t).split('.')[0]  # Remove microseconds if present
                t_dt = pd.to_datetime(t_str, format='%H:%M:%S').time()
                available_dt.append(t_dt)
            except (ValueError, TypeError):
                logger.debug(f"Invalid time format: {t}")
                continue
        
        if not available_dt:
            return None
        
        # Convert times to seconds since midnight for comparison
        target_secs = target_dt.hour * 3600 + target_dt.minute * 60 + target_dt.second
        differences = []
        for t in available_dt:
            t_secs = t.hour * 3600 + t.minute * 60 + t.second
            differences.append((t, abs(t_secs - target_secs)))
        
        nearest_time, diff = min(differences, key=lambda x: x[1])
        if diff > max_diff.total_seconds():
            logger.warning(f"No time within {max_diff} of {target_dt}")
            return None
        return nearest_time
    except (ValueError, TypeError) as e:
        logger.error(f"Time parsing error for {target_time}: {str(e)}")
        return None

def predict_entry_exit(test_df, muhurat_dates, best_time, worst_time):
    """Predict entry and exit prices for test Muhurat dates."""
    predictions = []
    skipped_dates = 0
    single_price_times = 0
    
    for date in muhurat_dates:
        muhurat_day_data = test_df.loc[test_df.index == date] if date in test_df.index else pd.DataFrame()
        if muhurat_day_data.empty:
            logger.warning(f"No test data for Muhurat date: {date.date()}")
            skipped_dates += 1
            continue
        
        muhurat_day_data = muhurat_day_data.assign(time=muhurat_day_data['timestamp'].dt.time)
        available_times = muhurat_day_data['time'].unique()
        
        # Handle best time
        best_time_use = best_time
        best_prices = muhurat_day_data[muhurat_day_data['time'] == best_time]['close']
        if best_prices.empty:
            best_time_use = find_nearest_time(best_time, available_times)
            if best_time_use:
                logger.debug(f"Using nearest best time {best_time_use} for {date.date()}")
                best_prices = muhurat_day_data[muhurat_day_data['time'] == best_time_use]['close']
            else:
                logger.warning(f"No suitable best time for {date.date()}")
                best_prices = pd.Series()
        if len(best_prices) == 1:
            logger.info(f"Only one price for best time {best_time_use} on {date.date()}: {best_prices.iloc[0]}")
            single_price_times += 1
        
        # Handle worst time
        worst_time_use = worst_time
        worst_prices = muhurat_day_data[muhurat_day_data['time'] == worst_time]['close']
        if worst_prices.empty:
            worst_time_use = find_nearest_time(worst_time, available_times)
            if worst_time_use:
                logger.debug(f"Using nearest worst time {worst_time_use} for {date.date()}")
                worst_prices = muhurat_day_data[muhurat_day_data['time'] == worst_time_use]['close']
            else:
                logger.warning(f"No suitable worst time for {date.date()}")
                worst_prices = pd.Series()
        if len(worst_prices) == 1:
            logger.info(f"Only one price for worst time {worst_time_use} on {date.date()}: {worst_prices.iloc[0]}")
            single_price_times += 1
        
        # Handle duplicates
        if not best_prices.empty and best_prices.index.duplicated().any():
            logger.warning(f"Duplicate timestamps for best time {best_time_use} on {date.date()}, using last price")
            best_prices = best_prices.groupby(best_prices.index).last()
        if not worst_prices.empty and worst_prices.index.duplicated().any():
            logger.warning(f"Duplicate timestamps for worst time {worst_time_use} on {date.date()}, using last price")
            worst_prices = worst_prices.groupby(worst_prices.index).last()
        
        # Create prediction record
        result = {
            'date': date.strftime('%d-%m-%Y'),
            'best_time': str(best_time_use) if best_time_use else None,
            'best_entry_price': best_prices.iloc[0] if not best_prices.empty else None,
            'best_exit_price': best_prices.iloc[-1] if len(best_prices) > 1 else None,
            'worst_time': str(worst_time_use) if worst_time_use else None,
            'worst_entry_price': worst_prices.iloc[0] if not worst_prices.empty else None,
            'worst_exit_price': best_prices.iloc[-1] if len(worst_prices) > 1 else None,
            'return_best': round(best_prices.iloc[-1] - best_prices.iloc[0], 2) if len(best_prices) > 1 else None,
            'return_worst': round(worst_prices.iloc[-1] - worst_prices.iloc[0], 2) if len(worst_prices) > 1 else None
        }
        predictions.append(result)
    
    return predictions, skipped_dates, single_price_times

def save_predictions(predictions, output_path='muhurat_entry_exit_predictions_final.csv', 
                    muhurat_dates=None, skipped_dates=0, single_price_times=0):
    """Save predictions to CSV and log summary statistics."""
    if not predictions:
        logger.error("No predictions to save")
        return
    
    if os.path.exists(output_path):
        logger.warning(f"Output file {output_path} exists. Appending timestamp to avoid overwrite.")
        base, ext = os.path.splitext(output_path)
        output_path = f"{base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"
    
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")
    
    valid_best = pred_df['return_best'].notna().sum()
    valid_worst = pred_df['return_worst'].notna().sum()
    avg_return_best = pred_df['return_best'].mean() if valid_best > 0 else None
    avg_return_worst = pred_df['return_worst'].mean() if valid_worst > 0 else None
    
    logger.info(f"Valid best time predictions: {valid_best}/{len(pred_df)}")
    logger.info(f"Valid worst time predictions: {valid_worst}/{len(pred_df)}")
    logger.info(f"Skipped test dates: {skipped_dates}/{len(muhurat_dates) if muhurat_dates is not None else 0}")
    logger.info(f"Single price times in test data: {single_price_times}")
    if avg_return_best is not None:
        logger.info(f"Average return at best time: {avg_return_best:.2f}")
    if avg_return_worst is not None:
        logger.info(f"Average return at worst time: {avg_return_worst:.2f}")

def main(
    input_path='nifty50_muhurat_data.csv',
    output_path='muhurat_entry_exit_predictions_final.csv',
    train_start=2012,
    train_end=2021,
    test_start=2022,
    test_end=2023
):
    """Main function to run the prediction pipeline."""
    # Validate year ranges
    if train_start > train_end:
        logger.error("Training start year must be less than or equal to end year")
        raise ValueError("Invalid training year range")
    if test_start > test_end:
        logger.error("Testing start year must be less than or equal to end year")
        raise ValueError("Invalid testing year range")
    if test_start <= train_end:
        logger.error("Test period must not overlap with training period")
        raise ValueError("Test period overlaps with training period")
    
    # Load and process data
    df = load_and_validate_data(input_path)
    explore_data(df)
    
    # Filter training and testing data
    train_df = df[(df['muhurat_date'].dt.year >= train_start) & (df['muhurat_date'].dt.year <= train_end)]
    test_df = df[(df['muhurat_date'].dt.year >= test_start) & (df['muhurat_date'].dt.year <= test_end)]
    
    if train_df.empty or test_df.empty:
        logger.error("Insufficient data for training or testing")
        return
    
    train_muhurat_dates = train_df.index.drop_duplicates().sort_values()
    test_muhurat_dates = test_df.index.drop_duplicates().sort_values()
    
    # Compute returns and identify best/worst times
    returns_by_time, skipped_train_dates, single_price_train = compute_intraday_return_by_time(train_df, train_muhurat_dates)
    if not returns_by_time:
        logger.error("Failed to compute return statistics")
        return
    
    logger.info(f"Skipped training dates: {skipped_train_dates}/{len(train_muhurat_dates)}")
    logger.info(f"Single price times in training data: {single_price_train}")
    
    best_time = list(returns_by_time.keys())[0]
    worst_time = list(returns_by_time.keys())[-1]
    logger.info(f"Best time (Train): {best_time}")
    logger.info(f"Worst time (Train): {worst_time}")
    
    # Predict entry/exit points
    predictions, skipped_test_dates, single_price_test = predict_entry_exit(test_df, test_muhurat_dates, best_time, worst_time)
    save_predictions(predictions, output_path, test_muhurat_dates, skipped_test_dates, single_price_test)

if __name__ == "__main__":
    main()