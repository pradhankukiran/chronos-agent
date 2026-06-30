import pandas as pd
import yfinance as yf
import numpy as np

def fetch_stock_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    """
    Downloads historical stock/crypto daily data from Yahoo Finance.
    Returns a formatted pandas DataFrame ready for Prophet and XGBoost.
    """
    ticker = ticker.upper().strip()
    # Download daily history
    raw_data = yf.download(ticker, period=period)
    
    if raw_data.empty:
        raise ValueError(f"No historical data found for ticker '{ticker}'")
    
    # Format to a standard DataFrame
    df = raw_data.reset_index()
    
    # Select only the Date and Close price columns
    df = df[['Date', 'Close']].copy()
    
    # Rename columns to Prophet's standard format (ds: date, y: value)
    df.columns = ['ds', 'y']
    
    # Remove timezone info if present
    if df['ds'].dt.tz is not None:
        df['ds'] = df['ds'].dt.tz_localize(None)
        
    # Ensure values are float64 type
    df['y'] = df['y'].astype(float)
    
    # Verify we have enough history to calculate indicators (like 20-day SMA)
    if len(df) < 20:
        raise ValueError(
            f"Ticker '{ticker}' has only {len(df)} days of history. "
            f"A minimum of 20 trading days is required to calculate indicators."
        )
    
    return df

def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates technical indicators and lag features for the XGBoost model.
    """
    df = df.copy()
    
    # 1. Simple Moving Averages (SMA)
    df['sma_5'] = df['y'].rolling(window=5).mean()
    df['sma_20'] = df['y'].rolling(window=20).mean()
    
    # 2. Daily Returns (percentage change day-over-day)
    df['returns'] = df['y'].pct_change()
    
    # 3. Lags (past prices)
    df['lag_1'] = df['y'].shift(1)
    df['lag_7'] = df['y'].shift(7)
    
    # Drop rows with NaN values created by rolling windows and lags
    df = df.dropna().reset_index(drop=True)
    
    return df
