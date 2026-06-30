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

def fetch_company_info(ticker: str) -> dict:
    """
    Retrieves key company metadata and financial indicators from Yahoo Finance.
    Returns a dictionary of cleaned metrics.
    """
    ticker = ticker.upper().strip()
    try:
        t = yf.Ticker(ticker)
        info = t.info
        
        # Format Market Cap to a readable string (e.g. 2.45 Trillion)
        cap = info.get("marketCap")
        if cap:
            if cap >= 1e12:
                cap_str = f"${cap / 1e12:.2f} Trillion"
            elif cap >= 1e9:
                cap_str = f"${cap / 1e9:.2f} Billion"
            else:
                cap_str = f"${cap / 1e6:.2f} Million"
        else:
            cap_str = "N/A"

        return {
            "name": info.get("longName") or info.get("shortName") or ticker,
            "summary": info.get("longBusinessSummary") or "Business summary is currently unavailable.",
            "sector": info.get("sector") or "N/A",
            "industry": info.get("industry") or "N/A",
            "employees": f"{info.get('fullTimeEmployees', 0):,}" if info.get("fullTimeEmployees") else "N/A",
            "market_cap": cap_str,
            "pe_ratio": f"{info.get('trailingPE'):.2f}" if info.get("trailingPE") else "N/A",
            "beta": f"{info.get('beta'):.2f}" if info.get("beta") else "N/A",
            "recommendation": info.get("recommendationKey", "N/A").replace("_", " ").title(),
            "high_52": info.get("fiftyTwoWeekHigh"),
            "low_52": info.get("fiftyTwoWeekLow")
        }
    except Exception:
        # Fallback dictionary if API call fails
        return {
            "name": ticker,
            "summary": "Financial metadata is currently unavailable for this asset.",
            "sector": "N/A",
            "industry": "N/A",
            "employees": "N/A",
            "market_cap": "N/A",
            "pe_ratio": "N/A",
            "beta": "N/A",
            "recommendation": "N/A",
            "high_52": None,
            "low_52": None
        }


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
