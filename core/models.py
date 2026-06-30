import pandas as pd
import numpy as np
from prophet import Prophet
from xgboost import XGBRegressor
from core.data import fetch_stock_data, add_technical_features
from core.cache import is_model_cached, save_model, load_model

FEATURE_COLS = ['sma_5', 'sma_20', 'returns', 'lag_1', 'lag_7', 'prophet_trend']

def train_hybrid_model(ticker: str) -> dict:
    """
    Downloads historical data, trains both Prophet and XGBoost,
    and caches the resulting model dictionary.
    """
    # 1. Fetch raw data (last 2 years of daily close prices)
    raw_df = fetch_stock_data(ticker, period="2y")
    
    # 2. Fit Prophet model on history
    prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    # Silent fitting outputs to keep console clean
    import logging
    logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
    prophet_model.fit(raw_df)
    
    # 3. Extract Prophet forecasts for historical data
    prophet_forecast = prophet_model.predict(raw_df)
    
    # 4. Prepare data for XGBoost (add technical indicators and Prophet trend)
    df_with_features = add_technical_features(raw_df)
    
    # Align Prophet trend outputs with feature DataFrame
    # Merge on date 'ds'
    df_merged = pd.merge(df_with_features, prophet_forecast[['ds', 'trend', 'yhat']], on='ds')
    df_merged = df_merged.rename(columns={'trend': 'prophet_trend', 'yhat': 'prophet_yhat'})
    
    # Extract training inputs and target
    X_train = df_merged[FEATURE_COLS]
    y_train = df_merged['y']
    
    # 5. Fit XGBoost Regressor
    xgb_model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )
    xgb_model.fit(X_train, y_train)
    
    # 6. Bundle models and required metadata
    model_data = {
        "ticker": ticker,
        "prophet_model": prophet_model,
        "xgb_model": xgb_model,
        "historical_df": df_merged[['ds', 'y']], # Need this to bootstrap autoregressive prediction
        "feature_cols": FEATURE_COLS
    }
    
    # Save to disk cache
    save_model(ticker, model_data)
    return model_data

def get_forecast(ticker: str, days_to_predict: int = 30) -> pd.DataFrame:
    """
    Loads the cached hybrid model (or trains it if missing/expired)
    and predicts the price for future dates using a step-by-step loop.
    """
    ticker = ticker.upper().strip()
    
    # Load model (train if not cached)
    if not is_model_cached(ticker):
        model_data = train_hybrid_model(ticker)
    else:
        model_data = load_model(ticker)
        
    prophet_model = model_data["prophet_model"]
    xgb_model = model_data["xgb_model"]
    historical_df = model_data["historical_df"].copy()
    
    # 1. Run Prophet into the future to get future trend lines
    future_dates = prophet_model.make_future_dataframe(periods=days_to_predict)
    prophet_forecast = prophet_model.predict(future_dates)
    
    # We only care about the future predicted dates
    future_rows = prophet_forecast.tail(days_to_predict)[['ds', 'trend', 'yhat']].copy()
    future_rows = future_rows.rename(columns={'trend': 'prophet_trend', 'yhat': 'prophet_yhat'}).reset_index(drop=True)
    
    # Keep track of running history to compute lagging indicators step-by-step
    running_history = historical_df.copy()
    
    future_predictions = []
    
    # 2. Autoregressive prediction loop (predict Day 1, use that to predict Day 2, etc.)
    for idx, row in future_rows.iterrows():
        current_date = row['ds']
        prophet_trend = row['prophet_trend']
        prophet_yhat = row['prophet_yhat']
        
        # Calculate features using the latest values in running_history
        lag_1 = running_history['y'].iloc[-1]
        lag_7 = running_history['y'].iloc[-7] if len(running_history) >= 7 else running_history['y'].mean()
        sma_5 = running_history['y'].tail(5).mean()
        sma_20 = running_history['y'].tail(20).mean() if len(running_history) >= 20 else running_history['y'].mean()
        
        # Calculate returns
        prev_price = running_history['y'].iloc[-2] if len(running_history) >= 2 else lag_1
        returns = (lag_1 - prev_price) / prev_price if prev_price != 0 else 0
        
        # Format feature row for XGBoost
        xgb_features = pd.DataFrame([{
            'sma_5': sma_5,
            'sma_20': sma_20,
            'returns': returns,
            'lag_1': lag_1,
            'lag_7': lag_7,
            'prophet_trend': prophet_trend
        }])
        
        # Run XGBoost prediction
        xgb_pred = float(xgb_model.predict(xgb_features[FEATURE_COLS])[0])
        
        # Ensemble Prediction (average of Prophet and XGBoost)
        # This reduces variance and combines trend (Prophet) with momentum (XGBoost)
        hybrid_pred = (prophet_yhat + xgb_pred) / 2
        
        # Append the new prediction to running_history so future loops can lag it
        new_row = pd.DataFrame({'ds': [current_date], 'y': [hybrid_pred]})
        running_history = pd.concat([running_history, new_row], ignore_index=True)
        
        # Save results
        future_predictions.append({
            "ds": current_date,
            "prophet_val": prophet_yhat,
            "xgb_val": xgb_pred,
            "hybrid_val": hybrid_pred
        })
        
    return pd.DataFrame(future_predictions)
