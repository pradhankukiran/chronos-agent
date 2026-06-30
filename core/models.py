import pandas as pd
import numpy as np
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from core.data import fetch_stock_data, add_technical_features
from core.cache import is_model_cached, save_model, load_model

FEATURE_COLS = ['sma_5', 'sma_20', 'returns', 'lag_1', 'lag_7', 'prophet_trend']

def train_generic_hybrid_model(raw_df: pd.DataFrame, name: str) -> dict:
    """
    Trains a Prophet + XGBoost hybrid ensemble model on a generic DataFrame (ds, y),
    calculates diagnostics, and caches the model data.
    """
    name = name.upper().strip()
    
    # 1. Fit Prophet model on history
    prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    import logging
    logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
    prophet_model.fit(raw_df)
    
    # Extract Prophet forecasts for historical data
    prophet_forecast = prophet_model.predict(raw_df)
    
    # 2. Prepare data for XGBoost (add technical indicators and Prophet trend)
    df_with_features = add_technical_features(raw_df)
    
    # Merge datasets
    df_merged = pd.merge(df_with_features, prophet_forecast[['ds', 'trend', 'yhat']], on='ds')
    df_merged = df_merged.rename(columns={'trend': 'prophet_trend', 'yhat': 'prophet_yhat'})
    
    X = df_merged[FEATURE_COLS]
    y = df_merged['y']
    
    # --- CALCULATE BACKTESTING ACCURACY (Train-Test Split) ---
    X_train_val, X_test_val, y_train_val, y_test_val = train_test_split(X, y, test_size=0.15, shuffle=False)
    
    # Temporary validation XGBoost fit
    val_xgb = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
    val_xgb.fit(X_train_val, y_train_val)
    
    # Combine Prophet and XGBoost validation predictions
    val_preds_xgb = val_xgb.predict(X_test_val)
    val_preds_prophet = df_merged.loc[y_test_val.index, 'prophet_yhat'].values
    val_hybrid_preds = (val_preds_prophet + val_preds_xgb) / 2
    
    # Metrics
    backtest_r2 = r2_score(y_test_val, val_hybrid_preds)
    backtest_mae = mean_absolute_error(y_test_val, val_hybrid_preds)
    accuracy_rating = max(0.0, min(100.0, backtest_r2 * 100)) # Represented as %
    
    # 3. Fit Full XGBoost Regressor on all data
    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
    xgb_model.fit(X, y)
    
    # 4. Calculate Diagnostics & Insights
    # Feature Importance
    importances = xgb_model.feature_importances_
    feature_importance = {feat: float(imp) for feat, imp in zip(FEATURE_COLS, importances)}
    
    # Autocorrelation (Durbin-Watson) on Full residuals
    full_preds = (df_merged['prophet_yhat'].values + xgb_model.predict(X)) / 2
    residuals = y.values - full_preds
    diff_res = np.diff(residuals)
    dw_stat = float(np.sum(diff_res**2) / np.sum(residuals**2)) if np.sum(residuals**2) > 0 else 2.0
    residual_std = float(np.std(residuals))
    avg_offset = float(np.mean(np.abs(xgb_model.predict(X) - df_merged['prophet_yhat'].values)))
    
    # Extract Prophet Changepoints (top 3 by magnitude)
    deltas = prophet_model.params['delta'].mean(axis=0)
    top_cp_indices = np.argsort(np.abs(deltas))[-3:][::-1]
    changepoints = []
    for idx in top_cp_indices:
        if idx < len(prophet_model.changepoints):
            cp_date = prophet_model.changepoints.iloc[idx].strftime('%Y-%m-%d')
            cp_val = float(deltas[idx])
            changepoints.append({"date": cp_date, "delta": cp_val})
            
    # Extract Seasonality Peaks (Weekly/Yearly)
    # Weekly peak
    weekly_days = pd.DataFrame({'ds': pd.date_range('2026-06-01', periods=7)}) # Monday-Sunday
    weekly_res = prophet_model.predict(weekly_days)
    days_list = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    best_day = days_list[np.argmax(weekly_res['weekly'].values)]
    worst_day = days_list[np.argmin(weekly_res['weekly'].values)]
    
    # Yearly peak
    yearly_months = pd.DataFrame({'ds': pd.date_range('2026-01-01', periods=12, freq='MS')})
    yearly_res = prophet_model.predict(yearly_months)
    months_list = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    best_month = months_list[np.argmax(yearly_res['yearly'].values)]
    worst_month = months_list[np.argmin(yearly_res['yearly'].values)]
    
    # Bundle models and diagnostic metadata
    model_data = {
        "ticker": name,
        "prophet_model": prophet_model,
        "xgb_model": xgb_model,
        "historical_df": df_merged[['ds', 'y']],
        "feature_cols": FEATURE_COLS,
        "diagnostics": {
            "accuracy_rating": accuracy_rating,
            "backtest_mae": backtest_mae,
            "durbin_watson": dw_stat,
            "residual_std": residual_std,
            "avg_offset": avg_offset,
            "feature_importance": feature_importance,
            "changepoints": changepoints,
            "weekly_best": best_day,
            "weekly_worst": worst_day,
            "yearly_best": best_month,
            "yearly_worst": worst_month
        }
    }
    
    save_model(name, model_data)
    return model_data

def get_generic_forecast(name: str, days_to_predict: int, df_fetch_func) -> pd.DataFrame:
    """
    Loads the cached hybrid model (or downloads data and trains if missing/expired)
    and predicts the price for future dates using an autoregressive loop.
    """
    name = name.upper().strip()
    
    # Load model (train if not cached)
    if not is_model_cached(name):
        raw_df = df_fetch_func()
        model_data = train_generic_hybrid_model(raw_df, name)
    else:
        model_data = load_model(name)
        
    prophet_model = model_data["prophet_model"]
    xgb_model = model_data["xgb_model"]
    historical_df = model_data["historical_df"].copy()
    
    # 1. Run Prophet into the future to get future trend lines
    future_dates = prophet_model.make_future_dataframe(periods=days_to_predict)
    prophet_forecast = prophet_model.predict(future_dates)
    
    # We only care about the future predicted dates (including uncertainty bounds)
    future_rows = prophet_forecast.tail(days_to_predict)[['ds', 'trend', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    future_rows = future_rows.rename(
        columns={
            'trend': 'prophet_trend',
            'yhat': 'prophet_yhat',
            'yhat_lower': 'prophet_lower',
            'yhat_upper': 'prophet_upper'
        }
    ).reset_index(drop=True)
    
    # Keep track of running history to compute lagging indicators step-by-step
    running_history = historical_df.copy()
    
    future_predictions = []
    
    # 2. Autoregressive prediction loop
    for idx, row in future_rows.iterrows():
        current_date = row['ds']
        prophet_trend = row['prophet_trend']
        prophet_yhat = row['prophet_yhat']
        prophet_lower = row['prophet_lower']
        prophet_upper = row['prophet_upper']
        
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
        
        # Ensemble Prediction
        hybrid_pred = (prophet_yhat + xgb_pred) / 2
        
        # Append the new prediction to running_history so future loops can lag it
        new_row = pd.DataFrame({'ds': [current_date], 'y': [hybrid_pred]})
        running_history = pd.concat([running_history, new_row], ignore_index=True)
        
        # Save results
        future_predictions.append({
            "ds": current_date,
            "prophet_val": prophet_yhat,
            "xgb_val": xgb_pred,
            "hybrid_val": hybrid_pred,
            "hybrid_lower": prophet_lower,
            "hybrid_upper": prophet_upper
        })
        
    return pd.DataFrame(future_predictions)

def get_forecast(ticker: str, days_to_predict: int = 30) -> pd.DataFrame:
    """
    Finance specific wrapper. Backward compatible.
    """
    return get_generic_forecast(ticker, days_to_predict, lambda: fetch_stock_data(ticker))
