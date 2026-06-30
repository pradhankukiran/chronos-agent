import os
import sys
from mcp.server.fastmcp import FastMCP

# Add the project root to path so we can import from core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models import get_forecast, get_generic_forecast
from core.data import fetch_stock_data, fetch_weather_data, fetch_economic_data

# Initialize FastMCP Server
mcp = FastMCP(
    "ChronosAgent",
    host=os.environ.get("FASTMCP_HOST", "0.0.0.0"),
    port=int(os.environ.get("PORT", os.environ.get("FASTMCP_PORT", 8000)))
)

@mcp.tool()
def forecast_asset_price(ticker: str, days_to_predict: int = 30) -> str:
    """
    Downloads historical stock or cryptocurrency prices from Yahoo Finance,
    applies a hybrid Prophet + XGBoost forecasting model, and returns predicted 
    future prices for the specified number of days.

    Args:
        ticker: The stock or crypto symbol (e.g., 'AAPL', 'TSLA', 'BTC-USD').
        days_to_predict: Number of days into the future to forecast (default is 30, max 90).
    """
    ticker = ticker.upper().strip()
    days_to_predict = min(max(int(days_to_predict), 5), 90) # Bound between 5 and 90 days
    
    try:
        # Fetch current price for context
        hist_df = fetch_stock_data(ticker, period="5d", verify_len=False)
        current_price = hist_df['y'].iloc[-1]
        
        # Get hybrid forecast (leverages 24h file cache)
        forecast_df = get_forecast(ticker, days_to_predict=days_to_predict)
        
        # Build a text report for the LLM chatbot
        lines = [
            f"### ChronosAgent Forecast Report: {ticker}",
            f"Current Stock/Crypto Price: ${current_price:.2f}",
            f"Forecast Horizon: {days_to_predict} days",
            "",
            "| Date | Hybrid Predicted Price | Prophet Component (Trend) | XGBoost Component (Momentum) |",
            "| :--- | :--- | :--- | :--- |"
        ]
        
        for _, row in forecast_df.iterrows():
            date_str = row['ds'].strftime('%Y-%m-%d')
            lines.append(
                f"| {date_str} | ${row['hybrid_val']:.2f} | ${row['prophet_val']:.2f} | ${row['xgb_val']:.2f} |"
            )
            
        return "\n".join(lines)
        
    except Exception as e:
        return f"Error executing forecast for ticker '{ticker}': {str(e)}"

@mcp.tool()
def forecast_weather(latitude: float, longitude: float, days_to_predict: int = 30) -> str:
    """
    Downloads historical maximum daily temperatures from Open-Meteo,
    applies the hybrid Prophet + XGBoost forecasting model, and returns predicted 
    temperatures for the specified number of days.

    Args:
        latitude: Latitude coordinate of the location.
        longitude: Longitude coordinate of the location.
        days_to_predict: Number of days into the future to forecast (default is 30, max 90).
    """
    days_to_predict = min(max(int(days_to_predict), 5), 90)
    cache_key = f"WEATHER_{latitude:.4f}_{longitude:.4f}"
    try:
        forecast_df = get_generic_forecast(
            cache_key, 
            days_to_predict, 
            lambda: fetch_weather_data(latitude, longitude)
        )
        
        lines = [
            f"### ChronosAgent Weather Forecast Report",
            f"Coordinates: ({latitude:.4f}, {longitude:.4f})",
            f"Forecast Horizon: {days_to_predict} days",
            "",
            "| Date | Hybrid Predicted Temp (°C) | Prophet Component | XGBoost Component |",
            "| :--- | :--- | :--- | :--- |"
        ]
        for _, row in forecast_df.iterrows():
            date_str = row['ds'].strftime('%Y-%m-%d')
            lines.append(
                f"| {date_str} | {row['hybrid_val']:.1f}°C | {row['prophet_val']:.1f}°C | {row['xgb_val']:.1f}°C |"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"Error executing weather forecast for ({latitude}, {longitude}): {str(e)}"

@mcp.tool()
def forecast_economic_indicator(series_id: str, days_to_predict: int = 30) -> str:
    """
    Downloads historical macroeconomic data from FRED using a Series ID,
    applies the hybrid Prophet + XGBoost forecasting model, and returns predictions 
    for the specified number of days.

    Args:
        series_id: FRED Series ID (e.g., 'GDP', 'UNRATE', 'CPIAUCSL').
        days_to_predict: Number of days into the future to forecast (default is 30, max 90).
    """
    series_id = series_id.upper().strip()
    days_to_predict = min(max(int(days_to_predict), 5), 90)
    try:
        forecast_df = get_generic_forecast(
            series_id, 
            days_to_predict, 
            lambda: fetch_economic_data(series_id)
        )
        
        lines = [
            f"### ChronosAgent Economic Forecast Report: {series_id}",
            f"Forecast Horizon: {days_to_predict} days",
            "",
            "| Date | Hybrid Predicted Value | Prophet Component | XGBoost Component |",
            "| :--- | :--- | :--- | :--- |"
        ]
        for _, row in forecast_df.iterrows():
            date_str = row['ds'].strftime('%Y-%m-%d')
            lines.append(
                f"| {date_str} | {row['hybrid_val']:.4f} | {row['prophet_val']:.4f} | {row['xgb_val']:.4f} |"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"Error executing economic forecast for series '{series_id}': {str(e)}"

if __name__ == "__main__":
    # If PORT is specified in the environment, run as SSE (for cloud deployment)
    if "PORT" in os.environ:
        mcp.run(transport="sse")
    else:
        # Start the MCP server using standard input/output (stdio) communication
        mcp.run()
