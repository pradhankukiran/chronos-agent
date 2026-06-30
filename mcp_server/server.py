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
def forecast_weather(location: str, days_to_predict: int = 30) -> str:
    """
    Downloads historical maximum daily temperatures from Open-Meteo for a given
    city name or coordinates, applies the hybrid Prophet + XGBoost forecasting model,
    and returns predicted future temperatures.

    Args:
        location: City name (e.g. 'London', 'San Francisco') or coordinates (e.g. '37.7749, -122.4194').
        days_to_predict: Number of days into the future to forecast (default is 30, max 90).
    """
    import re
    import requests
    days_to_predict = min(max(int(days_to_predict), 5), 90)
    
    # 1. Check if input is coordinate format (lat, lon)
    coord_match = re.match(r"^\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*$", location)
    if coord_match:
        latitude = float(coord_match.group(1))
        longitude = float(coord_match.group(2))
        resolved_name = f"Coordinates ({latitude:.4f}, {longitude:.4f})"
    else:
        # 2. Resolve city name using Geocoding API
        geocode_url = f"https://geocoding-api.open-meteo.com/v1/search?name={requests.utils.quote(location)}&count=1&language=en&format=json"
        try:
            r = requests.get(geocode_url, timeout=5)
            r.raise_for_status()
            results = r.json().get("results", [])
            if not results:
                return f"Error: Could not resolve location name '{location}' to coordinates. Please try a more specific city/region name."
            first = results[0]
            latitude = first["latitude"]
            longitude = first["longitude"]
            
            # Format location label
            name = first.get("name")
            admin1 = first.get("admin1")
            country = first.get("country")
            resolved_name = f"{name}"
            if admin1:
                resolved_name += f", {admin1}"
            if country:
                resolved_name += f", {country}"
            resolved_name += f" ({latitude:.4f}, {longitude:.4f})"
        except Exception as e:
            return f"Error resolving geocoding for '{location}': {str(e)}"

    cache_key = f"WEATHER_{latitude:.4f}_{longitude:.4f}"
    try:
        forecast_df = get_generic_forecast(
            cache_key, 
            days_to_predict, 
            lambda: fetch_weather_data(latitude, longitude)
        )
        
        lines = [
            f"### ChronosAgent Weather Forecast Report: {resolved_name}",
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
        return f"Error executing weather forecast for '{resolved_name}': {str(e)}"

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
