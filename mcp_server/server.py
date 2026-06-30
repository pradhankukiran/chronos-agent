import os
import sys
from mcp.server.fastmcp import FastMCP

# Add the project root to path so we can import from core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models import get_forecast
from core.data import fetch_stock_data

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

if __name__ == "__main__":
    # If PORT is specified in the environment, run as SSE (for cloud deployment)
    if "PORT" in os.environ:
        mcp.run(transport="sse")
    else:
        # Start the MCP server using standard input/output (stdio) communication
        mcp.run()
