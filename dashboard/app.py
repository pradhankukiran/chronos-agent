import os
import sys
import pandas as pd
import streamlit as st
import datetime

# Add the project root to path so we can import from core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models import get_forecast
from core.data import fetch_stock_data

# Set Page Config
st.set_page_config(
    page_title="ChronosAgent - UK GOV Design System",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom GDS styling injection
st.markdown("""
<style>
    /* Import standard sans-serif font */
    html, body, [class*="css"] {
        font-family: "GDS Transport", Arial, sans-serif !important;
        background-color: #f3f2f1 !important; /* GDS Light Grey background */
        color: #0b0c0c !important; /* GDS Dark Grey text */
    }
    
    /* GDS Header Banner styling */
    .gds-header {
        background-color: #0b0c0c;
        padding: 10px 20px;
        color: white;
        margin-bottom: 30px;
        border-bottom: 10px solid #00a33b; /* Accent green line */
        display: flex;
        align-items: center;
    }
    .gds-header h1 {
        color: white !important;
        font-size: 24px !important;
        font-weight: bold !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    .gds-header span {
        background-color: #00a33b;
        color: white;
        font-size: 12px;
        font-weight: bold;
        padding: 2px 8px;
        margin-left: 15px;
        text-transform: uppercase;
    }

    /* GDS Container Card styling */
    .gds-card {
        background-color: #ffffff;
        border: 2px solid #b1b4b6;
        padding: 20px;
        margin-bottom: 25px;
    }

    /* GDS Callout / Warning box (Thick blue border) */
    .gds-callout {
        border-left: 10px solid #1d70b8; /* GDS Blue */
        background-color: #f3f2f1;
        padding: 15px;
        margin-bottom: 20px;
    }

    /* GDS Button override */
    div.stButton > button {
        background-color: #00703c !important; /* GDS Green */
        color: white !important;
        border: none !important;
        padding: 10px 20px !important;
        font-weight: bold !important;
        border-radius: 0px !important; /* Sharp corners */
        border-bottom: 3px solid #005a30 !important;
        font-size: 16px !important;
    }
    div.stButton > button:hover {
        background-color: #005a30 !important;
        cursor: pointer !important;
    }
    div.stButton > button:focus {
        background-color: #ffdd00 !important; /* GDS Yellow Focus */
        color: #0b0c0c !important;
        border: 3px solid #0b0c0c !important;
    }
    
    /* Input border formatting */
    input {
        border: 2px solid #0b0c0c !important;
        border-radius: 0px !important;
    }
</style>
""", unsafe_allow_html=True)

# 1. Render GOV.UK Header Bar
st.markdown("""
<div class="gds-header">
    <h1>CHRONOS.AGY</h1>
    <span>Alpha Prototype</span>
</div>
""", unsafe_allow_html=True)

# 2. Main Layout - Split into Sidebar panel and Main Panel
col_side, col_main = st.columns([1, 3])

with col_side:
    st.markdown('<div class="gds-card">', unsafe_allow_html=True)
    st.subheader("Configure Parameters")
    
    # Stock Ticker input (Capitalized automatically)
    ticker = st.text_input("Enter Ticker Symbol", value="AAPL", help="e.g., AAPL, MSFT, BTC-USD").upper().strip()
    
    # Days to Predict (GDS elements use clean sliders or text inputs)
    days_to_predict = st.slider("Forecast Horizon (Days)", min_value=5, max_value=90, value=30)
    
    # Run button
    run_clicked = st.button("Generate Forecast")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # System Info box
    st.markdown("""
    <div class="gds-callout">
        <strong>System Information:</strong><br>
        This prototype uses a hybrid ensemble of <strong>Meta's Prophet procedure</strong> and <strong>XGBoost Regressor</strong>, executing model updates in a cached environment.
    </div>
    """, unsafe_allow_html=True)

with col_main:
    if run_clicked or ticker:
        with st.spinner("Processing forecasting calculations..."):
            try:
                # Get historical prices for context metrics
                hist_data = fetch_stock_data(ticker, period="1mo")
                current_price = hist_data['y'].iloc[-1]
                
                # Fetch prediction from models module (leverages 24h caching)
                forecast_df = get_forecast(ticker, days_to_predict=days_to_predict)
                
                final_predicted_price = forecast_df['hybrid_val'].iloc[-1]
                price_change = final_predicted_price - current_price
                pct_change = (price_change / current_price) * 100
                
                # Render GDS style Metrics
                st.markdown('<div class="gds-card">', unsafe_allow_html=True)
                st.subheader("Key Findings")
                
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                with metric_col1:
                    st.metric("Current Price", f"${current_price:.2f}")
                with metric_col2:
                    st.metric(f"Forecast Price ({days_to_predict} Days)", f"${final_predicted_price:.2f}")
                with metric_col3:
                    st.metric("Predicted Change", f"{pct_change:+.2f}%", delta_color="normal")
                    
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Render Visual Forecast Plot
                st.markdown('<div class="gds-card">', unsafe_allow_html=True)
                st.subheader("Historical vs. Forecast Trend Chart")
                
                # Combine historical and forecasted data for a single clean chart
                hist_plot = hist_data.tail(30)[['ds', 'y']].copy()
                hist_plot.columns = ['ds', 'Actual Price']
                
                forecast_plot = forecast_df[['ds', 'hybrid_val', 'prophet_val', 'xgb_val']].copy()
                forecast_plot.columns = ['ds', 'Hybrid Prediction', 'Prophet Component', 'XGBoost Component']
                
                # Merge datasets for plotting
                combined_plot = pd.merge(hist_plot, forecast_plot, on='ds', how='outer')
                combined_plot = combined_plot.set_index('ds')
                
                # Line chart
                st.line_chart(combined_plot, color=["#000000", "#1d70b8", "#f44336", "#4caf50"])
                st.markdown("<p style='font-size:12px;color:#505a5f;'>Chart showing Actual Price (black), Hybrid Prediction (blue), Prophet Component (red), and XGBoost Component (green).</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Raw data table in GDS style
                st.markdown('<div class="gds-card">', unsafe_allow_html=True)
                st.subheader("Forecast Raw Output Data")
                
                # Clean up formatting for readability
                display_df = forecast_df.copy()
                display_df.columns = ['Date', 'Prophet Component', 'XGBoost Component', 'Hybrid Prediction']
                display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
                
                # Display table
                st.dataframe(
                    display_df.style.format({
                        'Prophet Component': '${:.2f}',
                        'XGBoost Component': '${:.2f}',
                        'Hybrid Prediction': '${:.2f}'
                    }),
                    use_container_width=True,
                    hide_index=True
                )
                st.markdown("</div>", unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error fetching/forecasting ticker '{ticker}': {str(e)}")
