import os
import sys
import pandas as pd
import streamlit as st

# Add the project root to path so we can import from core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models import get_forecast
from core.data import fetch_stock_data

# Set Page Config
st.set_page_config(
    page_title="ChronosAgent - GOV.UK",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Deep override of Streamlit internals to enforce UK GDS specifications
st.markdown("""
<style>
    /* 1. Global Font and Background overrides */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stApp"] {
        font-family: Arial, sans-serif !important;
        background-color: #f3f2f1 !important; /* GDS light grey background */
        color: #0b0c0c !important; /* GDS text black */
    }

    /* Remove default Streamlit top header line and icons */
    [data-testid="stHeader"] {
        background-color: #0b0c0c !important;
        height: 50px !important;
        border-bottom: 10px solid #008543 !important; /* GOV.UK Green line */
    }
    
    /* 2. UK GOV.UK Header Styling */
    .govuk-header-container {
        background-color: #0b0c0c;
        padding: 12px 24px;
        border-bottom: 10px solid #008543;
        margin-bottom: 25px;
        display: flex;
        align-items: center;
    }
    .govuk-header-title {
        color: #ffffff !important;
        font-size: 24px !important;
        font-weight: 700 !important;
        text-decoration: none;
        margin: 0 !important;
        letter-spacing: -1px;
    }
    .govuk-phase-tag {
        display: inline-block;
        outline: 2px solid #ffffff;
        color: #ffffff;
        background-color: #005a30;
        font-size: 12px;
        font-weight: 700;
        padding: 2px 8px;
        margin-left: 15px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* 3. GDS Layout Containers (Panels) */
    .govuk-main-wrapper {
        padding: 15px;
    }
    
    /* GDS Style Section Card */
    .govuk-panel {
        background-color: #ffffff !important;
        border: 2px solid #0b0c0c !important; /* Thick black border */
        padding: 24px !important;
        margin-bottom: 24px !important;
        border-radius: 0px !important; /* Enforce sharp corners */
    }

    /* GDS Signature Inset Text Callout (Blue left border) */
    .govuk-inset-text {
        border-left: 10px solid #1d70b8 !important; /* GDS Blue */
        background-color: #f3f2f1 !important;
        padding: 15px !important;
        margin-top: 15px !important;
        margin-bottom: 15px !important;
        color: #0b0c0c !important;
        font-size: 16px !important;
    }

    /* 4. Streamlit Widget Overrides (Buttons, Inputs, Selectors) */
    
    /* GDS Primary green button with flat sharp corners */
    div.stButton > button {
        background-color: #008543 !important; /* GDS Green */
        color: #ffffff !important;
        font-size: 18px !important;
        font-weight: 700 !important;
        padding: 10px 20px !important;
        border: none !important;
        border-radius: 0px !important;
        border-bottom: 4px solid #005a30 !important; /* Darker bottom edge shadow */
        box-shadow: none !important;
        width: 100% !important;
    }
    div.stButton > button:hover {
        background-color: #005a30 !important;
        cursor: pointer !important;
    }
    div.stButton > button:active, div.stButton > button:focus {
        background-color: #ffdd00 !important; /* GDS Yellow Focus */
        color: #0b0c0c !important;
        border: 3px solid #0b0c0c !important;
        outline: none !important;
    }

    /* GDS Text Input formatting */
    div[data-testid="stTextInput"] input {
        border: 2px solid #0b0c0c !important;
        border-radius: 0px !important;
        font-size: 18px !important;
        color: #0b0c0c !important;
        padding: 8px !important;
    }
    div[data-testid="stTextInput"] input:focus {
        outline: 3px solid #ffdd00 !important; /* GDS Yellow highlight */
        outline-offset: 0px !important;
        border: 2px solid #0b0c0c !important;
    }

    /* GDS Metrics overrides */
    div[data-testid="stMetric"] {
        border: 2px solid #b1b4b6 !important;
        background-color: #f8f8f8 !important;
        padding: 15px !important;
        border-radius: 0px !important;
    }
    div[data-testid="stMetricValue"] > div {
        font-size: 36px !important;
        font-weight: 700 !important;
        color: #0b0c0c !important;
    }
    div[data-testid="stMetricLabel"] > div {
        font-size: 16px !important;
        font-weight: 700 !important;
        color: #505a5f !important;
        text-transform: uppercase;
    }

    /* Enforce bold sans-serif GDS headings */
    h1, h2, h3, h4 {
        font-family: Arial, sans-serif !important;
        font-weight: 700 !important;
        color: #0b0c0c !important;
        margin-top: 0px !important;
    }
    h2 {
        font-size: 28px !important;
        border-bottom: 4px solid #0b0c0c !important; /* Thick GDS divider line */
        padding-bottom: 8px !important;
        margin-bottom: 20px !important;
    }
    h3 {
        font-size: 20px !important;
    }
    
    /* GDS Tables overrides */
    div[data-testid="stDataFrame"] {
        border: 2px solid #0b0c0c !important;
        border-radius: 0px !important;
    }
</style>
""", unsafe_allow_html=True)

# 1. Custom GOV.UK Header Bar
st.markdown("""
<div class="govuk-header-container">
    <a href="#" class="govuk-header-title">GOV.UK</a>
    <span class="govuk-header-title" style="font-weight: 300; margin-left: 10px;">| Chronos Forecasting Service</span>
    <span class="govuk-phase-tag">Alpha</span>
</div>
""", unsafe_allow_html=True)

# 2. Grid Layout setup
col_side, col_main = st.columns([1, 2.5])

with col_side:
    st.markdown('<div class="govuk-panel">', unsafe_allow_html=True)
    st.markdown("<h3>1. Choose asset parameters</h3>", unsafe_allow_html=True)
    
    # Input field
    ticker = st.text_input("Asset Ticker Symbol", value="AAPL", help="Enter stock (e.g. AAPL) or crypto (e.g. BTC-USD)").upper().strip()
    
    # Slider
    days_to_predict = st.slider("Prediction Horizon (Days)", min_value=5, max_value=90, value=30)
    
    # Submit button
    run_clicked = st.button("Calculate Forecast")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # GDS Inset callout text
    st.markdown("""
    <div class="govuk-inset-text">
        <strong>Information:</strong> This calculations engine utilizes a hybrid stacking regressor comprising a Bayesian generalized additive model (Prophet) and a gradient boosting regressor (XGBoost).
    </div>
    """, unsafe_allow_html=True)

with col_main:
    if run_clicked or ticker:
        with st.spinner("Executing statistical calculations..."):
            try:
                # Fetch data
                hist_data = fetch_stock_data(ticker, period="1mo")
                current_price = hist_data['y'].iloc[-1]
                
                # Fetch predictions
                forecast_df = get_forecast(ticker, days_to_predict=days_to_predict)
                
                final_predicted_price = forecast_df['hybrid_val'].iloc[-1]
                price_change = final_predicted_price - current_price
                pct_change = (price_change / current_price) * 100
                
                # Output metrics block
                st.markdown('<div class="govuk-panel">', unsafe_allow_html=True)
                st.markdown("<h2>Forecasting Summary</h2>", unsafe_allow_html=True)
                
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Current Asset Value", f"${current_price:.2f}")
                with m2:
                    st.metric(f"Forecasted Value ({days_to_predict} Days)", f"${final_predicted_price:.2f}")
                with m3:
                    st.metric("Estimated Change (%)", f"{pct_change:+.2f}%")
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Output visual chart block
                st.markdown('<div class="govuk-panel">', unsafe_allow_html=True)
                st.markdown("<h2>Analytical Forecast Trend Line</h2>", unsafe_allow_html=True)
                
                hist_plot = hist_data.tail(30)[['ds', 'y']].copy()
                hist_plot.columns = ['ds', 'Actual Value']
                
                forecast_plot = forecast_df[['ds', 'hybrid_val', 'prophet_val', 'xgb_val']].copy()
                forecast_plot.columns = ['ds', 'Hybrid Prediction', 'Prophet Trend', 'XGBoost Momentum']
                
                combined_plot = pd.merge(hist_plot, forecast_plot, on='ds', how='outer').set_index('ds')
                
                # GDS high-contrast colors (Black for actuals, GDS Blue for predictions, red/green accents)
                st.line_chart(combined_plot, color=["#0b0c0c", "#1d70b8", "#d4351c", "#00703c"])
                
                st.markdown("<p style='font-size: 14px; color: #505a5f; margin-top: 10px;'>Figure 1: Historic daily actual value (black) compared with the generated hybrid ensemble model projection (blue), Prophet component (red), and XGBoost component (green).</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Output data table block
                st.markdown('<div class="govuk-panel">', unsafe_allow_html=True)
                st.markdown("<h2>Calculated Tabular Outputs</h2>", unsafe_allow_html=True)
                
                display_df = forecast_df.copy()
                display_df.columns = ['Calculation Date', 'Prophet Baseline ($)', 'XGBoost Offset ($)', 'Combined Prediction ($)']
                display_df['Calculation Date'] = display_df['Calculation Date'].dt.strftime('%Y-%m-%d')
                
                st.dataframe(
                    display_df.style.format({
                        'Prophet Baseline ($)': '{:.2f}',
                        'XGBoost Offset ($)': '{:.2f}',
                        'Combined Prediction ($)': '{:.2f}'
                    }),
                    use_container_width=True,
                    hide_index=True
                )
                st.markdown("</div>", unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Failed to generate calculation: {str(e)}")
